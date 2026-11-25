#!/usr/bin/env python3
"""Worker local de separação de 6 stems (HTDemucs 6s) para o Tab Transcriber."""

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

EXPECTED_STEMS = ("guitar", "bass", "piano", "vocals", "drums", "other")


def find_bin(name: str) -> str | None:
    for base in (Path(sys.prefix) / "bin", Path(sys.executable).parent):
        candidate = base / name
        if candidate.exists():
            return str(candidate)
    return shutil.which(name)


def has_module(mod_name: str) -> bool:
    return importlib.util.find_spec(mod_name) is not None


def probe_cli_help(base_cmd: list[str]) -> str:
    try:
        res = subprocess.run([*base_cmd, "--help"], capture_output=True, text=True, timeout=10)
        return (res.stdout or "") + "\n" + (res.stderr or "")
    except Exception:
        return ""


def run_demucs_mlx(input_wav: Path, tmp_dir: Path, model: str, quality: str) -> str:
    bin_path = find_bin("demucs-mlx")
    if bin_path:
        base_cmd = [bin_path]
    elif has_module("demucs_mlx"):
        base_cmd = [sys.executable, "-m", "demucs_mlx"]
    else:
        raise FileNotFoundError("demucs-mlx não encontrado")

    help_text = probe_cli_help(base_cmd)
    cmd = [*base_cmd, "-n", model, "-o", str(tmp_dir)]

    # Alta fidelidade: equivariância por shifts + overlap + saída 24-bit (somente se suportado pelo binário)
    if quality == "high":
        if "--shifts" in help_text:
            cmd.extend(["--shifts", "2"])
        if "--overlap" in help_text:
            cmd.extend(["--overlap", "0.25"])
    else:
        if "--shifts" in help_text:
            cmd.extend(["--shifts", "1"])

    if "--int24" in help_text:
        cmd.append("--int24")

    cmd.append(str(input_wav))

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"demucs-mlx falhou:\n{res.stderr.strip() or res.stdout.strip()}")

    tag = "HQ 24-bit" if quality == "high" else "Rápido"
    return f"demucs-mlx ({tag})"


def run_mlx_demucs(input_wav: Path, tmp_dir: Path, model: str, quality: str) -> str:
    bin_path = find_bin("mlx-demucs")
    if not bin_path:
        raise FileNotFoundError("mlx-demucs não encontrado")

    cmd = [bin_path, str(input_wav), "-m", model, "-o", str(tmp_dir)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"mlx-demucs falhou:\n{res.stderr.strip() or res.stdout.strip()}")
    return f"mlx-demucs ({quality})"


def run_demucs_infer(input_wav: Path, out_dir: Path, model: str) -> str:
    from demucs_infer import DemucsSession  # type: ignore
    import soundfile as sf  # type: ignore

    with DemucsSession(model=model, device="cpu") as session:
        _, stems = session.infer(str(input_wav))
        sr = getattr(session, "samplerate", 44100)
        for name, tensor in stems.items():
            arr = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else tensor
            if arr.ndim == 2 and arr.shape[0] <= 2:
                arr = arr.T
            sf.write(str(out_dir / f"{name}.wav"), arr, sr, subtype="PCM_24")
    return "demucs-infer (24-bit)"


def collect_stems(search_root: Path, out_dir: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for wav_path in search_root.rglob("*.wav"):
        stem_key = wav_path.stem.lower()
        if stem_key in EXPECTED_STEMS:
            dest = out_dir / f"{stem_key}.wav"
            if wav_path.resolve() != dest.resolve():
                shutil.copy2(wav_path, dest)
            found[stem_key] = str(dest)
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model", default="htdemucs_6s")
    parser.add_argument("--quality", default="high", choices=["fast", "high"])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        if find_bin("demucs-mlx") or has_module("demucs_mlx"):
            print(json.dumps({"ready": True, "engine": "demucs-mlx (Apple MLX)"}))
            return 0
        if find_bin("mlx-demucs"):
            print(json.dumps({"ready": True, "engine": "mlx-demucs (Apple MLX)"}))
            return 0
        if has_module("demucs_infer"):
            print(json.dumps({"ready": True, "engine": "demucs-infer (PyTorch)"}))
            return 0
        print(json.dumps({"ready": False, "engine": "none"}))
        return 1

    input_wav = Path(args.input).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not input_wav.exists():
        print(json.dumps({"ok": False, "error": f"Arquivo de entrada não encontrado: {input_wav}"}))
        return 1

    engine_used = None
    last_error = "Nenhum backend Demucs encontrado (.venv-stems sem demucs-mlx ou demucs-infer)."

    with tempfile.TemporaryDirectory(prefix="htdemucs_") as tmp:
        tmp_dir = Path(tmp)

        for runner in (run_demucs_mlx, run_mlx_demucs):
            try:
                engine_used = runner(input_wav, tmp_dir, args.model, args.quality)
                break
            except FileNotFoundError:
                continue
            except Exception as exc:
                last_error = str(exc)
                break

        if engine_used is None:
            try:
                engine_used = run_demucs_infer(input_wav, out_dir, args.model)
            except ImportError:
                pass
            except Exception as exc:
                last_error = str(exc)

        if engine_used is None:
            print(json.dumps({"ok": False, "error": last_error}))
            return 1

        stems = collect_stems(tmp_dir, out_dir)
        if not stems:
            stems = collect_stems(out_dir, out_dir)

    if not stems:
        print(json.dumps({"ok": False, "error": "Separação concluiu, mas nenhum arquivo .wav de stem foi localizado."}))
        return 1

    print(json.dumps({"ok": True, "engine": engine_used, "stems": stems}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
