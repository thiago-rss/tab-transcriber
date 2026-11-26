#!/usr/bin/env python3
"""Worker local de transcrição neural (MuScriptor + Beat This!) para o Tab Transcriber."""

import argparse
import contextlib
import importlib.util
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def find_bin(name: str) -> str | None:
    for base in (Path(sys.prefix) / "bin", Path(sys.executable).parent):
        candidate = base / name
        if candidate.exists():
            return str(candidate)
    return shutil.which(name)


def has_module(mod_name: str) -> bool:
    return importlib.util.find_spec(mod_name) is not None


def detect_torch_device(preferred: str = "auto") -> tuple[str, bool]:
    try:
        import torch  # type: ignore

        mps_ok = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
        if preferred == "cpu":
            return "cpu", mps_ok
        if preferred == "mps":
            return ("mps" if mps_ok else "cpu"), mps_ok
        return ("mps" if mps_ok else "cpu"), mps_ok
    except Exception:
        return "cpu", False


# ============================================================================
# PARSER STANDARD MIDI FILE (SMF FORMAT 0 / 1) EM PYTHON PURO
# ============================================================================
def _read_vlq(data: bytes, pos: int) -> tuple[int, int]:
    val = 0
    while pos < len(data):
        b = data[pos]
        pos += 1
        val = (val << 7) | (b & 0x7F)
        if not (b & 0x80):
            break
    return val, pos


def parse_midi_bytes(raw: bytes) -> list[dict]:
    if len(raw) < 14 or raw[:4] != b"MThd":
        raise ValueError("Cabeçalho MIDI inválido")

    hdr_len = struct.unpack(">I", raw[4:8])[0]
    _fmt, n_tracks, division = struct.unpack(">HHH", raw[8:14])
    tpq = division if not (division & 0x8000) else 480

    pos = 8 + hdr_len
    tempo_events: list[tuple[int, int]] = [(0, 500_000)]
    raw_notes: list[tuple[int, int, int, int, int]] = []

    for _ in range(n_tracks):
        if pos + 8 > len(raw) or raw[pos : pos + 4] != b"MTrk":
            break
        trk_len = struct.unpack(">I", raw[pos + 4 : pos + 8])[0]
        pos += 8
        trk_end = min(pos + trk_len, len(raw))

        abs_tick = 0
        running_status = 0
        ch_programs = [0] * 16
        active: dict[tuple[int, int], list[tuple[int, int, int]]] = {}

        while pos < trk_end:
            delta, pos = _read_vlq(raw, pos)
            abs_tick += delta
            if pos >= trk_end:
                break

            status = raw[pos]
            if status & 0x80:
                pos += 1
                if status < 0xF0:
                    running_status = status
            else:
                status = running_status

            if status == 0xFF:
                if pos >= trk_end:
                    break
                meta_type = raw[pos]
                pos += 1
                length, pos = _read_vlq(raw, pos)
                meta_data = raw[pos : pos + length]
                pos += length
                if meta_type == 0x51 and len(meta_data) == 3:
                    us_per_q = (meta_data[0] << 16) | (meta_data[1] << 8) | meta_data[2]
                    tempo_events.append((abs_tick, max(100_000, us_per_q)))
                elif meta_type == 0x2F:
                    break
            elif status in (0xF0, 0xF7):
                length, pos = _read_vlq(raw, pos)
                pos += length
            elif 0x80 <= status <= 0xEF:
                kind = status & 0xF0
                ch = status & 0x0F
                if kind in (0xC0, 0xD0):
                    if pos >= trk_end:
                        break
                    d1 = raw[pos]
                    pos += 1
                    if kind == 0xC0:
                        ch_programs[ch] = d1
                else:
                    if pos + 1 >= trk_end:
                        break
                    d1, d2 = raw[pos], raw[pos + 1]
                    pos += 2
                    if ch == 9:
                        continue  # Ignora canal de bateria GM
                    key = (ch, d1)
                    if kind == 0x90 and d2 > 0:
                        active.setdefault(key, []).append((abs_tick, d2, ch_programs[ch]))
                    elif kind == 0x80 or (kind == 0x90 and d2 == 0):
                        stack = active.get(key)
                        if stack:
                            st, vel, prog = stack.pop(0)
                            if abs_tick > st:
                                raw_notes.append((st, abs_tick, d1, vel, prog))
            else:
                break

        pos = trk_end

    tempo_events.sort(key=lambda x: x[0])
    dedup_tempos: list[tuple[int, int]] = []
    for t, us in tempo_events:
        if dedup_tempos and dedup_tempos[-1][0] == t:
            dedup_tempos[-1] = (t, us)
        else:
            dedup_tempos.append((t, us))

    seg_ticks: list[int] = []
    seg_ms: list[float] = []
    seg_us: list[int] = []
    acc_ms = 0.0
    prev_t = 0
    prev_us = 500_000
    for t, us in dedup_tempos:
        acc_ms += ((t - prev_t) / tpq) * (prev_us / 1000.0)
        seg_ticks.append(t)
        seg_ms.append(acc_ms)
        seg_us.append(us)
        prev_t = t
        prev_us = us

    def tick_to_ms(tick: int) -> int:
        idx = 0
        for i, st in enumerate(seg_ticks):
            if st <= tick:
                idx = i
            else:
                break
        ms = seg_ms[idx] + ((tick - seg_ticks[idx]) / tpq) * (seg_us[idx] / 1000.0)
        return max(0, int(round(ms)))

    notes: list[dict] = []
    for st, en, pitch, vel, prog in raw_notes:
        s_ms = tick_to_ms(st)
        e_ms = tick_to_ms(en)
        dur = e_ms - s_ms
        if dur >= 35 and 21 <= pitch <= 108:
            notes.append(
                {
                    "start_ms": s_ms,
                    "duration_ms": dur,
                    "midi_note": int(pitch),
                    "velocity": int(vel),
                    "program": int(prog),
                }
            )

    notes.sort(key=lambda n: (n["start_ms"], n["midi_note"]))
    return notes


# ============================================================================
# BEAT THIS! (DETECÇÃO NEURAL DE BATIDAS E CABEÇAS DE COMPASSO)
# ============================================================================
def run_beat_this(mix_wav: Path, device: str) -> tuple[list[float], list[float], str]:
    from beat_this.inference import File2Beats  # type: ignore

    last_exc = None
    fallback_res: tuple[list[float], list[float], str] | None = None
    for dev in ([device, "cpu"] if device != "cpu" else ["cpu"]):
        try:
            tracker = File2Beats(checkpoint_path="final0", device=dev, dbn=False)
            beats, downbeats = tracker(str(mix_wav))
            b_list = sorted(float(x) for x in beats if x == x)
            db_list = sorted(float(x) for x in downbeats if x == x)
            if len(b_list) >= 2:
                return b_list, db_list, f"beat-this ({dev})"
            fallback_res = (b_list, db_list, f"beat-this ({dev})")
        except Exception as exc:
            last_exc = exc
            continue
    if fallback_res is not None:
        return fallback_res
    raise RuntimeError(f"Beat This! falhou: {last_exc}")


# ============================================================================
# MUSCRIPTOR (API PYTHON OFICIAL + FALLBACK CLI `muscriptor transcribe`)
# ============================================================================
def run_muscriptor(audio_wav: Path, device: str) -> tuple[list[dict], str]:
    model_size = os.environ.get("TAB_TRANSCRIBER_MUSCRIPTOR_MODEL", "medium").strip() or "medium"
    api_err = None

    # 1. Caminho primário: API Python oficial `TranscriptionModel.load_model()`
    if has_module("muscriptor"):
        try:
            from muscriptor import TranscriptionModel  # type: ignore

            try:
                model = TranscriptionModel.load_model(model_size, device=device)
            except TypeError:
                model = TranscriptionModel.load_model(model_size)

            midi_bytes = model.transcribe_to_midi(str(audio_wav))
            if isinstance(midi_bytes, (bytes, bytearray)):
                notes = parse_midi_bytes(bytes(midi_bytes))
                return notes, f"MuScriptor-{model_size} ({device})"
        except Exception as exc:
            api_err = str(exc)

    # 2. Fallback: subcomando CLI oficial `muscriptor transcribe`
    bin_path = find_bin("muscriptor")
    if bin_path:
        base_cmd = [bin_path, "transcribe"]
    elif has_module("muscriptor"):
        base_cmd = [sys.executable, "-m", "muscriptor", "transcribe"]
    else:
        raise FileNotFoundError(api_err or "muscriptor não instalado na .venv-neural")

    with tempfile.TemporaryDirectory(prefix="muscriptor_") as tmp:
        tmp_dir = Path(tmp)
        out_midi = tmp_dir / "transcription.mid"
        cmd = [*base_cmd, "--model", model_size, str(audio_wav), "-o", str(out_midi)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            err_msg = (res.stderr or res.stdout).strip()
            combined = f"{api_err} | CLI: {err_msg}" if api_err else err_msg
            raise RuntimeError(combined[:400])

        midi_files = [out_midi] if out_midi.is_file() else sorted(tmp_dir.rglob("*.mid"))
        if not midi_files:
            midi_files = sorted(tmp_dir.rglob("*.midi"))
        if not midi_files:
            raise RuntimeError("muscriptor transcribe concluiu sem gerar arquivo .mid.")

        notes = parse_midi_bytes(midi_files[0].read_bytes())
        return notes, f"MuScriptor-{model_size} ({device})"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default="", help="WAV alvo para transcrição de notas")
    parser.add_argument("--mix", default="", help="WAV do mix completo para Beat This!")
    parser.add_argument("--stem", default="mix", help="Nome do stem")
    parser.add_argument("--engine", default="auto", choices=["auto", "muscriptor", "beats_only"])
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cpu"])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    device, mps_ok = detect_torch_device(args.device)
    has_muscriptor = bool(find_bin("muscriptor") or has_module("muscriptor"))
    has_beat_this = has_module("beat_this")

    if args.check:
        parts = []
        if has_muscriptor:
            parts.append(f"MuScriptor ({device.upper()})")
        if has_beat_this:
            parts.append(f"Beat This! ({device.upper()})")
        hint = (
            " + ".join(parts)
            if parts
            else "Instale beat-this e muscriptor em ~/tab-transcriber/.venv-neural"
        )
        print(
            json.dumps(
                {
                    "muscriptor_ready": has_muscriptor,
                    "beat_this_ready": has_beat_this,
                    "mps_available": mps_ok,
                    "device": device,
                    "hint": hint,
                }
            )
        )
        return 0

    audio_wav = Path(args.audio).resolve()
    mix_wav = Path(args.mix).resolve() if args.mix else audio_wav

    if not audio_wav.is_file():
        print(json.dumps({"ok": False, "error": f"WAV não encontrado: {audio_wav}"}))
        return 1

    beats_sec: list[float] = []
    downbeats_sec: list[float] = []
    beat_engine = "none"
    beat_error = None

    if has_beat_this and mix_wav.is_file():
        try:
            with contextlib.redirect_stdout(sys.stderr):
                beats_sec, downbeats_sec, beat_engine = run_beat_this(mix_wav, device)
        except Exception as exc:
            beat_error = str(exc)

    notes: list[dict] = []
    muscriptor_used = False
    muscriptor_engine = "none"
    muscriptor_error = None

    if args.engine in ("auto", "muscriptor") and has_muscriptor:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                notes, muscriptor_engine = run_muscriptor(audio_wav, device)
            muscriptor_used = True
        except Exception as exc:
            muscriptor_error = str(exc)

    print(
        json.dumps(
            {
                "ok": True,
                "muscriptor_used": muscriptor_used,
                "muscriptor_engine": muscriptor_engine,
                "muscriptor_error": muscriptor_error,
                "beat_engine": beat_engine,
                "beat_error": beat_error,
                "beats_sec": beats_sec,
                "downbeats_sec": downbeats_sec,
                "notes": notes,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
