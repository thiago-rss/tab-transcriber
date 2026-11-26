import './style.css';
import * as alphaTab from '@coderline/alphatab';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

interface AudioSourceDto {
  id: string;
  name: string;
  bundle_id: string;
  pids: number[];
  object_ids: number[];
  is_running_output: boolean;
}

interface NeuralBackendStatus {
  muscriptor_ready: boolean;
  beat_this_ready: boolean;
  mps_available: boolean;
  device: string;
  hint: string;
}

interface BackendStatus {
  ready: boolean;
  engine: string;
  hint: string;
}

interface StemActivity {
  stem: string;
  label: string;
  rms_db: number;
  active: boolean;
}

interface StoredStemSessionDto {
  id: string;
  label: string;
  created_at: number;
  session_dir: string;
  engine: string;
  active_stems: string[];
}

interface UserPreferences {
  mode: string;
  stereo_mode: string;
  auto_level: boolean;
  enable_hpss: boolean;
  focus: string;
  tuning: string;
  neck_region: string;
  prefer_closed: boolean;
  time_signature: string;
  bpm: string;
  view: string;
  stem_quality: string;
  stem: string;
  playback_speed: number;
  metronome_enabled: boolean;
}

interface StemSeparationResultDto {
  engine: string;
  available_stems: string[];
  stem_activities: StemActivity[];
  active_stem: string;
  separation_ms: number;
  transcription_ms: number;
  event_count: number;
  session_dir: string;
}

interface SnippetProgressPayload {
  elapsed_sec: number;
  max_sec: number;
}

interface TranscriptionUpdatePayload {
  alphatex: string;
  latest_notes: string[];
  latest_chord: string | null;
  detected_key: string;
  detected_instrument: string;
  time_signature: string;
  event_count: number;
  latency_ms: number;
  detected_bpm: number;
  estimated_a4_hz: number;
  pruned_overtones: number;
  bend_semitones: number;
  is_reattack: boolean;
}

interface AudioLevelPayload {
  rms_db: number;
  peak_db: number;
  silence_warning: boolean;
  dropped_samples: number;
  device_switches: number;
  clip_events: number;
  clock_glitches: number;
  phase_corr: number;
  sample_rate: number;
  format_label: string;
}

const sourceSelect = document.getElementById('source-select') as HTMLSelectElement;
const refreshBtn = document.getElementById('refresh-btn') as HTMLButtonElement;
const modeSelect = document.getElementById('mode-select') as HTMLSelectElement;
const stereoSelect = document.getElementById('stereo-select') as HTMLSelectElement;
const autolevelSelect = document.getElementById('autolevel-select') as HTMLSelectElement;
const hpssSelect = document.getElementById('hpss-select') as HTMLSelectElement;
const focusSelect = document.getElementById('focus-select') as HTMLSelectElement;
const tuningSelect = document.getElementById('tuning-select') as HTMLSelectElement;
const regionField = document.getElementById('region-field') as HTMLLabelElement;
const closedField = document.getElementById('closed-field') as HTMLLabelElement;
const regionSelect = document.getElementById('region-select') as HTMLSelectElement;
const closedSelect = document.getElementById('closed-select') as HTMLSelectElement;
const tsSelect = document.getElementById('ts-select') as HTMLSelectElement;
const bpmSelect = document.getElementById('bpm-select') as HTMLSelectElement;
const viewSelect = document.getElementById('view-select') as HTMLSelectElement;
const toggleBtn = document.getElementById('toggle-btn') as HTMLButtonElement;
const clearBtn = document.getElementById('clear-btn') as HTMLButtonElement;
const exportXmlBtn = document.getElementById('export-xml-btn') as HTMLButtonElement;
const exportMidBtn = document.getElementById('export-mid-btn') as HTMLButtonElement;
const openDirBtn = document.getElementById('open-dir-btn') as HTMLButtonElement;
const statusDot = document.getElementById('status-dot') as HTMLSpanElement;
const statusText = document.getElementById('status-text') as HTMLSpanElement;
const meterFill = document.getElementById('meter-fill') as HTMLDivElement;
const dbLabel = document.getElementById('db-label') as HTMLSpanElement;
const phaseLabel = document.getElementById('phase-label') as HTMLSpanElement;
const instrLabel = document.getElementById('instr-label') as HTMLSpanElement;
const keyLabel = document.getElementById('key-label') as HTMLSpanElement;
const chordLabel = document.getElementById('chord-label') as HTMLSpanElement;
const bpmLabel = document.getElementById('bpm-label') as HTMLSpanElement;
const bpmHalfBtn = document.getElementById('bpm-half-btn') as HTMLButtonElement;
const bpmDoubleBtn = document.getElementById('bpm-double-btn') as HTMLButtonElement;
const a4Label = document.getElementById('a4-label') as HTMLSpanElement;
const latestNotesEl = document.getElementById('latest-notes') as HTMLSpanElement;
const warningBanner = document.getElementById('warning-banner') as HTMLDivElement;
const scoreViewport = document.getElementById('score-viewport') as HTMLElement;
const scoreContainer = document.getElementById('alphaTab') as HTMLDivElement;

const recordStemBtn = document.getElementById('record-stem-btn') as HTMLButtonElement;
const cancelStemBtn = document.getElementById('cancel-stem-btn') as HTMLButtonElement;
const importAudioBtn = document.getElementById('import-audio-btn') as HTMLButtonElement;
const recTimerEl = document.getElementById('rec-timer') as HTMLSpanElement;
const savedSessionsSelect = document.getElementById('saved-sessions-select') as HTMLSelectElement;
const stemQualitySelect = document.getElementById('stem-quality-select') as HTMLSelectElement;
const neuralEngineSelect = document.getElementById('neural-engine-select') as HTMLSelectElement;
const stemSelect = document.getElementById('stem-select') as HTMLSelectElement;
const retranscribeStemBtn = document.getElementById('retranscribe-stem-btn') as HTMLButtonElement;
const stemPlayBtn = document.getElementById('stem-play-btn') as HTMLButtonElement;
const stemAbBtn = document.getElementById('stem-ab-btn') as HTMLButtonElement;
const stemSeek = document.getElementById('stem-seek') as HTMLInputElement;
const stemAudioTime = document.getElementById('stem-audio-time') as HTMLSpanElement;
const stemsStatusEl = document.getElementById('stems-status') as HTMLSpanElement;

const playPauseBtn = document.getElementById('play-pause-btn') as HTMLButtonElement;
const stopPlayBtn = document.getElementById('stop-play-btn') as HTMLButtonElement;
const loopBtn = document.getElementById('loop-btn') as HTMLButtonElement;
const metronomeBtn = document.getElementById('metronome-btn') as HTMLButtonElement;
const speedSelect = document.getElementById('speed-select') as HTMLSelectElement;
const timeLabel = document.getElementById('time-label') as HTMLSpanElement;
const sfStatus = document.getElementById('sf-status') as HTMLSpanElement;

let isRunning = false;
let isRecordingSnippet = false;
let isSeparating = false;
let isTransitioning = false;
let hasSeparatedStems = false;
let isPlayerReady = false;
let lastFormattedTime = '';
let lastDeviceSwitches = 0;
let sourcesMap = new Map<string, AudioSourceDto>();

const stemAudio = new Audio();
const stemBlobCache = new Map<string, string>();
let stemAudioListenMix = false;
let loadedAudioStemKey: string | null = null;
let isScrubbingStem = false;
let desiredMetronomeEnabled = false;
let desiredPlaybackSpeed = 1.0;
let savePrefsTimer: number | null = null;

const api = new alphaTab.AlphaTabApi(scoreContainer, {
  core: {
    fontDirectory: `${import.meta.env.BASE_URL}font/`,
    useWorkers: true,
  },
  display: {
    staveProfile: alphaTab.StaveProfile.ScoreTab,
    layoutMode: alphaTab.LayoutMode.Page,
    scale: 1.0,
  },
  player: {
    playerMode: alphaTab.PlayerMode.EnabledSynthesizer,
    soundFont: `${import.meta.env.BASE_URL}soundfont/sonivox.sf2`,
    enableCursor: true,
    enableAnimatedBeatCursor: true,
    enableElementHighlighting: true,
    enableSeekToClick: true,
    enablePlaybackRangeSelection: true,
    scrollElement: scoreViewport,
    scrollMode: alphaTab.ScrollMode.Continuous,
    scrollOffsetY: -24,
  },
});

function formatDuration(ms: number): string {
  const totalSec = Math.max(0, Math.floor(ms / 1000));
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return `${String(min).padStart(2, '0')}:${String(sec).padStart(2, '0')}`;
}

function setPlayerControlsEnabled(enabled: boolean) {
  playPauseBtn.disabled = !enabled;
  stopPlayBtn.disabled = !enabled;
  loopBtn.disabled = !enabled;
  metronomeBtn.disabled = !enabled;
  speedSelect.disabled = !enabled;
}

function clearStemAudioCache() {
  stemAudio.pause();
  stemAudio.removeAttribute('src');
  stemAudio.load();
  for (const url of stemBlobCache.values()) {
    URL.revokeObjectURL(url);
  }
  stemBlobCache.clear();
  loadedAudioStemKey = null;
  stemAudioListenMix = false;
  stemPlayBtn.textContent = '▶ Ouvir Áudio';
  stemPlayBtn.classList.remove('toggle-active');
  stemAbBtn.textContent = 'A/B: Stem';
  stemAbBtn.classList.remove('toggle-active');
  stemSeek.value = '0';
  stemAudioTime.textContent = '00:00 / 00:00';
}

function setStemAudioControlsEnabled(enabled: boolean) {
  stemPlayBtn.disabled = !enabled;
  stemAbBtn.disabled = !enabled;
  stemSeek.disabled = !enabled;
}

function syncUiControls() {
  toggleBtn.textContent = isRunning ? 'Parar' : 'Ao Vivo';
  toggleBtn.classList.toggle('running', isRunning);
  toggleBtn.disabled = isSeparating || isTransitioning;

  recordStemBtn.textContent = isSeparating
    ? 'Separando…'
    : isRecordingSnippet
      ? '■ Parar e Separar'
      : '● Gravar Trecho';
  recordStemBtn.classList.toggle('recording', isRecordingSnippet);
  recordStemBtn.disabled = isSeparating || isTransitioning;
  importAudioBtn.disabled = isSeparating || isTransitioning || isRecordingSnippet;
  savedSessionsSelect.disabled = isSeparating || isTransitioning || isRecordingSnippet;
  neuralEngineSelect.disabled = isSeparating || isTransitioning;

  cancelStemBtn.classList.toggle('hidden', !isRecordingSnippet);
  cancelStemBtn.disabled = isSeparating || isTransitioning;

  retranscribeStemBtn.disabled = !hasSeparatedStems || isSeparating || isRecordingSnippet || isRunning;

  statusDot.classList.toggle('live', isRunning || isRecordingSnippet);

  sourceSelect.disabled = isRunning || isRecordingSnippet || isSeparating;
  modeSelect.disabled = isRunning || isSeparating;
  stereoSelect.disabled = isRunning || isSeparating;
  autolevelSelect.disabled = isRunning || isSeparating;
  hpssSelect.disabled = isRunning || isSeparating;
  focusSelect.disabled = isRunning || isSeparating;
  tuningSelect.disabled = isRunning || isSeparating;
  regionSelect.disabled = isRunning || isSeparating;
  closedSelect.disabled = isRunning || isSeparating;
  tsSelect.disabled = isRunning || isSeparating;
  bpmSelect.disabled = isRunning || isSeparating;

  if (!isRunning && !isRecordingSnippet) {
    meterFill.style.width = '0%';
    dbLabel.textContent = '-∞ dBFS';
    warningBanner.classList.add('hidden');
  }
}

function updateStemSelectWithActivities(activities: StemActivity[]) {
  const prev = stemSelect.value;
  stemSelect.innerHTML = '';

  const order = ['guitar', 'bass', 'piano', 'vocals', 'other', 'drums'];
  const actMap = new Map(activities.map((a) => [a.stem, a]));

  for (const key of order) {
    const act = actMap.get(key);
    if (!act) continue;
    const opt = document.createElement('option');
    opt.value = key;
    const badge = act.active ? '●' : '○';
    const dbTxt = act.rms_db <= -90 ? 'silêncio' : `${act.rms_db.toFixed(0)} dB`;
    opt.textContent = `${badge} ${act.label} (${dbTxt})`;
    stemSelect.appendChild(opt);
  }

  const mixOpt = document.createElement('option');
  mixOpt.value = 'mix';
  mixOpt.textContent = 'Mix Original';
  stemSelect.appendChild(mixOpt);

  if (prev && Array.from(stemSelect.options).some((o) => o.value === prev)) {
    stemSelect.value = prev;
  }
}


const NON_FRETTED_PRESETS = new Set(['piano', 'vocals', 'strings', 'cello', 'winds', 'synth']);

function syncStaveProfileFromTex(tex: string) {
  const isScoreOnly = tex.includes('\\staff {score}') && !tex.includes('\\staff {score tabs}');
  if (isScoreOnly) {
    if (api.settings.display.staveProfile !== alphaTab.StaveProfile.Score) {
      api.settings.display.staveProfile = alphaTab.StaveProfile.Score;
      api.updateSettings();
    }
  } else {
    const desired =
      viewSelect.value === 'tab'
        ? alphaTab.StaveProfile.Tab
        : viewSelect.value === 'score'
          ? alphaTab.StaveProfile.Score
          : alphaTab.StaveProfile.ScoreTab;
    if (api.settings.display.staveProfile !== desired) {
      api.settings.display.staveProfile = desired;
      api.updateSettings();
    }
  }
}

function syncInstrumentAdaptiveUi() {
  const isFretted = !NON_FRETTED_PRESETS.has(tuningSelect.value);
  regionField.classList.toggle('hidden', !isFretted);
  closedField.classList.toggle('hidden', !isFretted);

  if (!isFretted) {
    viewSelect.value = 'score';
    api.settings.display.staveProfile = alphaTab.StaveProfile.Score;
    api.updateSettings();
  } else if (viewSelect.value === 'score') {
    viewSelect.value = 'scoretab';
    api.settings.display.staveProfile = alphaTab.StaveProfile.ScoreTab;
    api.updateSettings();
  }
}

function autoMatchFocusForInstrument(presetId: string) {
  const map: Record<string, string> = {
    piano: 'piano',
    vocals: 'vocals',
    strings: 'strings',
    cello: 'strings',
    winds: 'winds',
    synth: 'piano',
    bass4: 'bass',
    bass5: 'bass',
    oud_syrian: 'oud',
  };
  if (map[presetId]) {
    focusSelect.value = map[presetId];
  }
}

function applyRecommendedPresetForStem(stem: string) {
  if (isRunning) return;
  if (stem === 'bass') {
    tuningSelect.value = 'bass4';
    focusSelect.value = 'bass';
  } else if (stem === 'piano') {
    tuningSelect.value = 'piano';
    focusSelect.value = 'piano';
  } else if (stem === 'vocals') {
    tuningSelect.value = 'vocals';
    focusSelect.value = 'vocals';
  } else if (stem === 'guitar') {
    if (['bass4', 'bass5', 'piano', 'vocals', 'strings', 'winds', 'synth'].includes(tuningSelect.value)) {
      tuningSelect.value = 'standard';
    }
    if (['bass', 'piano', 'vocals'].includes(focusSelect.value)) {
      focusSelect.value = 'guitar';
    }
  }
}

async function getStemBlobUrl(stemKey: string): Promise<string> {
  const cached = stemBlobCache.get(stemKey);
  if (cached) return cached;

  const buffer = await invoke<ArrayBuffer>('load_stem_audio', { stem: stemKey });
  const blob = new Blob([buffer], { type: 'audio/wav' });
  const url = URL.createObjectURL(blob);
  stemBlobCache.set(stemKey, url);
  return url;
}

async function loadStemIntoAudioElement(stemKey: string, preserveTime: boolean) {
  const prevTime = preserveTime ? stemAudio.currentTime : 0;
  const wasPlaying = preserveTime && !stemAudio.paused;

  const url = await getStemBlobUrl(stemKey);
  if (loadedAudioStemKey === stemKey && stemAudio.src === url) {
    return;
  }

  loadedAudioStemKey = stemKey;
  stemAudio.src = url;

  await new Promise<void>((resolve) => {
    const onReady = () => {
      stemAudio.removeEventListener('loadedmetadata', onReady);
      resolve();
    };
    stemAudio.addEventListener('loadedmetadata', onReady);
  });

  if (preserveTime && Number.isFinite(stemAudio.duration) && prevTime < stemAudio.duration) {
    stemAudio.currentTime = prevTime;
  }
  updateStemTimeDisplay();

  if (wasPlaying) {
    await stemAudio.play();
  }
}

function updateStemTimeDisplay() {
  const cur = Number.isFinite(stemAudio.currentTime) ? stemAudio.currentTime : 0;
  const dur = Number.isFinite(stemAudio.duration) ? stemAudio.duration : 0;
  stemAudioTime.textContent = `${formatDuration(cur * 1000)} / ${formatDuration(dur * 1000)}`;
  if (!isScrubbingStem && dur > 0) {
    stemSeek.value = String(Math.round((cur / dur) * 1000));
  }
}

stemAudio.addEventListener('timeupdate', updateStemTimeDisplay);
stemAudio.addEventListener('ended', () => {
  stemPlayBtn.textContent = '▶ Ouvir Áudio';
  stemPlayBtn.classList.remove('toggle-active');
});

stemSeek.addEventListener('input', () => {
  isScrubbingStem = true;
  const dur = Number.isFinite(stemAudio.duration) ? stemAudio.duration : 0;
  if (dur > 0) {
    const target = (Number(stemSeek.value) / 1000) * dur;
    stemAudioTime.textContent = `${formatDuration(target * 1000)} / ${formatDuration(dur * 1000)}`;
  }
});

stemSeek.addEventListener('change', () => {
  const dur = Number.isFinite(stemAudio.duration) ? stemAudio.duration : 0;
  if (dur > 0) {
    stemAudio.currentTime = (Number(stemSeek.value) / 1000) * dur;
  }
  isScrubbingStem = false;
});

stemPlayBtn.addEventListener('click', async () => {
  if (!hasSeparatedStems) return;

  if (!stemAudio.paused) {
    stemAudio.pause();
    stemPlayBtn.textContent = '▶ Ouvir Áudio';
    stemPlayBtn.classList.remove('toggle-active');
    return;
  }

  if (isPlayerReady) api.pause();

  try {
    const targetKey = stemAudioListenMix ? 'mix' : stemSelect.value;
    await loadStemIntoAudioElement(targetKey, false);
    await stemAudio.play();
    stemPlayBtn.textContent = '❚❚ Pausar Áudio';
    stemPlayBtn.classList.add('toggle-active');
  } catch (err) {
    statusText.textContent = `Erro no player de stem: ${String(err)}`;
  }
});

stemAbBtn.addEventListener('click', async () => {
  if (!hasSeparatedStems) return;
  stemAudioListenMix = !stemAudioListenMix;
  stemAbBtn.textContent = stemAudioListenMix ? 'A/B: Mix' : 'A/B: Stem';
  stemAbBtn.classList.toggle('toggle-active', stemAudioListenMix);

  try {
    const targetKey = stemAudioListenMix ? 'mix' : stemSelect.value;
    await loadStemIntoAudioElement(targetKey, true);
  } catch (err) {
    statusText.textContent = `Erro ao alternar A/B: ${String(err)}`;
  }
});

function buildStemTranscribeConfig() {
  return {
    stem: stemSelect.value,
    mode: modeSelect.value,
    tuning: tuningSelect.value,
    bpm: bpmSelect.value === 'auto' ? null : Number(bpmSelect.value),
    focus: focusSelect.value,
    time_signature: tsSelect.value,
    neck_region: regionSelect.value,
    prefer_closed: closedSelect.value === 'closed',
    quality: stemQualitySelect.value,
    neural_engine: neuralEngineSelect.value,
  };
}

api.soundFontLoad.on((e) => {
  if (e.total > 0) {
    const pct = Math.min(100, Math.round((e.loaded / e.total) * 100));
    sfStatus.textContent = `SoundFont ${pct}%`;
  }
});

api.playerReady.on(() => {
  isPlayerReady = true;
  setPlayerControlsEnabled(true);
  sfStatus.textContent = 'SoundFont local pronto · Clique para navegar / Arraste para loop';
  sfStatus.classList.add('ready');
  api.playbackSpeed = desiredPlaybackSpeed;
  api.metronomeVolume = desiredMetronomeEnabled ? 0.6 : 0;
  metronomeBtn.classList.toggle('toggle-active', desiredMetronomeEnabled);
});

api.playerStateChanged.on((e) => {
  const playing = e.state === alphaTab.synth.PlayerState.Playing;
  playPauseBtn.textContent = playing ? '❚❚ Pause Tab' : '▶ Play Tab';
  playPauseBtn.classList.toggle('toggle-active', playing);
});

api.playerPositionChanged.on((e) => {
  const formatted = `${formatDuration(e.currentTime)} / ${formatDuration(e.endTime)}`;
  if (formatted !== lastFormattedTime) {
    lastFormattedTime = formatted;
    timeLabel.textContent = formatted;
  }
});

async function handlePlayPause() {
  if (!isPlayerReady) return;

  if (!stemAudio.paused) {
    stemAudio.pause();
    stemPlayBtn.textContent = '▶ Ouvir Áudio';
    stemPlayBtn.classList.remove('toggle-active');
  }

  if (isRunning) {
    await invoke('stop_transcription');
    isRunning = false;
    syncUiControls();
    statusText.textContent = 'Captura parada para revisão';
  }

  api.playPause();
}

playPauseBtn.addEventListener('click', () => {
  void handlePlayPause();
});

stopPlayBtn.addEventListener('click', () => {
  if (!isPlayerReady) return;
  api.stop();
});

loopBtn.addEventListener('click', () => {
  api.isLooping = !api.isLooping;
  loopBtn.classList.toggle('toggle-active', api.isLooping);
});

metronomeBtn.addEventListener('click', () => {
  const nextActive = api.metronomeVolume === 0;
  api.metronomeVolume = nextActive ? 0.6 : 0;
  desiredMetronomeEnabled = nextActive;
  metronomeBtn.classList.toggle('toggle-active', nextActive);
  scheduleSavePreferences();
});

speedSelect.addEventListener('change', () => {
  desiredPlaybackSpeed = Number(speedSelect.value);
  api.playbackSpeed = desiredPlaybackSpeed;
  scheduleSavePreferences();
});

window.addEventListener('keydown', (e) => {
  if (e.code !== 'Space') return;
  const tag = (e.target as HTMLElement | null)?.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || tag === 'BUTTON') {
    return;
  }
  e.preventDefault();
  void handlePlayPause();
});

async function renderEmptyScore() {
  const tex = await invoke<string>('get_initial_tex', {
    tuning: tuningSelect.value,
    timeSignature: tsSelect.value,
  });
  bpmLabel.textContent = `100 BPM · ${tsSelect.value}`;
  syncStaveProfileFromTex(tex);
      api.tex(tex);
}

async function refreshAudioSources() {
  const prevSelection = sourceSelect.value;
  try {
    const sources = await invoke<AudioSourceDto[]>('list_audio_sources', {
      onlyActive: false,
    });
    sourcesMap.clear();
    sourceSelect.innerHTML = '';

    const globalOpt = document.createElement('option');
    globalOpt.value = '__global__';
    globalOpt.textContent = 'Áudio Global do Sistema';
    sourceSelect.appendChild(globalOpt);

    for (const src of sources) {
      sourcesMap.set(src.id, src);
      const opt = document.createElement('option');
      opt.value = src.id;
      const badge = src.is_running_output ? '● ' : '○ ';
      const pidHint = src.pids.length > 1 ? ` (${src.pids.length} proc)` : '';
      opt.textContent = `${badge}${src.name}${pidHint}`;
      sourceSelect.appendChild(opt);
    }

    if (prevSelection && (prevSelection === '__global__' || sourcesMap.has(prevSelection))) {
      sourceSelect.value = prevSelection;
    }
  } catch (err) {
    statusText.textContent = `Erro ao listar fontes: ${String(err)}`;
  }
}

async function checkStemsAvailability() {
  try {
    const [st, neural] = await Promise.all([
      invoke<BackendStatus>('check_stems_backend'),
      invoke<NeuralBackendStatus>('check_neural_backend'),
    ]);
    const neuralTag =
      neural.muscriptor_ready || neural.beat_this_ready
        ? ` · ${neural.hint}`
        : ' · Neural offline: Basic Pitch';
    if (st.ready) {
      stemsStatusEl.textContent = `HTDemucs 6s (${st.engine})${neuralTag}`;
      stemsStatusEl.classList.add('ready');
    } else {
      stemsStatusEl.textContent = `HTDemucs: ${st.hint}${neuralTag}`;
      stemsStatusEl.classList.remove('ready');
    }
  } catch (err) {
    stemsStatusEl.textContent = `Erro ao verificar backends: ${String(err)}`;
  }
}

// Alterna livremente para Ao Vivo (mesmo se uma gravação de trecho estiver ativa)
toggleBtn.addEventListener('click', async () => {
  if (isSeparating || isTransitioning) return;

  isTransitioning = true;
  syncUiControls();

  try {
    if (isRunning) {
      await invoke('stop_transcription');
      isRunning = false;
      statusText.textContent = 'Parado (pronto para ouvir, exportar ou gravar stems)';
      return;
    }

    if (isRecordingSnippet) {
      await invoke('cancel_snippet_recording');
      isRecordingSnippet = false;
      recTimerEl.textContent = '00:00';
    }

    if (isPlayerReady) api.stop();
    stemAudio.pause();
    stemPlayBtn.textContent = '▶ Ouvir Áudio';
    stemPlayBtn.classList.remove('toggle-active');

    const selected = sourcesMap.get(sourceSelect.value);
    const targetIds = selected ? selected.object_ids : [];
    const bpmValue = bpmSelect.value === 'auto' ? null : Number(bpmSelect.value);

    lastDeviceSwitches = 0;
    statusText.textContent = 'Inicializando captura ao vivo...';

    const msg = await invoke<string>('start_transcription', {
      config: {
        mode: modeSelect.value,
        tuning: tuningSelect.value,
        target_ids: targetIds,
        source_id: sourceSelect.value,
        bpm: bpmValue,
        focus: focusSelect.value,
        time_signature: tsSelect.value,
        stereo_mode: stereoSelect.value,
        enable_hpss: hpssSelect.value === 'on',
        neck_region: regionSelect.value,
        prefer_closed: closedSelect.value === 'closed',
        auto_level: autolevelSelect.value === 'on',
      },
    });
    isRunning = true;
    statusText.textContent = msg;
  } catch (err) {
    isRunning = false;
    statusText.textContent = `Erro: ${String(err)}`;
  } finally {
    isTransitioning = false;
    syncUiControls();
  }
});

// Alterna livremente para Gravar Trecho (mesmo se Ao Vivo estiver rodando)
recordStemBtn.addEventListener('click', async () => {
  if (isSeparating || isTransitioning) return;

  if (!isRecordingSnippet) {
    isTransitioning = true;
    syncUiControls();

    try {
      if (isRunning) {
        await invoke('stop_transcription');
        isRunning = false;
      }

      if (isPlayerReady) api.stop();
      clearStemAudioCache();
      setStemAudioControlsEnabled(false);

      const selected = sourcesMap.get(sourceSelect.value);
      const targetIds = selected ? selected.object_ids : [];
      lastDeviceSwitches = 0;

      const msg = await invoke<string>('start_snippet_recording', {
        targetIds,
        sourceId: sourceSelect.value,
      });
      isRecordingSnippet = true;
      recTimerEl.textContent = '00:00';
      statusText.textContent = msg;
    } catch (err) {
      isRecordingSnippet = false;
      statusText.textContent = `Erro ao gravar trecho: ${String(err)}`;
    } finally {
      isTransitioning = false;
      syncUiControls();
    }
    return;
  }

  // Parar gravação e disparar separação neural + transcrição offline
  isRecordingSnippet = false;
  isSeparating = true;
  syncUiControls();

  try {
    const res = await invoke<StemSeparationResultDto>('stop_and_separate_snippet', {
      config: buildStemTranscribeConfig(),
    });
    hasSeparatedStems = true;
    updateStemSelectWithActivities(res.stem_activities);
    stemSelect.value = res.active_stem;
    setStemAudioControlsEnabled(true);

    await loadStemIntoAudioElement(res.active_stem, false);

    const activeCount = res.stem_activities.filter((a) => a.active).length;
    const sepSec = (res.separation_ms / 1000).toFixed(1);
    stemsStatusEl.textContent = `${activeCount}/6 stems ativos (${res.engine}) em ${sepSec}s · ativo: ${res.active_stem}`;
    void refreshSavedSessions(res.session_dir);
    stemsStatusEl.classList.add('ready');
    statusText.textContent = `Stem '${res.active_stem}': ${res.event_count} eventos (${res.transcription_ms} ms)`;
  } catch (err) {
    stemsStatusEl.textContent = `Falha: ${String(err)}`;
    stemsStatusEl.classList.remove('ready');
    statusText.textContent = `Erro HTDemucs: ${String(err)}`;
  } finally {
    isSeparating = false;
    syncUiControls();
  }
});

importAudioBtn.addEventListener('click', async () => {
  if (isSeparating || isTransitioning || isRecordingSnippet) return;

  isTransitioning = true;
  syncUiControls();

  let pickedPath: string | null = null;
  try {
    pickedPath = await invoke<string | null>('pick_audio_file');
  } catch (err) {
    statusText.textContent = `Erro ao selecionar arquivo: ${String(err)}`;
  } finally {
    isTransitioning = false;
    syncUiControls();
  }

  if (!pickedPath) return;

  if (isPlayerReady) api.stop();
  clearStemAudioCache();
  setStemAudioControlsEnabled(false);
  isRunning = false;
  isSeparating = true;
  syncUiControls();

  try {
    const res = await invoke<StemSeparationResultDto>('separate_audio_file', {
      filePath: pickedPath,
      config: buildStemTranscribeConfig(),
    });
    hasSeparatedStems = true;
    updateStemSelectWithActivities(res.stem_activities);
    stemSelect.value = res.active_stem;
    setStemAudioControlsEnabled(true);

    await loadStemIntoAudioElement(res.active_stem, false);

    const fileName = pickedPath.split('/').pop() ?? pickedPath;
    const activeCount = res.stem_activities.filter((a) => a.active).length;
    const sepSec = (res.separation_ms / 1000).toFixed(1);
    stemsStatusEl.textContent = `${fileName} · ${activeCount}/6 stems (${res.engine}) em ${sepSec}s`;
    void refreshSavedSessions(res.session_dir);
    stemsStatusEl.classList.add('ready');
    statusText.textContent = `Stem '${res.active_stem}': ${res.event_count} eventos (${res.transcription_ms} ms)`;
  } catch (err) {
    stemsStatusEl.textContent = `Falha: ${String(err)}`;
    stemsStatusEl.classList.remove('ready');
    statusText.textContent = `Erro ao importar áudio: ${String(err)}`;
  } finally {
    isSeparating = false;
    syncUiControls();
  }
});

cancelStemBtn.addEventListener('click', async () => {
  if (!isRecordingSnippet || isTransitioning) return;
  isTransitioning = true;
  syncUiControls();
  try {
    await invoke('cancel_snippet_recording');
    isRecordingSnippet = false;
    recTimerEl.textContent = '00:00';
    statusText.textContent = 'Gravação de trecho cancelada';
  } finally {
    isTransitioning = false;
    syncUiControls();
  }
});

async function retranscribeCurrentStem() {
  if (!hasSeparatedStems || isSeparating || isRecordingSnippet || isRunning) return;
  if (isPlayerReady) api.stop();

  isSeparating = true;
  syncUiControls();

  try {
    const msg = await invoke<string>('transcribe_existing_stem', {
      config: buildStemTranscribeConfig(),
    });
    if (!stemAudioListenMix) {
      await loadStemIntoAudioElement(stemSelect.value, true);
    }
    statusText.textContent = msg;
    stemsStatusEl.textContent = `Stem ativo: ${stemSelect.value} (cache local)`;
  } catch (err) {
    statusText.textContent = `Transcrição de stem: ${String(err)}`;
  } finally {
    isSeparating = false;
    syncUiControls();
  }
}

retranscribeStemBtn.addEventListener('click', () => {
  void retranscribeCurrentStem();
});

stemSelect.addEventListener('change', () => {
  applyRecommendedPresetForStem(stemSelect.value);
  if (hasSeparatedStems && !isRunning && !isRecordingSnippet) {
    void retranscribeCurrentStem();
  }
});

async function rescaleBpm(factor: number) {
  try {
    const newBpm = await invoke<number>('rescale_score_bpm', { factor });
    statusText.textContent = `Andamento ajustado para ${newBpm} BPM`;
  } catch (err) {
    statusText.textContent = String(err);
  }
}

bpmHalfBtn.addEventListener('click', () => {
  void rescaleBpm(0.5);
});

bpmDoubleBtn.addEventListener('click', () => {
  void rescaleBpm(2.0);
});

refreshBtn.addEventListener('click', () => {
  void refreshAudioSources();
});

clearBtn.addEventListener('click', async () => {
  if (isPlayerReady) api.stop();
  stemAudio.pause();
  stemPlayBtn.textContent = '▶ Ouvir Áudio';
  stemPlayBtn.classList.remove('toggle-active');
  latestNotesEl.textContent = '—';
  instrLabel.textContent = '—';
  keyLabel.textContent = '—';
  chordLabel.textContent = '—';
  bpmLabel.textContent = `100 BPM · ${tsSelect.value}`;
  a4Label.textContent = '440.0 Hz';
  timeLabel.textContent = '00:00 / 00:00';
  lastFormattedTime = '';
  await invoke('clear_transcription');
});

async function handleExport(format: 'musicxml' | 'midi') {
  try {
    const savedPath = await invoke<string>('export_score', { format });
    const fileName = savedPath.split('/').pop() ?? savedPath;
    statusText.textContent = `Exportado: ${fileName}`;
  } catch (err) {
    statusText.textContent = `Exportação: ${String(err)}`;
  }
}

exportXmlBtn.addEventListener('click', () => {
  void handleExport('musicxml');
});

exportMidBtn.addEventListener('click', () => {
  void handleExport('midi');
});

openDirBtn.addEventListener('click', async () => {
  try {
    await invoke('open_exports_folder');
  } catch (err) {
    statusText.textContent = `Erro ao abrir pasta: ${String(err)}`;
  }
});

tuningSelect.addEventListener('change', () => {
  syncInstrumentAdaptiveUi();
  if (!isRunning) {
    autoMatchFocusForInstrument(tuningSelect.value);
    if (!hasSeparatedStems) {
      void renderEmptyScore();
    }
  }
});

tsSelect.addEventListener('change', () => {
  if (!isRunning && !hasSeparatedStems) {
    void renderEmptyScore();
  }
});

viewSelect.addEventListener('change', () => {
  const profileMap: Record<string, alphaTab.StaveProfile> = {
    scoretab: alphaTab.StaveProfile.ScoreTab,
    tab: alphaTab.StaveProfile.Tab,
    score: alphaTab.StaveProfile.Score,
  };
  api.settings.display.staveProfile =
    profileMap[viewSelect.value] ?? alphaTab.StaveProfile.ScoreTab;
  api.updateSettings();
  api.render();
});

function setSelectValueIfValid(sel: HTMLSelectElement, val: string) {
  if (Array.from(sel.options).some((o) => o.value === val)) {
    sel.value = val;
  }
}

function collectCurrentPreferences(): UserPreferences {
  return {
    mode: modeSelect.value,
    stereo_mode: stereoSelect.value,
    auto_level: autolevelSelect.value === 'on',
    enable_hpss: hpssSelect.value === 'on',
    focus: focusSelect.value,
    tuning: tuningSelect.value,
    neck_region: regionSelect.value,
    prefer_closed: closedSelect.value === 'closed',
    time_signature: tsSelect.value,
    bpm: bpmSelect.value,
    view: viewSelect.value,
    stem_quality: stemQualitySelect.value,
    stem: stemSelect.value,
    playback_speed: desiredPlaybackSpeed,
    metronome_enabled: desiredMetronomeEnabled,
  };
}

function scheduleSavePreferences() {
  if (savePrefsTimer !== null) {
    window.clearTimeout(savePrefsTimer);
  }
  savePrefsTimer = window.setTimeout(() => {
    savePrefsTimer = null;
    void invoke('save_user_preferences', { prefs: collectCurrentPreferences() }).catch(() => {});
  }, 250);
}

async function restoreUserPreferences() {
  try {
    const prefs = await invoke<UserPreferences>('load_user_preferences');
    setSelectValueIfValid(modeSelect, prefs.mode);
    setSelectValueIfValid(stereoSelect, prefs.stereo_mode);
    autolevelSelect.value = prefs.auto_level ? 'on' : 'off';
    hpssSelect.value = prefs.enable_hpss ? 'on' : 'off';
    setSelectValueIfValid(focusSelect, prefs.focus);
    setSelectValueIfValid(tuningSelect, prefs.tuning);
    setSelectValueIfValid(regionSelect, prefs.neck_region);
    closedSelect.value = prefs.prefer_closed ? 'closed' : 'auto';
    setSelectValueIfValid(tsSelect, prefs.time_signature);
    setSelectValueIfValid(bpmSelect, prefs.bpm);
    setSelectValueIfValid(viewSelect, prefs.view);
    setSelectValueIfValid(stemQualitySelect, prefs.stem_quality);
    setSelectValueIfValid(stemSelect, prefs.stem);
    setSelectValueIfValid(speedSelect, String(prefs.playback_speed));

    desiredPlaybackSpeed = Number(speedSelect.value) || 1.0;
    desiredMetronomeEnabled = Boolean(prefs.metronome_enabled);

    syncInstrumentAdaptiveUi();

    const profileMap: Record<string, alphaTab.StaveProfile> = {
      scoretab: alphaTab.StaveProfile.ScoreTab,
      tab: alphaTab.StaveProfile.Tab,
      score: alphaTab.StaveProfile.Score,
    };
    if (profileMap[viewSelect.value] !== undefined) {
      api.settings.display.staveProfile = profileMap[viewSelect.value];
      api.updateSettings();
    }
  } catch {
    // Mantém padrões na primeira execução
  }
}

async function refreshSavedSessions(selectedDir?: string) {
  try {
    const sessions = await invoke<StoredStemSessionDto[]>('list_stem_sessions');
    const prev = selectedDir ?? savedSessionsSelect.value;
    savedSessionsSelect.innerHTML = '';

    const placeholder = document.createElement('option');
    placeholder.value = '';
    placeholder.textContent =
      sessions.length > 0 ? `Sessões Salvas (${sessions.length})` : 'Nenhuma sessão salva';
    savedSessionsSelect.appendChild(placeholder);

    for (const s of sessions) {
      const opt = document.createElement('option');
      opt.value = s.session_dir;
      const stemsHint = s.active_stems.length > 0 ? ` · ${s.active_stems.join('/')}` : '';
      opt.textContent = `${s.label}${stemsHint}`;
      savedSessionsSelect.appendChild(opt);
    }

    if (prev && Array.from(savedSessionsSelect.options).some((o) => o.value === prev)) {
      savedSessionsSelect.value = prev;
    }
  } catch {
    // Ignora se a pasta ainda não existe
  }
}

savedSessionsSelect.addEventListener('change', async () => {
  const sessionDir = savedSessionsSelect.value;
  if (!sessionDir || isSeparating || isTransitioning || isRecordingSnippet) return;

  if (isPlayerReady) api.stop();
  clearStemAudioCache();
  setStemAudioControlsEnabled(false);
  isRunning = false;
  isSeparating = true;
  syncUiControls();

  try {
    const res = await invoke<StemSeparationResultDto>('load_stem_session', {
      sessionDir,
      config: buildStemTranscribeConfig(),
    });
    hasSeparatedStems = true;
    updateStemSelectWithActivities(res.stem_activities);
    stemSelect.value = res.active_stem;
    setStemAudioControlsEnabled(true);

    await loadStemIntoAudioElement(res.active_stem, false);

    const activeCount = res.stem_activities.filter((a) => a.active).length;
    stemsStatusEl.textContent = `Sessão reaberta · ${activeCount}/6 stems (${res.engine})`;
    stemsStatusEl.classList.add('ready');
    statusText.textContent = `Stem '${res.active_stem}': ${res.event_count} eventos (${res.transcription_ms} ms)`;
  } catch (err) {
    stemsStatusEl.textContent = `Erro ao reabrir: ${String(err)}`;
    stemsStatusEl.classList.remove('ready');
    statusText.textContent = `Erro ao carregar sessão: ${String(err)}`;
  } finally {
    isSeparating = false;
    syncUiControls();
  }
});

for (const el of [
  modeSelect,
  stereoSelect,
  autolevelSelect,
  hpssSelect,
  focusSelect,
  tuningSelect,
  regionSelect,
  closedSelect,
  tsSelect,
  bpmSelect,
  viewSelect,
  stemQualitySelect,
  stemSelect,
]) {
  el.addEventListener('change', scheduleSavePreferences);
}

async function bootstrap() {
  await listen<TranscriptionUpdatePayload>('transcription-update', (event) => {
    const {
      alphatex,
      latest_notes,
      latest_chord,
      detected_key,
      detected_instrument,
      time_signature,
      event_count,
      latency_ms,
      detected_bpm,
      estimated_a4_hz,
      pruned_overtones,
      bend_semitones,
      is_reattack,
    } = event.payload;

    syncStaveProfileFromTex(alphatex);
      api.tex(alphatex);

    let noteDisplay = latest_notes.length > 0 ? latest_notes.join(' + ') : '—';
    if (bend_semitones >= 0.45) {
      noteDisplay += ` ↗+${bend_semitones.toFixed(1)}st`;
    } else if (is_reattack) {
      noteDisplay += ' ↺';
    }
    latestNotesEl.textContent = noteDisplay;
    instrLabel.textContent = detected_instrument || '—';
    keyLabel.textContent = detected_key || '—';
    chordLabel.textContent = latest_chord ?? '—';
    bpmLabel.textContent = `${detected_bpm} BPM · ${time_signature}`;
    a4Label.textContent = `${estimated_a4_hz.toFixed(1)} Hz`;

    if (isRunning) {
      const prunedTxt = pruned_overtones > 0 ? ` | harmônicos podados: ${pruned_overtones}` : '';
      statusText.textContent = `${event_count} eventos | ${latency_ms} ms${prunedTxt}`;
    }
  });

  await listen<AudioLevelPayload>('audio-level', (event) => {
    const {
      rms_db,
      peak_db,
      silence_warning,
      dropped_samples,
      device_switches,
      clip_events,
      clock_glitches,
      phase_corr,
      sample_rate,
      format_label,
    } = event.payload;
    const pct = Math.max(0, Math.min(100, ((rms_db + 60) / 60) * 100));
    meterFill.style.width = `${pct.toFixed(1)}%`;
    const khz = (sample_rate / 1000).toFixed(sample_rate % 1000 === 0 ? 0 : 1);
    const clipTag = clip_events > 0 ? ' · CLIP' : '';
    const glitchTag = clock_glitches > 0 ? ` · GLITCH(${clock_glitches})` : '';
    phaseLabel.textContent = (phase_corr >= 0 ? '+' : '') + phase_corr.toFixed(2);
    dbLabel.textContent =
      rms_db <= -110
        ? `-∞ dBFS (${khz}k ${format_label})`
        : `${rms_db.toFixed(1)} dB · pk ${peak_db.toFixed(1)} (${khz}k ${format_label}${clipTag}${glitchTag})`;
    warningBanner.classList.toggle('hidden', !silence_warning);

    if (device_switches > lastDeviceSwitches) {
      lastDeviceSwitches = device_switches;
      statusText.textContent = `Saída alterada (${khz} kHz ${format_label}) — loopback reconectado`;
    } else if (dropped_samples > 0 && isRunning) {
      statusText.textContent = `Drops no buffer: ${dropped_samples}`;
    }
  });

  await listen<SnippetProgressPayload>('snippet-progress', (event) => {
    recTimerEl.textContent = formatDuration(event.payload.elapsed_sec * 1000);
  });

  await listen<string>('stem-status', (event) => {
    stemsStatusEl.textContent = event.payload;
    statusText.textContent = event.payload;
  });

  await listen<string>('transcription-error', (event) => {
    statusText.textContent = `Erro no motor: ${event.payload}`;
  });

  await restoreUserPreferences();
  syncInstrumentAdaptiveUi();
  syncUiControls();
  await Promise.all([refreshAudioSources(), checkStemsAvailability(), refreshSavedSessions(), renderEmptyScore()]);
}

void bootstrap();
