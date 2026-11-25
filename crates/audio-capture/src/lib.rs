pub mod capture;

pub use capture::{AudioCapturePort, AudioProcessInfo, AudioStreamInfo, AudioStreamPair, CaptureTarget};
#[cfg(target_os = "macos")]
pub use capture::macos::{
    find_process_by_pid, find_processes_by_query, list_audio_processes, MacOSLoopbackCapture,
};
