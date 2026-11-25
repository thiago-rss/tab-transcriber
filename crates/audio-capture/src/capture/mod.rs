#[cfg(target_os = "macos")]
pub mod macos;

use rtrb::Consumer;

pub type StereoFrame = [f32; 2];

#[derive(Debug, Clone, Copy)]
pub struct AudioStreamInfo {
    pub sample_rate: f64,
    pub channels: u16,
}

#[derive(Debug, Clone)]
pub struct AudioProcessInfo {
    pub object_id: u32,
    pub pid: i32,
    pub name: String,
    pub bundle_id: String,
    pub is_running_output: bool,
}

#[derive(Debug, Clone)]
pub enum CaptureTarget {
    SystemGlobal,
    ProcessObjects(Vec<u32>),
}

pub trait AudioCapturePort {
    fn stream_info(&self) -> AudioStreamInfo;
    fn dropped_samples(&self) -> u64;
    fn device_switches(&self) -> u64 {
        0
    }
    fn clip_events(&self) -> u64 {
        0
    }
    fn clock_glitches(&self) -> u64 {
        0
    }
    fn current_sample_rate(&self) -> f64 {
        self.stream_info().sample_rate
    }
    fn format_label(&self) -> String {
        "F32".to_string()
    }
}

pub type AudioStreamPair<T> = (T, Consumer<StereoFrame>);
