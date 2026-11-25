use std::ffi::{c_char, c_void, CStr};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use rtrb::{Producer, RingBuffer};

use super::{
    AudioCapturePort, AudioProcessInfo, AudioStreamInfo, AudioStreamPair, CaptureTarget,
    StereoFrame,
};

#[repr(C)]
struct MacOSTapHandle {
    _private: [u8; 0],
}

#[repr(C)]
struct CMacOSAudioProcessInfo {
    object_id: u32,
    pid: i32,
    is_running_output: u8,
    name: [c_char; 128],
    bundle_id: [c_char; 256],
}

type StereoSamplesCallback =
    extern "C" fn(interleaved_lr: *const f32, frame_count: u32, user_data: *mut c_void);

extern "C" {
    fn macos_list_audio_processes(out_list: *mut *mut CMacOSAudioProcessInfo) -> u32;
    fn macos_free_audio_processes(list: *mut CMacOSAudioProcessInfo);

    fn macos_tap_start(
        target_object_ids: *const u32,
        target_count: u32,
        callback: StereoSamplesCallback,
        user_data: *mut c_void,
        out_sample_rate: *mut f64,
        out_status: *mut i32,
    ) -> *mut MacOSTapHandle;

    fn macos_tap_device_switches(handle: *mut MacOSTapHandle) -> u64;
    fn macos_tap_clip_events(handle: *mut MacOSTapHandle) -> u64;
    fn macos_tap_clock_glitches(handle: *mut MacOSTapHandle) -> u64;
    fn macos_tap_current_sample_rate(handle: *mut MacOSTapHandle) -> f64;
    fn macos_tap_format_info(handle: *mut MacOSTapHandle, out_bits: *mut u32, out_is_float: *mut u32);
    fn macos_tap_stop(handle: *mut MacOSTapHandle);
}

fn c_buf_to_string(buf: &[c_char]) -> String {
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

pub fn list_audio_processes(only_active: bool) -> Vec<AudioProcessInfo> {
    let mut raw_ptr: *mut CMacOSAudioProcessInfo = ptr::null_mut();
    let count = unsafe { macos_list_audio_processes(&mut raw_ptr) };

    if count == 0 || raw_ptr.is_null() {
        return Vec::new();
    }

    let slice = unsafe { std::slice::from_raw_parts(raw_ptr, count as usize) };
    let mut processes: Vec<AudioProcessInfo> = slice
        .iter()
        .map(|entry| AudioProcessInfo {
            object_id: entry.object_id,
            pid: entry.pid,
            name: c_buf_to_string(&entry.name),
            bundle_id: c_buf_to_string(&entry.bundle_id),
            is_running_output: entry.is_running_output != 0,
        })
        .filter(|p| !only_active || p.is_running_output)
        .collect();

    unsafe { macos_free_audio_processes(raw_ptr) };

    processes.sort_by(|a, b| {
        b.is_running_output
            .cmp(&a.is_running_output)
            .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
    });

    processes
}

pub fn find_processes_by_query(query: &str) -> Vec<AudioProcessInfo> {
    let q = query.to_lowercase();
    list_audio_processes(false)
        .into_iter()
        .filter(|p| p.name.to_lowercase().contains(&q) || p.bundle_id.to_lowercase().contains(&q))
        .collect()
}

pub fn find_process_by_pid(pid: i32) -> Option<AudioProcessInfo> {
    list_audio_processes(false)
        .into_iter()
        .find(|p| p.pid == pid)
}

struct CallbackContext {
    producer: Producer<StereoFrame>,
    dropped_samples: Arc<AtomicU64>,
}

extern "C" fn rust_audio_callback(
    interleaved_lr: *const f32,
    frame_count: u32,
    user_data: *mut c_void,
) {
    if interleaved_lr.is_null() || frame_count == 0 || user_data.is_null() {
        return;
    }

    let ctx = unsafe { &mut *(user_data as *mut CallbackContext) };
    let frames = unsafe {
        std::slice::from_raw_parts(interleaved_lr as *const StereoFrame, frame_count as usize)
    };

    let (_written, remainder) = ctx.producer.push_partial_slice(frames);
    if !remainder.is_empty() {
        ctx.dropped_samples
            .fetch_add(remainder.len() as u64, Ordering::Relaxed);
    }
}

pub struct MacOSLoopbackCapture {
    handle: *mut MacOSTapHandle,
    ctx_ptr: *mut CallbackContext,
    info: AudioStreamInfo,
    dropped_samples: Arc<AtomicU64>,
}

impl MacOSLoopbackCapture {
    pub fn start(ring_capacity_frames: usize) -> Result<AudioStreamPair<Self>, String> {
        Self::start_with_target(ring_capacity_frames, CaptureTarget::SystemGlobal)
    }

    pub fn start_with_target(
        ring_capacity_frames: usize,
        target: CaptureTarget,
    ) -> Result<AudioStreamPair<Self>, String> {
        let (target_ptr, target_len, _owned_ids) = match target {
            CaptureTarget::SystemGlobal => (ptr::null(), 0u32, Vec::new()),
            CaptureTarget::ProcessObjects(ids) => {
                if ids.is_empty() {
                    return Err("Nenhum processo selecionado para captura.".to_string());
                }
                let ptr = ids.as_ptr();
                let len = ids.len() as u32;
                (ptr, len, ids)
            }
        };

        let (producer, consumer) = RingBuffer::<StereoFrame>::new(ring_capacity_frames);
        let dropped_samples = Arc::new(AtomicU64::new(0));

        let ctx = Box::new(CallbackContext {
            producer,
            dropped_samples: Arc::clone(&dropped_samples),
        });
        let ctx_ptr = Box::into_raw(ctx);

        let mut sample_rate: f64 = 0.0;
        let mut status: i32 = 0;

        let mut handle = unsafe {
            macos_tap_start(
                target_ptr,
                target_len,
                rust_audio_callback,
                ctx_ptr as *mut c_void,
                &mut sample_rate,
                &mut status,
            )
        };

        if handle.is_null() {
            std::thread::sleep(std::time::Duration::from_millis(150));
            handle = unsafe {
                macos_tap_start(
                    target_ptr,
                    target_len,
                    rust_audio_callback,
                    ctx_ptr as *mut c_void,
                    &mut sample_rate,
                    &mut status,
                )
            };
        }

        if handle.is_null() {
            unsafe { drop(Box::from_raw(ctx_ptr)) };
            return Err(format!(
                "Falha ao iniciar Core Audio Process Tap (OSStatus: {})",
                status
            ));
        }

        let capture = Self {
            handle,
            ctx_ptr,
            info: AudioStreamInfo {
                sample_rate,
                channels: 2,
            },
            dropped_samples,
        };

        Ok((capture, consumer))
    }
}

impl AudioCapturePort for MacOSLoopbackCapture {
    fn stream_info(&self) -> AudioStreamInfo {
        self.info
    }

    fn dropped_samples(&self) -> u64 {
        self.dropped_samples.load(Ordering::Relaxed)
    }

    fn device_switches(&self) -> u64 {
        unsafe { macos_tap_device_switches(self.handle) }
    }

    fn clip_events(&self) -> u64 {
        unsafe { macos_tap_clip_events(self.handle) }
    }

    fn clock_glitches(&self) -> u64 {
        unsafe { macos_tap_clock_glitches(self.handle) }
    }

    fn current_sample_rate(&self) -> f64 {
        unsafe { macos_tap_current_sample_rate(self.handle) }
    }

    fn format_label(&self) -> String {
        let mut bits = 32u32;
        let mut is_float = 1u32;
        unsafe { macos_tap_format_info(self.handle, &mut bits, &mut is_float) };
        if is_float != 0 {
            format!("F{}", bits)
        } else {
            format!("I{}", bits)
        }
    }
}

impl Drop for MacOSLoopbackCapture {
    fn drop(&mut self) {
        unsafe {
            macos_tap_stop(self.handle);
            drop(Box::from_raw(self.ctx_ptr));
        }
    }
}
