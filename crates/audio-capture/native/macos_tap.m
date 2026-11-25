#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>
#import <CoreAudio/CoreAudio.h>
#import <CoreAudio/AudioHardwareTapping.h>
#import <CoreAudio/CATapDescription.h>
#import <dispatch/dispatch.h>
#include <libproc.h>
#include <math.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef void (*StereoSamplesCallback)(const float *interleaved_lr, uint32_t frame_count, void *user_data);

typedef struct {
    uint32_t object_id;
    int32_t  pid;
    uint8_t  is_running_output;
    char     name[128];
    char     bundle_id[256];
} MacOSAudioProcessInfo;

typedef struct MacOSTapHandle MacOSTapHandle;

static _Atomic uint64_t g_tap_generation = 0;

// Contexto 100% C puro acessado pela thread de tempo real do Core Audio.
typedef struct {
    StereoSamplesCallback callback;
    void *user_data;
    _Atomic uint64_t device_switches;
    _Atomic uint64_t clip_events;
    _Atomic uint64_t clock_glitches;
    _Atomic uint32_t sample_rate_hz;
    _Atomic uint32_t format_flags;
    _Atomic uint32_t bits_per_channel;
    _Atomic uint8_t  active;
    double last_end_sample_time;
    uint8_t has_sample_time;
} RealtimeTapCtx;

static AudioObjectID get_default_output_device_id(void) {
    AudioObjectPropertyAddress dev_addr = {
        kAudioHardwarePropertyDefaultOutputDevice,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    AudioObjectID dev_id = kAudioObjectUnknown;
    UInt32 size = sizeof(dev_id);

    if (AudioObjectGetPropertyData(kAudioObjectSystemObject, &dev_addr, 0, NULL, &size, &dev_id) != noErr) {
        return kAudioObjectUnknown;
    }
    return dev_id;
}

static NSString *get_device_uid(AudioObjectID dev_id) {
    if (dev_id == kAudioObjectUnknown) return nil;

    CFStringRef uid_ref = NULL;
    UInt32 uid_size = sizeof(uid_ref);
    AudioObjectPropertyAddress uid_addr = {
        kAudioDevicePropertyDeviceUID,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };

    if (AudioObjectGetPropertyData(dev_id, &uid_addr, 0, NULL, &uid_size, &uid_ref) != noErr || !uid_ref) {
        return nil;
    }
    return CFBridgingRelease(uid_ref);
}

static double get_device_nominal_sample_rate(AudioObjectID dev_id) {
    if (dev_id == kAudioObjectUnknown) return 0.0;
    Float64 sr = 0.0;
    UInt32 size = sizeof(sr);
    AudioObjectPropertyAddress sr_addr = {
        kAudioDevicePropertyNominalSampleRate,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    if (AudioObjectGetPropertyData(dev_id, &sr_addr, 0, NULL, &size, &sr) == noErr && sr > 1000.0) {
        return (double)sr;
    }
    return 0.0;
}

static AudioObjectID get_self_process_object_id(void) {
    pid_t pid = getpid();
    AudioObjectPropertyAddress addr = {
        kAudioHardwarePropertyTranslatePIDToProcessObject,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    AudioObjectID proc_id = kAudioObjectUnknown;
    UInt32 size = sizeof(proc_id);

    if (AudioObjectGetPropertyData(kAudioObjectSystemObject, &addr, sizeof(pid), &pid, &size, &proc_id) != noErr) {
        return kAudioObjectUnknown;
    }
    return proc_id;
}

uint32_t macos_list_audio_processes(MacOSAudioProcessInfo **out_list) {
    if (!out_list) return 0;
    *out_list = NULL;

    @autoreleasepool {
        AudioObjectPropertyAddress list_addr = {
            kAudioHardwarePropertyProcessObjectList,
            kAudioObjectPropertyScopeGlobal,
            kAudioObjectPropertyElementMain
        };

        UInt32 data_size = 0;
        if (AudioObjectGetPropertyDataSize(kAudioObjectSystemObject, &list_addr, 0, NULL, &data_size) != noErr ||
            data_size == 0) {
            return 0;
        }

        uint32_t raw_count = data_size / sizeof(AudioObjectID);
        AudioObjectID *object_ids = (AudioObjectID *)malloc(data_size);
        if (!object_ids) return 0;

        if (AudioObjectGetPropertyData(kAudioObjectSystemObject, &list_addr, 0, NULL, &data_size, object_ids) != noErr) {
            free(object_ids);
            return 0;
        }

        MacOSAudioProcessInfo *results = (MacOSAudioProcessInfo *)calloc(raw_count, sizeof(MacOSAudioProcessInfo));
        if (!results) {
            free(object_ids);
            return 0;
        }

        pid_t self_pid = getpid();
        uint32_t out_count = 0;

        AudioObjectPropertyAddress pid_addr = {
            kAudioProcessPropertyPID,
            kAudioObjectPropertyScopeGlobal,
            kAudioObjectPropertyElementMain
        };
        AudioObjectPropertyAddress running_addr = {
            kAudioProcessPropertyIsRunningOutput,
            kAudioObjectPropertyScopeGlobal,
            kAudioObjectPropertyElementMain
        };
        AudioObjectPropertyAddress bundle_addr = {
            kAudioProcessPropertyBundleID,
            kAudioObjectPropertyScopeGlobal,
            kAudioObjectPropertyElementMain
        };

        for (uint32_t i = 0; i < raw_count; i++) {
            AudioObjectID obj_id = object_ids[i];

            pid_t pid = -1;
            UInt32 pid_size = sizeof(pid);
            if (AudioObjectGetPropertyData(obj_id, &pid_addr, 0, NULL, &pid_size, &pid) != noErr ||
                pid <= 0 || pid == self_pid) {
                continue;
            }

            UInt32 is_running = 0;
            UInt32 running_size = sizeof(is_running);
            AudioObjectGetPropertyData(obj_id, &running_addr, 0, NULL, &running_size, &is_running);

            CFStringRef bundle_ref = NULL;
            UInt32 bundle_size = sizeof(bundle_ref);
            NSString *bundle_str = @"";
            if (AudioObjectGetPropertyData(obj_id, &bundle_addr, 0, NULL, &bundle_size, &bundle_ref) == noErr &&
                bundle_ref != NULL) {
                bundle_str = CFBridgingRelease(bundle_ref);
            }

            NSString *display_name = nil;
            NSRunningApplication *app = [NSRunningApplication runningApplicationWithProcessIdentifier:pid];
            if (app && app.localizedName.length > 0) {
                display_name = app.localizedName;
            } else {
                char raw_name[PROC_PIDPATHINFO_MAXSIZE] = {0};
                if (proc_name(pid, raw_name, sizeof(raw_name)) > 0) {
                    display_name = [NSString stringWithUTF8String:raw_name];
                }
            }
            if (!display_name || display_name.length == 0) {
                display_name = bundle_str.length > 0 ? bundle_str : @"Unknown";
            }

            MacOSAudioProcessInfo *entry = &results[out_count++];
            entry->object_id = (uint32_t)obj_id;
            entry->pid = (int32_t)pid;
            entry->is_running_output = (is_running != 0) ? 1 : 0;

            const char *name_utf8 = [display_name UTF8String];
            const char *bundle_utf8 = [bundle_str UTF8String];
            strlcpy(entry->name, name_utf8 ? name_utf8 : "", sizeof(entry->name));
            strlcpy(entry->bundle_id, bundle_utf8 ? bundle_utf8 : "", sizeof(entry->bundle_id));
        }

        free(object_ids);

        if (out_count == 0) {
            free(results);
            return 0;
        }

        *out_list = results;
        return out_count;
    }
}

void macos_free_audio_processes(MacOSAudioProcessInfo *list) {
    if (list) free(list);
}

static inline float read_pcm_sample(
    const uint8_t *base,
    uint32_t index,
    uint32_t bits,
    uint32_t flags
) {
    if ((flags & kAudioFormatFlagIsFloat) != 0) {
        if (bits == 64) {
            return (float)((const double *)base)[index];
        }
        return ((const float *)base)[index];
    }

    if ((flags & kAudioFormatFlagIsSignedInteger) != 0) {
        if (bits == 16) {
            return (float)((const int16_t *)base)[index] / 32768.0f;
        }
        if (bits == 24) {
            const uint8_t *p = base + (index * 3);
            int32_t v = (int32_t)p[0] | ((int32_t)p[1] << 8) | ((int32_t)((int8_t)p[2]) << 16);
            return (float)v / 8388608.0f;
        }
        if (bits == 32) {
            return (float)((const int32_t *)base)[index] / 2147483648.0f;
        }
    }

    return ((const float *)base)[index];
}

static OSStatus io_proc_callback(
    AudioObjectID inDevice,
    const AudioTimeStamp *inNow,
    const AudioBufferList *inInputData,
    const AudioTimeStamp *inInputTime,
    AudioBufferList *outOutputData,
    const AudioTimeStamp *inOutputTime,
    void *inClientData
) {
    (void)inDevice; (void)inNow; (void)outOutputData; (void)inOutputTime;

    RealtimeTapCtx *ctx = (RealtimeTapCtx *)inClientData;
    if (!ctx || !atomic_load_explicit(&ctx->active, memory_order_acquire) ||
        !ctx->callback || !inInputData || inInputData->mNumberBuffers == 0) {
        return noErr;
    }

    uint32_t bits = atomic_load_explicit(&ctx->bits_per_channel, memory_order_relaxed);
    uint32_t flags = atomic_load_explicit(&ctx->format_flags, memory_order_relaxed);
    if (bits == 0) bits = 32;
    uint32_t bytes_per_sample = bits / 8;
    if (bytes_per_sample == 0) bytes_per_sample = 4;

    int first_idx = -1;
    for (uint32_t b = 0; b < inInputData->mNumberBuffers; b++) {
        if (inInputData->mBuffers[b].mData != NULL && inInputData->mBuffers[b].mDataByteSize > 0) {
            first_idx = (int)b;
            break;
        }
    }
    if (first_idx < 0) return noErr;

    float stereo_chunk[512 * 2];
    int clipped_in_cb = 0;
    uint32_t total_frames = 0;

    const AudioBuffer *b0 = &inInputData->mBuffers[first_idx];
    uint32_t ch0 = b0->mNumberChannels > 0 ? b0->mNumberChannels : 1;

    if (ch0 >= 2) {
        const uint8_t *src = (const uint8_t *)b0->mData;
        uint32_t frame_bytes = bytes_per_sample * ch0;
        total_frames = b0->mDataByteSize / frame_bytes;

        uint32_t offset = 0;
        while (offset < total_frames) {
            uint32_t n = total_frames - offset;
            if (n > 512) n = 512;

            for (uint32_t f = 0; f < n; f++) {
                uint32_t base = (offset + f) * ch0;
                float l = read_pcm_sample(src, base, bits, flags);
                float r = read_pcm_sample(src, base + 1, bits, flags);
                if (!isfinite(l)) l = 0.0f;
                if (!isfinite(r)) r = 0.0f;
                if (fabsf(l) >= 0.999f || fabsf(r) >= 0.999f) clipped_in_cb = 1;
                stereo_chunk[2 * f] = l;
                stereo_chunk[2 * f + 1] = r;
            }
            ctx->callback(stereo_chunk, n, ctx->user_data);
            offset += n;
        }
    } else {
        int second_idx = -1;
        for (uint32_t b = (uint32_t)(first_idx + 1); b < inInputData->mNumberBuffers; b++) {
            if (inInputData->mBuffers[b].mData != NULL &&
                inInputData->mBuffers[b].mDataByteSize > 0 &&
                inInputData->mBuffers[b].mNumberChannels == 1) {
                second_idx = (int)b;
                break;
            }
        }

        const uint8_t *left = (const uint8_t *)b0->mData;
        const uint8_t *right = (second_idx >= 0) ? (const uint8_t *)inInputData->mBuffers[second_idx].mData : left;
        total_frames = b0->mDataByteSize / bytes_per_sample;
        if (second_idx >= 0) {
            uint32_t r_frames = inInputData->mBuffers[second_idx].mDataByteSize / bytes_per_sample;
            if (r_frames < total_frames) total_frames = r_frames;
        }

        uint32_t offset = 0;
        while (offset < total_frames) {
            uint32_t n = total_frames - offset;
            if (n > 512) n = 512;

            for (uint32_t f = 0; f < n; f++) {
                float l = read_pcm_sample(left, offset + f, bits, flags);
                float r = read_pcm_sample(right, offset + f, bits, flags);
                if (!isfinite(l)) l = 0.0f;
                if (!isfinite(r)) r = 0.0f;
                if (fabsf(l) >= 0.999f || fabsf(r) >= 0.999f) clipped_in_cb = 1;
                stereo_chunk[2 * f] = l;
                stereo_chunk[2 * f + 1] = r;
            }
            ctx->callback(stereo_chunk, n, ctx->user_data);
            offset += n;
        }
    }

    if (inInputTime && (inInputTime->mFlags & kAudioTimeStampSampleTimeValid) && total_frames > 0) {
        if (ctx->has_sample_time) {
            double gap = fabs(inInputTime->mSampleTime - ctx->last_end_sample_time);
            if (gap > 4.0) {
                atomic_fetch_add_explicit(&ctx->clock_glitches, 1, memory_order_relaxed);
            }
        }
        ctx->last_end_sample_time = inInputTime->mSampleTime + (double)total_frames;
        ctx->has_sample_time = 1;
    }

    if (clipped_in_cb) {
        atomic_fetch_add_explicit(&ctx->clip_events, 1, memory_order_relaxed);
    }

    return noErr;
}

@interface MacOSTapSession : NSObject
@property (nonatomic, assign) AudioObjectID tapId;
@property (nonatomic, assign) AudioObjectID aggregateId;
@property (nonatomic, assign) AudioDeviceIOProcID ioProcId;
@property (nonatomic, assign) uint64_t generation;
@property (nonatomic, copy) NSString *tapUuid;
@property (nonatomic, copy) NSString *currentOutputUid;
@property (nonatomic, strong) dispatch_queue_t queue;
@property (nonatomic, copy) AudioObjectPropertyListenerBlock defaultOutListener;
@property (nonatomic, copy) AudioObjectPropertyListenerBlock formatListener;
- (BOOL)startAggregateForOutputDevice:(AudioObjectID)outputDevId outStatus:(int32_t *)outStatus;
- (void)teardownAggregateUnlocked;
- (void)rebuildIfOutputChangedUnlocked;
- (void)stopAndDestroy;
- (uint64_t)deviceSwitchCount;
- (uint64_t)clipEventCount;
- (uint64_t)clockGlitchCount;
- (double)currentSampleRate;
- (void)formatInfoBits:(uint32_t *)outBits isFloat:(uint32_t *)outIsFloat;
@end

@implementation MacOSTapSession {
    RealtimeTapCtx *_rtCtx;
    _Atomic bool _stopping;
}

- (instancetype)initWithRealtimeCtx:(RealtimeTapCtx *)rtCtx generation:(uint64_t)gen {
    self = [super init];
    if (self) {
        _tapId = kAudioObjectUnknown;
        _aggregateId = kAudioObjectUnknown;
        _ioProcId = NULL;
        _generation = gen;
        _rtCtx = rtCtx;
        atomic_init(&_stopping, false);
        NSString *qName = [NSString stringWithFormat:@"com.tabtranscriber.coreaudio.watch.%llu", gen];
        _queue = dispatch_queue_create([qName UTF8String], DISPATCH_QUEUE_SERIAL);
    }
    return self;
}

- (void)configureOptimalBufferSize:(AudioObjectID)deviceId {
    AudioObjectPropertyAddress range_addr = {
        kAudioDevicePropertyBufferFrameSizeRange,
        kAudioObjectPropertyScopeInput,
        kAudioObjectPropertyElementMain
    };
    AudioValueRange range = { 64.0, 4096.0 };
    UInt32 range_size = sizeof(range);
    if (AudioObjectGetPropertyData(deviceId, &range_addr, 0, NULL, &range_size, &range) != noErr) {
        range_addr.mScope = kAudioObjectPropertyScopeGlobal;
        AudioObjectGetPropertyData(deviceId, &range_addr, 0, NULL, &range_size, &range);
    }

    UInt32 target = 512;
    if (target < (UInt32)range.mMinimum) target = (UInt32)range.mMinimum;
    if (target > (UInt32)range.mMaximum) target = (UInt32)range.mMaximum;

    AudioObjectPropertyAddress buf_addr = {
        kAudioDevicePropertyBufferFrameSize,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    AudioObjectSetPropertyData(deviceId, &buf_addr, 0, NULL, sizeof(target), &target);
}

- (void)inspectAndPublishStreamFormat:(AudioObjectID)aggregateId {
    AudioStreamBasicDescription asbd = {0};
    UInt32 asbd_size = sizeof(asbd);
    BOOL got_asbd = NO;

    AudioObjectPropertyAddress in_fmt_addr = {
        kAudioDevicePropertyStreamFormat,
        kAudioObjectPropertyScopeInput,
        kAudioObjectPropertyElementMain
    };
    if (AudioObjectGetPropertyData(aggregateId, &in_fmt_addr, 0, NULL, &asbd_size, &asbd) == noErr &&
        asbd.mSampleRate > 0) {
        got_asbd = YES;
    }

    if (!got_asbd && self.tapId != kAudioObjectUnknown) {
        AudioObjectPropertyAddress tap_fmt_addr = {
            kAudioTapPropertyFormat,
            kAudioObjectPropertyScopeGlobal,
            kAudioObjectPropertyElementMain
        };
        if (AudioObjectGetPropertyData(self.tapId, &tap_fmt_addr, 0, NULL, &asbd_size, &asbd) == noErr &&
            asbd.mSampleRate > 0) {
            got_asbd = YES;
        }
    }

    double nominal_sr = get_device_nominal_sample_rate(aggregateId);
    double effective_sr = (nominal_sr > 1000.0)
        ? nominal_sr
        : ((got_asbd && asbd.mSampleRate > 1000.0) ? asbd.mSampleRate : 48000.0);

    uint32_t bits = (got_asbd && asbd.mBitsPerChannel > 0) ? (uint32_t)asbd.mBitsPerChannel : 32;
    uint32_t flags = got_asbd ? (uint32_t)asbd.mFormatFlags : (uint32_t)kAudioFormatFlagIsFloat;

    if (_rtCtx) {
        atomic_store(&_rtCtx->sample_rate_hz, (uint32_t)llround(effective_sr));
        atomic_store(&_rtCtx->bits_per_channel, bits);
        atomic_store(&_rtCtx->format_flags, flags);
    }
}

- (void)removeAggregateListenersUnlocked {
    if (self.aggregateId == kAudioObjectUnknown || !self.formatListener) {
        return;
    }
    AudioObjectPropertyAddress sr_addr = {
        kAudioDevicePropertyNominalSampleRate,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    AudioObjectPropertyAddress fmt_addr = {
        kAudioDevicePropertyStreamFormat,
        kAudioObjectPropertyScopeInput,
        kAudioObjectPropertyElementMain
    };
    AudioObjectRemovePropertyListenerBlock(self.aggregateId, &sr_addr, self.queue, self.formatListener);
    AudioObjectRemovePropertyListenerBlock(self.aggregateId, &fmt_addr, self.queue, self.formatListener);
    self.formatListener = nil;
}

- (void)teardownAggregateUnlocked {
    [self removeAggregateListenersUnlocked];
    if (_rtCtx) {
        atomic_store_explicit(&_rtCtx->active, 0, memory_order_release);
    }
    if (self.aggregateId != kAudioObjectUnknown && self.ioProcId != NULL) {
        AudioDeviceStop(self.aggregateId, self.ioProcId);
        // Pequeno tempo de acomodação para o último IOProc em voo encerrar antes de destruir o ID
        usleep(20000);
        AudioDeviceDestroyIOProcID(self.aggregateId, self.ioProcId);
        self.ioProcId = NULL;
    }
    if (self.aggregateId != kAudioObjectUnknown) {
        AudioHardwareDestroyAggregateDevice(self.aggregateId);
        self.aggregateId = kAudioObjectUnknown;
    }
}

- (BOOL)startAggregateForOutputDevice:(AudioObjectID)outputDevId outStatus:(int32_t *)outStatus {
    NSString *outputUid = get_device_uid(outputDevId);
    if (!outputUid || !self.tapUuid) {
        if (outStatus) *outStatus = -1;
        return NO;
    }

    double out_sr = get_device_nominal_sample_rate(outputDevId);

    NSMutableDictionary *subtap = [NSMutableDictionary dictionaryWithDictionary:@{
        @(kAudioSubTapUIDKey): self.tapUuid,
        @(kAudioSubTapDriftCompensationKey): @YES,
    }];
#if defined(kAudioSubTapDriftCompensationQualityKey) && defined(kAudioAggregateDriftCompensationMaxQuality)
    subtap[@(kAudioSubTapDriftCompensationQualityKey)] = @(kAudioAggregateDriftCompensationMaxQuality);
#endif

    NSString *aggName = [NSString stringWithFormat:@"LoopbackTabAgg_%llu", self.generation];
    NSDictionary *agg_desc = @{
        @(kAudioAggregateDeviceNameKey): aggName,
        @(kAudioAggregateDeviceUIDKey): [[NSUUID UUID] UUIDString],
        @(kAudioAggregateDeviceClockDeviceKey): outputUid,
        @(kAudioAggregateDeviceMainSubDeviceKey): outputUid,
        @(kAudioAggregateDeviceIsPrivateKey): @YES,
        @(kAudioAggregateDeviceIsStackedKey): @NO,
        @(kAudioAggregateDeviceTapAutoStartKey): @YES,
        @(kAudioAggregateDeviceTapListKey): @[ subtap ]
    };

    AudioObjectID aggregate_id = kAudioObjectUnknown;
    OSStatus status = AudioHardwareCreateAggregateDevice((__bridge CFDictionaryRef)agg_desc, &aggregate_id);
    if (status != noErr) {
        if (outStatus) *outStatus = (int32_t)status;
        return NO;
    }

    if (out_sr > 1000.0) {
        Float64 desired_sr = (Float64)out_sr;
        AudioObjectPropertyAddress sr_addr = {
            kAudioDevicePropertyNominalSampleRate,
            kAudioObjectPropertyScopeGlobal,
            kAudioObjectPropertyElementMain
        };
        AudioObjectSetPropertyData(aggregate_id, &sr_addr, 0, NULL, sizeof(desired_sr), &desired_sr);
    }

    [self configureOptimalBufferSize:aggregate_id];
    [self inspectAndPublishStreamFormat:aggregate_id];

    if (_rtCtx) {
        _rtCtx->has_sample_time = 0;
        _rtCtx->last_end_sample_time = 0.0;
        atomic_store_explicit(&_rtCtx->active, 1, memory_order_release);
    }

    AudioDeviceIOProcID io_proc_id = NULL;
    status = AudioDeviceCreateIOProcID(aggregate_id, io_proc_callback, _rtCtx, &io_proc_id);
    if (status != noErr) {
        AudioHardwareDestroyAggregateDevice(aggregate_id);
        if (outStatus) *outStatus = (int32_t)status;
        return NO;
    }

    status = AudioDeviceStart(aggregate_id, io_proc_id);
    if (status != noErr) {
        AudioDeviceDestroyIOProcID(aggregate_id, io_proc_id);
        AudioHardwareDestroyAggregateDevice(aggregate_id);
        if (outStatus) *outStatus = (int32_t)status;
        return NO;
    }

    self.aggregateId = aggregate_id;
    self.ioProcId = io_proc_id;
    self.currentOutputUid = outputUid;

    __weak MacOSTapSession *weakSelf = self;
    self.formatListener = ^(UInt32 inNumberAddresses, const AudioObjectPropertyAddress *inAddresses) {
        (void)inNumberAddresses; (void)inAddresses;
        MacOSTapSession *strongSelf = weakSelf;
        if (strongSelf && !atomic_load(&strongSelf->_stopping) && strongSelf.aggregateId != kAudioObjectUnknown) {
            [strongSelf inspectAndPublishStreamFormat:strongSelf.aggregateId];
        }
    };
    AudioObjectPropertyAddress sr_addr = {
        kAudioDevicePropertyNominalSampleRate,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };
    AudioObjectPropertyAddress fmt_addr = {
        kAudioDevicePropertyStreamFormat,
        kAudioObjectPropertyScopeInput,
        kAudioObjectPropertyElementMain
    };
    AudioObjectAddPropertyListenerBlock(aggregate_id, &sr_addr, self.queue, self.formatListener);
    AudioObjectAddPropertyListenerBlock(aggregate_id, &fmt_addr, self.queue, self.formatListener);

    if (outStatus) *outStatus = 0;
    return YES;
}

- (void)rebuildIfOutputChangedUnlocked {
    if (atomic_load(&_stopping)) return;

    AudioObjectID newDevId = get_default_output_device_id();
    NSString *newUid = get_device_uid(newDevId);
    if (!newUid || (self.currentOutputUid && [newUid isEqualToString:self.currentOutputUid])) {
        return;
    }

    [self teardownAggregateUnlocked];

    int32_t status = 0;
    if ([self startAggregateForOutputDevice:newDevId outStatus:&status]) {
        if (_rtCtx) {
            atomic_fetch_add(&_rtCtx->device_switches, 1);
        }
    }
}

- (void)registerDefaultOutputListener {
    AudioObjectPropertyAddress default_out_addr = {
        kAudioHardwarePropertyDefaultOutputDevice,
        kAudioObjectPropertyScopeGlobal,
        kAudioObjectPropertyElementMain
    };

    __weak MacOSTapSession *weakSelf = self;
    self.defaultOutListener = ^(UInt32 inNumberAddresses, const AudioObjectPropertyAddress *inAddresses) {
        (void)inNumberAddresses; (void)inAddresses;
        MacOSTapSession *strongSelf = weakSelf;
        if (strongSelf) {
            [strongSelf rebuildIfOutputChangedUnlocked];
        }
    };

    AudioObjectAddPropertyListenerBlock(
        kAudioObjectSystemObject,
        &default_out_addr,
        self.queue,
        self.defaultOutListener
    );
}

- (uint64_t)deviceSwitchCount {
    return _rtCtx ? atomic_load(&_rtCtx->device_switches) : 0;
}

- (uint64_t)clipEventCount {
    return _rtCtx ? atomic_load(&_rtCtx->clip_events) : 0;
}

- (uint64_t)clockGlitchCount {
    return _rtCtx ? atomic_load(&_rtCtx->clock_glitches) : 0;
}

- (double)currentSampleRate {
    uint32_t sr = _rtCtx ? atomic_load(&_rtCtx->sample_rate_hz) : 48000;
    return sr > 0 ? (double)sr : 48000.0;
}

- (void)formatInfoBits:(uint32_t *)outBits isFloat:(uint32_t *)outIsFloat {
    uint32_t bits = _rtCtx ? atomic_load(&_rtCtx->bits_per_channel) : 32;
    uint32_t flags = _rtCtx ? atomic_load(&_rtCtx->format_flags) : (uint32_t)kAudioFormatFlagIsFloat;
    if (outBits) *outBits = bits;
    if (outIsFloat) *outIsFloat = ((flags & kAudioFormatFlagIsFloat) != 0) ? 1 : 0;
}

- (void)stopAndDestroy {
    atomic_store(&_stopping, true);
    if (_rtCtx) {
        atomic_store_explicit(&_rtCtx->active, 0, memory_order_release);
    }

    // Remove listener e destrói tudo sincronizadamente na fila serial (zero race ao alternar modos)
    dispatch_sync(self.queue, ^{
        if (self.defaultOutListener) {
            AudioObjectPropertyAddress default_out_addr = {
                kAudioHardwarePropertyDefaultOutputDevice,
                kAudioObjectPropertyScopeGlobal,
                kAudioObjectPropertyElementMain
            };
            AudioObjectRemovePropertyListenerBlock(
                kAudioObjectSystemObject,
                &default_out_addr,
                self.queue,
                self.defaultOutListener
            );
            self.defaultOutListener = nil;
        }

        [self teardownAggregateUnlocked];
        if (self.tapId != kAudioObjectUnknown) {
            AudioHardwareDestroyProcessTap(self.tapId);
            self.tapId = kAudioObjectUnknown;
        }
    });

    // Aguarda drenagem do coreaudiod antes de liberar o contexto e permitir nova captura imediata
    usleep(35000);

    if (_rtCtx) {
        free(_rtCtx);
        _rtCtx = NULL;
    }
}

@end

MacOSTapHandle *macos_tap_start(
    const uint32_t *target_object_ids,
    uint32_t target_count,
    StereoSamplesCallback callback,
    void *user_data,
    double *out_sample_rate,
    int32_t *out_status
) {
    @autoreleasepool {
        int32_t last_status = -1;

        // Até 3 tentativas caso o coreaudiod ainda esteja finalizando o teardown da sessão anterior
        for (int attempt = 0; attempt < 3; attempt++) {
            if (attempt > 0) {
                usleep(120000);
            }

            uint64_t gen = atomic_fetch_add(&g_tap_generation, 1) + 1;
            CATapDescription *tap_desc = nil;

            if (target_object_ids != NULL && target_count > 0) {
                NSMutableArray<NSNumber *> *targets = [NSMutableArray arrayWithCapacity:target_count];
                for (uint32_t i = 0; i < target_count; i++) {
                    [targets addObject:@((AudioObjectID)target_object_ids[i])];
                }
                tap_desc = [[CATapDescription alloc] initStereoMixdownOfProcesses:targets];
            } else {
                AudioObjectID self_proc = get_self_process_object_id();
                NSArray<NSNumber *> *exclude = (self_proc != kAudioObjectUnknown) ? @[@(self_proc)] : @[];
                tap_desc = [[CATapDescription alloc] initStereoGlobalTapButExcludeProcesses:exclude];
            }

            NSUUID *tap_uuid = [NSUUID UUID];
            tap_desc.UUID = tap_uuid;
            [tap_desc setName:[NSString stringWithFormat:@"LoopbackTabTap_%llu", gen]];
            [tap_desc setMuteBehavior:CATapUnmuted];
            [tap_desc setPrivate:YES];

            AudioObjectID tap_id = kAudioObjectUnknown;
            OSStatus status = AudioHardwareCreateProcessTap(tap_desc, &tap_id);
            if (status != noErr) {
                last_status = (int32_t)status;
                continue;
            }

            AudioObjectID output_dev = get_default_output_device_id();
            if (output_dev == kAudioObjectUnknown) {
                AudioHardwareDestroyProcessTap(tap_id);
                last_status = -1;
                continue;
            }

            RealtimeTapCtx *rt_ctx = (RealtimeTapCtx *)calloc(1, sizeof(RealtimeTapCtx));
            if (!rt_ctx) {
                AudioHardwareDestroyProcessTap(tap_id);
                last_status = -2;
                break;
            }
            rt_ctx->callback = callback;
            rt_ctx->user_data = user_data;
            atomic_init(&rt_ctx->device_switches, 0);
            atomic_init(&rt_ctx->clip_events, 0);
            atomic_init(&rt_ctx->clock_glitches, 0);
            atomic_init(&rt_ctx->sample_rate_hz, 48000);
            atomic_init(&rt_ctx->bits_per_channel, 32);
            atomic_init(&rt_ctx->format_flags, (uint32_t)kAudioFormatFlagIsFloat);
            atomic_init(&rt_ctx->active, 0);

            MacOSTapSession *session = [[MacOSTapSession alloc] initWithRealtimeCtx:rt_ctx generation:gen];
            session.tapId = tap_id;
            session.tapUuid = tap_uuid.UUIDString;

            int32_t agg_status = 0;
            if (![session startAggregateForOutputDevice:output_dev outStatus:&agg_status]) {
                AudioHardwareDestroyProcessTap(tap_id);
                free(rt_ctx);
                last_status = agg_status;
                continue;
            }

            [session registerDefaultOutputListener];

            if (out_sample_rate) *out_sample_rate = [session currentSampleRate];
            if (out_status) *out_status = 0;
            return (MacOSTapHandle *)CFBridgingRetain(session);
        }

        if (out_status) *out_status = last_status;
        return NULL;
    }
}

uint64_t macos_tap_device_switches(MacOSTapHandle *handle) {
    if (!handle) return 0;
    MacOSTapSession *session = (__bridge MacOSTapSession *)handle;
    return [session deviceSwitchCount];
}

uint64_t macos_tap_clip_events(MacOSTapHandle *handle) {
    if (!handle) return 0;
    MacOSTapSession *session = (__bridge MacOSTapSession *)handle;
    return [session clipEventCount];
}

uint64_t macos_tap_clock_glitches(MacOSTapHandle *handle) {
    if (!handle) return 0;
    MacOSTapSession *session = (__bridge MacOSTapSession *)handle;
    return [session clockGlitchCount];
}

double macos_tap_current_sample_rate(MacOSTapHandle *handle) {
    if (!handle) return 48000.0;
    MacOSTapSession *session = (__bridge MacOSTapSession *)handle;
    return [session currentSampleRate];
}

void macos_tap_format_info(MacOSTapHandle *handle, uint32_t *out_bits, uint32_t *out_is_float) {
    if (!handle) {
        if (out_bits) *out_bits = 32;
        if (out_is_float) *out_is_float = 1;
        return;
    }
    MacOSTapSession *session = (__bridge MacOSTapSession *)handle;
    [session formatInfoBits:out_bits isFloat:out_is_float];
}

void macos_tap_stop(MacOSTapHandle *handle) {
    if (!handle) return;
    @autoreleasepool {
        MacOSTapSession *session = (__bridge_transfer MacOSTapSession *)handle;
        [session stopAndDestroy];
    }
}
