//! GPU-side timestamp query capture (Phase 1 of the profiling epic, see
//! issue #57). Kept separate from `flamegraph.rs` so the CPU-only capture
//! types there never need to depend on `wgpu`, and so the two genuinely
//! distinct concerns (thread-local span stacks vs. GPU resource lifecycle)
//! don't mix into one large file. `pub(crate)` only: no public API surface
//! this round.
//!
//! # Threading
//!
//! Every method here is called from the render thread, which is the same
//! thread that owns `wgpu::Device`/`wgpu::Queue` and already actively uses
//! them (see the `NOTE: We do NOT call device.poll()` comment in
//! `surface_registry.rs`'s `swap_ready_display`, about *cross*-thread polling
//! causing driver contention). `poll_readback`'s non-blocking
//! `device.poll(PollType::Poll)` call runs on that same render thread as an
//! ordinary step of the per-frame draw, not from a separate background task —
//! deliberately deviating from this repo's issue #57, which suggested a
//! dedicated `BackgroundExecutor` task. A background task would poll the
//! device from a *different* thread than the one actively recording/
//! submitting to it, which is exactly the cross-thread pattern
//! `surface_registry.rs` warns is unsafe for this device. Polling
//! synchronously from the render thread avoids that risk entirely and still
//! never blocks (`PollType::Poll`, not `Wait`), except in `calibrate`, which
//! is explicitly allowed to block (rare, user-triggered, capture-start only).

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use crate::flamegraph::{self, GpuClockCalibration, GpuPassKind, GpuSpan, SpanName};

/// Timestamp-pair budget per frame: main/main_resumed/up to 4 nested filter
/// groups + resumes/fast_surface_blit, with headroom, plus the whole-encoder
/// GpuSubmitPresent bracket. Exceeding this sets `FrameCapture::
/// gpu_spans_truncated` on that frame rather than growing the query set
/// mid-frame or panicking.
pub(crate) const MAX_GPU_SPANS_PER_FRAME: usize = 16;

const QUERIES_PER_FRAME: u32 = MAX_GPU_SPANS_PER_FRAME as u32 * 2;
const GENERATION_COUNT: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GenerationState {
    Idle,
    Recording,
    Resolving,
    MapPending,
}

struct PendingSpanLabel {
    name: SpanName,
    pass_kind: GpuPassKind,
    begin_query: u32,
    end_query: u32,
}

struct QueryGeneration {
    query_set: wgpu::QuerySet,
    resolve_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    pending_labels: Vec<PendingSpanLabel>,
    frame_index: u64,
    state: GenerationState,
    next_query_index: u32,
    truncated: bool,
    /// Bumped every `reset_for_frame`, so a future viewer can diagnose
    /// readback ordering (`GpuSpan::query_set_generation`).
    epoch: u64,
    map_ready: Arc<AtomicBool>,
}

impl QueryGeneration {
    fn new(device: &wgpu::Device) -> Self {
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("flamegraph_gpu_query_set"),
            ty: wgpu::QueryType::Timestamp,
            count: QUERIES_PER_FRAME,
        });
        let buffer_size = (QUERIES_PER_FRAME as u64) * 8;
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flamegraph_gpu_resolve_buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flamegraph_gpu_staging_buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            query_set,
            resolve_buffer,
            staging_buffer,
            pending_labels: Vec::with_capacity(MAX_GPU_SPANS_PER_FRAME),
            frame_index: 0,
            state: GenerationState::Idle,
            next_query_index: 0,
            truncated: false,
            epoch: 0,
            map_ready: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Reset this generation for a new frame. If the generation was still
    /// awaiting readback from an earlier frame (the triple-buffer depth was
    /// exceeded, i.e. GPU readback fell more than two frames behind), that
    /// earlier frame's GPU spans are dropped rather than blocking here —
    /// `FrameCapture::gpu_spans_finalized` simply stays false for it forever,
    /// which a viewer can treat the same as "never arrived".
    fn reset_for_frame(&mut self, frame_index: u64) {
        self.pending_labels.clear();
        self.frame_index = frame_index;
        self.state = GenerationState::Recording;
        self.next_query_index = 0;
        self.truncated = false;
        self.epoch += 1;
        self.map_ready.store(false, Ordering::Release);
    }

    /// Reserve a begin/end timestamp-write pair. Returns `None` (without
    /// mutating any wgpu resources) when this generation isn't actively
    /// recording a frame, or when `MAX_GPU_SPANS_PER_FRAME` is exceeded (in
    /// which case `truncated` is set).
    fn reserve_pair(&mut self, name: SpanName, pass_kind: GpuPassKind) -> Option<(wgpu::QuerySet, u32, u32)> {
        if self.state != GenerationState::Recording {
            return None;
        }
        if self.pending_labels.len() >= MAX_GPU_SPANS_PER_FRAME {
            self.truncated = true;
            return None;
        }
        let begin = self.next_query_index;
        let end = begin + 1;
        self.next_query_index += 2;
        self.pending_labels.push(PendingSpanLabel {
            name,
            pass_kind,
            begin_query: begin,
            end_query: end,
        });
        Some((self.query_set.clone(), begin, end))
    }

    /// Record the resolve + resolve-to-staging copy into `encoder`. Must be
    /// called before `encoder.finish()`. No-ops (and returns to `Idle`
    /// immediately, skipping readback) if nothing was recorded this frame.
    fn resolve(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.state != GenerationState::Recording {
            return;
        }
        if self.next_query_index == 0 {
            self.state = GenerationState::Idle;
            return;
        }
        encoder.resolve_query_set(&self.query_set, 0..self.next_query_index, &self.resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve_buffer,
            0,
            &self.staging_buffer,
            0,
            (self.next_query_index as u64) * 8,
        );
        self.state = GenerationState::Resolving;
    }

    /// Start the async readback map. Must be called after the encoder
    /// recorded by `resolve` has been submitted.
    fn begin_readback(&mut self) {
        if self.state != GenerationState::Resolving {
            return;
        }
        self.state = GenerationState::MapPending;
        let size = (self.next_query_index as u64) * 8;
        let map_ready = self.map_ready.clone();
        self.staging_buffer
            .slice(0..size)
            .map_async(wgpu::MapMode::Read, move |result| {
                map_ready.store(result.is_ok(), Ordering::Release);
            });
    }

    /// If the async map has completed, read the resolved timestamps, convert
    /// them to `GpuSpan`s using the session's calibration, attach them to the
    /// owning `FrameCapture`, and return this generation to `Idle`.
    fn try_harvest(&mut self) {
        if self.state != GenerationState::MapPending || !self.map_ready.load(Ordering::Acquire) {
            return;
        }

        let size = (self.next_query_index as u64) * 8;
        let calibration = flamegraph::gpu_calibration();
        let spans = {
            let slice = self.staging_buffer.slice(0..size);
            match slice.get_mapped_range() {
                Ok(view) => self
                    .pending_labels
                    .iter()
                    .filter_map(|label| {
                        let begin_ticks = read_u64(&view, label.begin_query)?;
                        let end_ticks = read_u64(&view, label.end_query)?;
                        Some(ticks_to_span(label, begin_ticks, end_ticks, &calibration, self.epoch))
                    })
                    .collect::<Vec<_>>(),
                Err(_) => Vec::new(),
            }
        };
        self.staging_buffer.unmap();

        flamegraph::attach_gpu_spans(self.frame_index, spans, self.truncated);
        self.state = GenerationState::Idle;
    }
}

fn read_u64(view: &[u8], index: u32) -> Option<u64> {
    let offset = (index as usize).checked_mul(8)?;
    let bytes = view.get(offset..offset + 8)?;
    Some(bytemuck::pod_read_unaligned(bytes))
}

fn ticks_to_span(
    label: &PendingSpanLabel,
    begin_ticks: u64,
    end_ticks: u64,
    calibration: &GpuClockCalibration,
    epoch: u64,
) -> GpuSpan {
    if !calibration.calibrated {
        return GpuSpan {
            name: label.name,
            start_ns: 0,
            duration_ns: 0,
            pass_kind: label.pass_kind,
            query_set_generation: epoch,
        };
    }

    let ns_per_tick = calibration.ns_per_tick as f64;
    let duration_ns = (end_ticks.saturating_sub(begin_ticks) as f64 * ns_per_tick)
        .round()
        .clamp(0.0, u32::MAX as f64) as u32;
    let ticks_from_anchor = begin_ticks.saturating_sub(calibration.gpu_anchor_ticks);
    let start_ns = calibration
        .cpu_anchor_ns
        .saturating_add((ticks_from_anchor as f64 * ns_per_tick).round() as u64);

    GpuSpan {
        name: label.name,
        start_ns,
        duration_ns,
        pass_kind: label.pass_kind,
        query_set_generation: epoch,
    }
}

/// A reserved timestamp-write pair, ready to be turned into
/// `wgpu::RenderPassTimestampWrites` for a `begin_render_pass` call. Holds an
/// owned (cheap, ref-counted) `QuerySet` clone rather than a borrow, so it
/// isn't tied to the `GpuQueryManager`'s mutex guard's lifetime.
pub(crate) struct ReservedTimestamps {
    query_set: wgpu::QuerySet,
    begin: u32,
    end: u32,
}

impl ReservedTimestamps {
    pub(crate) fn writes(&self) -> wgpu::RenderPassTimestampWrites<'_> {
        wgpu::RenderPassTimestampWrites {
            query_set: &self.query_set,
            beginning_of_pass_write_index: Some(self.begin),
            end_of_pass_write_index: Some(self.end),
        }
    }

    /// For encoder-level (not render-pass-level) brackets, e.g. the
    /// whole-frame `GpuSubmitPresent` span, which uses
    /// `CommandEncoder::write_timestamp` directly rather than
    /// `RenderPassTimestampWrites`.
    pub(crate) fn query_set(&self) -> &wgpu::QuerySet {
        &self.query_set
    }

    pub(crate) fn begin_index(&self) -> u32 {
        self.begin
    }

    pub(crate) fn end_index(&self) -> u32 {
        self.end
    }
}

/// Owns the triple-buffered `QuerySet`/resolve/staging generations and the
/// session's CPU/GPU clock calibration. Allocated lazily (see
/// `sync_with_active_capture`) only while a capture with `capture_gpu: true`
/// is active, so GPU query VRAM/setup cost is zero when not recording.
pub(crate) struct GpuQueryManager {
    generations: [QueryGeneration; GENERATION_COUNT],
    current: usize,
}

impl GpuQueryManager {
    fn new(device: &wgpu::Device) -> Self {
        Self {
            generations: std::array::from_fn(|_| QueryGeneration::new(device)),
            current: 0,
        }
    }

    /// One-time CPU/GPU clock calibration: submit a trivial bracketing pair
    /// of `write_timestamp` calls and blocking-poll *this one calibration
    /// submission only* — acceptable since it's rare, user-triggered, and not
    /// on the per-frame hot path, unlike `poll_readback`.
    fn calibrate(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let Some(cpu_anchor) = flamegraph::capture_anchor() else {
            return;
        };

        let calibration_query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("flamegraph_gpu_calibration_query_set"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flamegraph_gpu_calibration_resolve_buffer"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flamegraph_gpu_calibration_staging_buffer"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cpu_anchor_ns = std::time::Instant::now().duration_since(cpu_anchor).as_nanos().min(u64::MAX as u128) as u64;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flamegraph_gpu_calibration_encoder"),
        });
        encoder.write_timestamp(&calibration_query_set, 0);
        encoder.write_timestamp(&calibration_query_set, 1);
        encoder.resolve_query_set(&calibration_query_set, 0..2, &resolve_buffer, 0);
        encoder.copy_buffer_to_buffer(&resolve_buffer, 0, &staging_buffer, 0, 16);
        queue.submit(Some(encoder.finish()));

        let mapped = Arc::new(AtomicBool::new(false));
        {
            let mapped = mapped.clone();
            staging_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    mapped.store(result.is_ok(), Ordering::Release);
                });
        }

        // Blocking poll: acceptable here only because calibration runs once
        // per capture session at start_capture time, not per frame.
        if device.poll(wgpu::PollType::wait_indefinitely()).is_err() || !mapped.load(Ordering::Acquire) {
            return;
        }

        let begin_ticks = {
            let slice = staging_buffer.slice(..);
            let ticks = match slice.get_mapped_range() {
                Ok(view) => read_u64(&view, 0),
                Err(_) => None,
            };
            ticks
        };
        staging_buffer.unmap();

        let Some(gpu_anchor_ticks) = begin_ticks else {
            return;
        };

        flamegraph::set_gpu_calibration(GpuClockCalibration {
            cpu_anchor_ns,
            gpu_anchor_ticks,
            ns_per_tick: queue.get_timestamp_period(),
            calibrated: true,
        });
    }

    /// Select and reset the next generation for a new frame, tagging it with
    /// the CPU-side frame index it belongs to.
    pub(crate) fn begin_frame(&mut self, frame_index: u64) {
        self.current = (self.current + 1) % GENERATION_COUNT;
        self.generations[self.current].reset_for_frame(frame_index);
    }

    /// Reserve a timestamp-write pair against the current generation. See
    /// `QueryGeneration::reserve_pair` for when this returns `None`.
    pub(crate) fn reserve_pair(&mut self, name: SpanName, pass_kind: GpuPassKind) -> Option<ReservedTimestamps> {
        let (query_set, begin, end) = self.generations[self.current].reserve_pair(name, pass_kind)?;
        Some(ReservedTimestamps { query_set, begin, end })
    }

    /// Record the resolve/copy commands for the current generation into
    /// `encoder`. Call once per frame, before `encoder.finish()`.
    pub(crate) fn finish_frame(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.generations[self.current].resolve(encoder);
    }

    /// Start the async readback for the current generation. Call once per
    /// frame, immediately after the encoder from `finish_frame` was submitted.
    pub(crate) fn begin_readback(&mut self) {
        self.generations[self.current].begin_readback();
    }

    /// Non-blocking poll (`PollType::Poll`, never `Wait`) that drives forward
    /// any in-flight `map_async` callbacks, then harvests any generation
    /// whose readback has completed. Safe to call every frame from the render
    /// thread (see the module-level threading note).
    pub(crate) fn poll_readback(&mut self, device: &wgpu::Device) {
        let _ = device.poll(wgpu::PollType::Poll);
        for generation in &mut self.generations {
            generation.try_harvest();
        }
    }
}

/// Lazily create or tear down the `GpuQueryManager` to match whether a
/// GPU-capturing session is currently active, so VRAM/setup cost is zero when
/// idle. Call once per frame from the render thread before reserving any
/// timestamps.
pub(crate) fn sync_with_active_capture(
    slot: &mut Option<GpuQueryManager>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) {
    let wants_gpu = flamegraph::capture_enabled() && flamegraph::active_capture_wants_gpu();
    match (wants_gpu, slot.is_some()) {
        (true, false) => {
            let manager = GpuQueryManager::new(device);
            manager.calibrate(device, queue);
            *slot = Some(manager);
        }
        (false, true) => {
            *slot = None;
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Phase 4: on-demand GPU deep capture (issue #60).
//
// `DeepCaptureRecorder`/`DeepCapturePendingReadback` deliberately do not
// reuse `GpuQueryManager`'s triple-buffered generation array: a deep capture
// is single-shot and torn down the moment its readback completes (see
// `flamegraph.rs`'s Phase 4 section doc comment), so there is nothing to
// triple-buffer -- at most one `DeepCapturePendingReadback` exists at a time,
// held in `WgpuRenderer::deep_capture`. The *shape* of the readback state
// machine (record during the frame -> copy into a staging buffer before
// `encoder.finish()` -> `map_async` after submit -> poll non-blockingly on
// the render thread every frame until ready -> harvest) is intentionally the
// same pattern `QueryGeneration` above uses, for the same reason documented
// in this file's module docs: this device may only safely be polled from the
// render thread that submits to it.
//
// This stage (first commit of Phase 4) wires the arm/fire/teardown lifecycle
// end-to-end with an intentionally empty command stream and no resource
// readback -- `WgpuRenderer::draw` doesn't call `record_draw_call` yet, so
// `finish` always sees zero touched buffers. Real per-draw-call recording and
// real buffer readback land in the two commits that follow, each replacing
// one of the two stub bodies below without changing this type's public
// shape.

/// Accumulates one triggered frame's command stream while `WgpuRenderer::draw`
/// is recording it. Created fresh by `WgpuRenderer::draw` when
/// `flamegraph::take_deep_capture_request()` returns true, consumed by
/// `finish` once the frame's `PrimitiveBatch` loop completes.
pub(crate) struct DeepCaptureRecorder {
    draw_calls: Vec<flamegraph::DeepCaptureDrawCall>,
}

impl DeepCaptureRecorder {
    pub(crate) fn new() -> Self {
        Self {
            draw_calls: Vec::new(),
        }
    }

    /// Finish recording and hand off to the readback phase. Must be called
    /// after the frame's render-pass loop has ended (so `draw_calls` is
    /// complete) and before `encoder.finish()` (once resource readback lands,
    /// this will need to record buffer-copy commands into `encoder`).
    pub(crate) fn finish(self, device: &wgpu::Device) -> DeepCapturePendingReadback {
        // Stub: no wgpu resources to copy yet, since `draw_calls` never holds
        // a `buffer_kind` this round. `device` is accepted now so the later
        // commit that starts creating staging buffers doesn't need to change
        // this method's call site in `WgpuRenderer::draw`.
        let _ = device;
        DeepCapturePendingReadback {
            draw_calls: self.draw_calls,
            resources_finalized: true,
        }
    }
}

/// A deep capture's command stream, mid-readback. Held in
/// `WgpuRenderer::deep_capture` from the frame it was recorded until
/// `poll` reports the readback complete, at which point `WgpuRenderer::draw`
/// drops it -- freeing any staging buffers it holds -- and publishes the
/// result via `flamegraph::complete_deep_capture`.
pub(crate) struct DeepCapturePendingReadback {
    draw_calls: Vec<flamegraph::DeepCaptureDrawCall>,
    resources_finalized: bool,
}

impl DeepCapturePendingReadback {
    /// Start any async readback maps. Must be called once, after the encoder
    /// `finish` recorded copy commands into (once resource readback lands)
    /// has actually been submitted to the queue -- mirrors
    /// `GpuQueryManager::begin_readback`'s same ordering requirement.
    pub(crate) fn begin_readback(&mut self) {
        // Stub: nothing to map yet.
    }

    /// Non-blocking poll (mirrors `GpuQueryManager::poll_readback`'s
    /// same-thread, non-blocking pattern documented at the top of this
    /// file). Returns the finished [`flamegraph::DeepCapture`] once every
    /// touched buffer's map has resolved (successfully or not); the caller
    /// is expected to drop `self` immediately afterward, satisfying "no
    /// persistent overhead, no persistent buffers" once a capture is done.
    pub(crate) fn poll(&mut self, device: &wgpu::Device) -> Option<flamegraph::DeepCapture> {
        // Stub: nothing pending, so this is always immediately "ready" -- the
        // very next `WgpuRenderer::draw` call after `begin_readback` reaps
        // it. Once real buffer readback lands, this gates on `map_async`
        // completion the same way `QueryGeneration::try_harvest` does.
        let _ = device;
        Some(flamegraph::DeepCapture {
            draw_calls: std::mem::take(&mut self.draw_calls),
            buffer_contents: Vec::new(),
            resources_finalized: self.resources_finalized,
        })
    }
}
