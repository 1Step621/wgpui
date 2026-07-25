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
    atomic::{AtomicBool, AtomicU8, Ordering},
};

use crate::flamegraph::{
    self, DeepCaptureBufferKind, DeepCaptureDrawCall, DrawCallKind, GpuClockCalibration, GpuPassKind, GpuSpan,
    SpanName,
};

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
// Resource readback covers `WgpuContext`'s fixed buffers only (quads,
// shadows, underlines, mono/poly sprites, backdrop filters, paths vertices)
// -- not the atlas or surface textures the issue also mentions. See
// `flamegraph.rs`'s Phase 4 section doc comment for why buffers are read
// back once per touched kind rather than once per draw call, and this
// module's final Phase 4 doc note (bottom of file) for why texture readback
// specifically was scoped out of this round.

/// Accumulates one triggered frame's command stream while `WgpuRenderer::draw`
/// is recording it. Created fresh by `WgpuRenderer::draw` when
/// `flamegraph::take_deep_capture_request()` returns true, consumed by
/// `finish` once the frame's `PrimitiveBatch` loop completes.
pub(crate) struct DeepCaptureRecorder {
    draw_calls: Vec<DeepCaptureDrawCall>,
    /// Distinct `DeepCaptureBufferKind`s seen across `draw_calls` so far, in
    /// first-seen order. Small (at most 7 possible kinds) so a linear-scan
    /// `contains` check on every `record_draw_call` is cheaper than a
    /// `HashSet` and avoids pulling in a hasher for something this size.
    touched_buffers: Vec<DeepCaptureBufferKind>,
}

impl DeepCaptureRecorder {
    pub(crate) fn new() -> Self {
        Self {
            draw_calls: Vec::new(),
            touched_buffers: Vec::new(),
        }
    }

    /// Record one `RenderPass::draw` call's full identifying detail. Called
    /// from every `PrimitiveBatch` match arm in `WgpuRenderer::draw` that
    /// phase 2's `record_draw_call` (a different, counter-only function of
    /// the same name in `flamegraph.rs`) already instruments -- same call
    /// sites, recording detail instead of just a running tally.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_draw_call(
        &mut self,
        kind: DrawCallKind,
        pipeline_label: &'static str,
        pass_label: &'static str,
        vertex_range: std::ops::Range<u32>,
        instance_range: std::ops::Range<u32>,
        bind_group_count: u32,
        buffer_kind: Option<DeepCaptureBufferKind>,
        atlas_texture_id: Option<u64>,
    ) {
        if let Some(buffer_kind) = buffer_kind
            && !self.touched_buffers.contains(&buffer_kind)
        {
            self.touched_buffers.push(buffer_kind);
        }

        let sequence = self.draw_calls.len() as u32;
        self.draw_calls.push(DeepCaptureDrawCall {
            sequence,
            kind,
            pipeline_label,
            pass_label,
            vertex_range: (vertex_range.start, vertex_range.end),
            instance_range: (instance_range.start, instance_range.end),
            bind_group_count,
            buffer_kind,
            atlas_texture_id,
        });
    }

    /// Finish recording and hand off to the readback phase: for every
    /// distinct buffer kind touched by this frame's command stream, look it
    /// up in `buffers`, allocate a same-sized `MAP_READ` staging buffer, and
    /// record a `copy_buffer_to_buffer` from the live buffer into it. Must be
    /// called after the frame's render-pass loop has ended (so `draw_calls`
    /// is complete, and the live buffers are no longer borrowed by an active
    /// `RenderPass`) and before `encoder.finish()` (the copy commands have to
    /// land in the same encoder submission as the draws they're snapshotting
    /// after).
    ///
    /// `buffers` is expected to list every `DeepCaptureBufferKind` `WgpuContext`
    /// actually has a fixed buffer for (see that type's variants); a touched
    /// kind missing from `buffers` is silently skipped rather than panicking,
    /// since `finish` has no way to distinguish "programmer error" from "this
    /// buffer kind doesn't apply to the current renderer configuration" --
    /// either way, dropping one buffer's contents from an otherwise-useful
    /// capture is better than losing the whole thing.
    pub(crate) fn finish(
        self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        buffers: &[(DeepCaptureBufferKind, &wgpu::Buffer)],
    ) -> DeepCapturePendingReadback {
        let mut staging = Vec::with_capacity(self.touched_buffers.len());
        for kind in &self.touched_buffers {
            let Some((_, source)) = buffers.iter().find(|(candidate, _)| candidate == kind) else {
                continue;
            };
            let size = source.size();
            if size == 0 {
                continue;
            }
            let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("flamegraph_deep_capture_staging_buffer"),
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            encoder.copy_buffer_to_buffer(source, 0, &staging_buffer, 0, size);
            staging.push(DeepCaptureStagingBuffer {
                kind: *kind,
                size,
                buffer: staging_buffer,
                map_state: Arc::new(AtomicU8::new(MAP_STATE_PENDING)),
            });
        }

        DeepCapturePendingReadback {
            draw_calls: self.draw_calls,
            staging,
            state: PendingReadbackState::AwaitingSubmit,
        }
    }
}

/// One touched buffer's readback: the live buffer's kind (for tagging the
/// resulting [`flamegraph::DeepCaptureBufferContents`]), its size (both
/// buffers are this size -- `copy_buffer_to_buffer` requires matching
/// lengths), the staging buffer itself, and the async map's completion state.
struct DeepCaptureStagingBuffer {
    kind: DeepCaptureBufferKind,
    size: u64,
    buffer: wgpu::Buffer,
    map_state: Arc<AtomicU8>,
}

const MAP_STATE_PENDING: u8 = 0;
const MAP_STATE_OK: u8 = 1;
const MAP_STATE_ERR: u8 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PendingReadbackState {
    /// `finish` has recorded the copy commands but the encoder holding them
    /// has not been submitted yet -- `map_async` must not be called until it
    /// has (mapping a buffer that a not-yet-submitted copy still targets is a
    /// wgpu validation error).
    AwaitingSubmit,
    /// `begin_readback` has been called; waiting on `staging`'s maps.
    MapPending,
}

/// A deep capture's command stream, mid-readback. Held in
/// `WgpuRenderer::deep_capture` from the frame it was recorded until
/// `poll` reports the readback complete, at which point `WgpuRenderer::draw`
/// drops it -- freeing any staging buffers it holds -- and publishes the
/// result via `flamegraph::complete_deep_capture`.
pub(crate) struct DeepCapturePendingReadback {
    draw_calls: Vec<DeepCaptureDrawCall>,
    staging: Vec<DeepCaptureStagingBuffer>,
    state: PendingReadbackState,
}

impl DeepCapturePendingReadback {
    /// Start the async readback maps for every staging buffer `finish`
    /// created. Must be called once, after the encoder `finish` recorded copy
    /// commands into has actually been submitted to the queue -- mirrors
    /// `GpuQueryManager::begin_readback`'s same ordering requirement, for the
    /// same reason (see [`PendingReadbackState::AwaitingSubmit`]).
    pub(crate) fn begin_readback(&mut self) {
        if self.state != PendingReadbackState::AwaitingSubmit {
            return;
        }
        for staging in &self.staging {
            let map_state = staging.map_state.clone();
            staging.buffer.slice(0..staging.size).map_async(wgpu::MapMode::Read, move |result| {
                map_state.store(
                    if result.is_ok() { MAP_STATE_OK } else { MAP_STATE_ERR },
                    Ordering::Release,
                );
            });
        }
        self.state = PendingReadbackState::MapPending;
    }

    /// Non-blocking poll (mirrors `GpuQueryManager::poll_readback`'s
    /// same-thread, non-blocking pattern documented at the top of this
    /// file). Returns the finished [`flamegraph::DeepCapture`] once every
    /// touched buffer's map has resolved -- successfully or not, so a single
    /// failed map can't leave this capture stuck pending forever; the caller
    /// is expected to drop `self` immediately afterward, satisfying "no
    /// persistent overhead, no persistent buffers" once a capture is done.
    pub(crate) fn poll(&mut self, device: &wgpu::Device) -> Option<flamegraph::DeepCapture> {
        if self.state != PendingReadbackState::MapPending {
            return None;
        }
        let _ = device.poll(wgpu::PollType::Poll);
        let all_resolved = self
            .staging
            .iter()
            .all(|staging| staging.map_state.load(Ordering::Acquire) != MAP_STATE_PENDING);
        if !all_resolved {
            return None;
        }

        let mut resources_finalized = true;
        let mut buffer_contents = Vec::with_capacity(self.staging.len());
        for staging in &self.staging {
            if staging.map_state.load(Ordering::Acquire) != MAP_STATE_OK {
                resources_finalized = false;
                continue;
            }
            let slice = staging.buffer.slice(0..staging.size);
            match slice.get_mapped_range() {
                Ok(view) => {
                    buffer_contents.push(flamegraph::DeepCaptureBufferContents {
                        kind: staging.kind,
                        bytes: view.to_vec(),
                    });
                }
                Err(_) => resources_finalized = false,
            }
            staging.buffer.unmap();
        }

        Some(flamegraph::DeepCapture {
            draw_calls: std::mem::take(&mut self.draw_calls),
            buffer_contents,
            resources_finalized,
        })
    }
}

// Deferred this round: texture content readback (atlas mono/poly textures,
// surface textures). The issue's "resource contents" bullet covers both
// buffers and textures; this implementation covers buffers only.
//
// Reasoning for the cut: every fixed buffer above is one `wgpu::Buffer` of a
// known, stable usage flag set, so one `copy_buffer_to_buffer` + one staging
// buffer per kind (7 total, worst case) is all `DeepCaptureRecorder::finish`
// needs. Textures are a meaningfully different shape of problem --
// `copy_texture_to_buffer` requires `bytes_per_row` padded to
// `COPY_BYTES_PER_ROW_ALIGNMENT` (256), the atlas has a dynamic, growable
// number of monochrome/polychrome texture pages (`WgpuAtlasStorage`, not a
// single fixed handle), and surface textures vary in count/size/format per
// registered `SurfaceId` (`SurfaceRegistry`). Getting that right (padding
// math, iterating a dynamic page/surface list, one staging buffer and
// `map_async` per texture rather than per fixed field) is a distinctly
// larger and riskier chunk of work than the buffer case, on top of an
// already GPU-readback-heavy phase. Per this phase's own risk guidance, a
// smaller correct slice now (full command stream + pipeline/shader identity
// + fixed-buffer contents) was chosen over forcing texture readback into the
// same session. `DeepCaptureDrawCall::atlas_texture_id` still identifies
// *which* atlas texture a sprite call bound, even though this round can't
// read that texture's pixels back -- a future pass can add
// `DeepCaptureTextureContents` alongside `DeepCaptureBufferContents` without
// needing to change anything landed here.

#[cfg(test)]
mod tests {
    use super::*;

    // `DeepCaptureRecorder::record_draw_call`/`touched_buffers` are pure
    // bookkeeping with no wgpu resource involved, so this is covered without
    // needing a live `wgpu::Device` -- exercising exactly the logic
    // `WgpuRenderer::draw`'s `PrimitiveBatch` match arms call into, and the
    // same touched-buffer tracking `finish` (tested below, where a device is
    // available) relies on to know which buffers to stage.
    #[test]
    fn record_draw_call_builds_command_stream_with_deduped_touched_buffers() {
        let mut recorder = DeepCaptureRecorder::new();

        recorder.record_draw_call(
            DrawCallKind::Quads,
            "quads",
            "main",
            0..4,
            0..10,
            2,
            Some(DeepCaptureBufferKind::Quads),
            None,
        );
        recorder.record_draw_call(
            DrawCallKind::MonoSprites,
            "mono_sprites",
            "main",
            0..4,
            10..15,
            4,
            Some(DeepCaptureBufferKind::MonoSprites),
            Some(42),
        );
        // A second `Quads` call: exercises that `touched_buffers` dedups
        // rather than recording `DeepCaptureBufferKind::Quads` twice.
        recorder.record_draw_call(
            DrawCallKind::Quads,
            "quads",
            "main_resumed",
            0..4,
            15..20,
            2,
            Some(DeepCaptureBufferKind::Quads),
            None,
        );

        assert_eq!(recorder.draw_calls.len(), 3, "all three draw calls should be recorded");
        assert_eq!(
            recorder.draw_calls.iter().map(|call| call.sequence).collect::<Vec<_>>(),
            vec![0, 1, 2],
            "sequence should be 0-based submission order"
        );
        assert_eq!(recorder.draw_calls[0].kind, DrawCallKind::Quads);
        assert_eq!(recorder.draw_calls[0].pipeline_label, "quads");
        assert_eq!(recorder.draw_calls[0].pass_label, "main");
        assert_eq!(recorder.draw_calls[0].vertex_range, (0, 4));
        assert_eq!(recorder.draw_calls[0].instance_range, (0, 10));
        assert_eq!(recorder.draw_calls[0].bind_group_count, 2);
        assert_eq!(recorder.draw_calls[0].buffer_kind, Some(DeepCaptureBufferKind::Quads));
        assert_eq!(recorder.draw_calls[0].atlas_texture_id, None);

        assert_eq!(recorder.draw_calls[1].kind, DrawCallKind::MonoSprites);
        assert_eq!(recorder.draw_calls[1].atlas_texture_id, Some(42));

        assert_eq!(
            recorder.draw_calls[2].pass_label, "main_resumed",
            "the third call's pass_label should reflect the pass at the time it was recorded, \
             independent of the first call's"
        );

        assert_eq!(
            recorder.touched_buffers,
            vec![DeepCaptureBufferKind::Quads, DeepCaptureBufferKind::MonoSprites],
            "Quads should appear once despite being touched by two draw calls, in first-seen order"
        );
    }

    #[test]
    fn record_draw_call_with_no_buffer_kind_does_not_touch_anything() {
        let mut recorder = DeepCaptureRecorder::new();
        recorder.record_draw_call(DrawCallKind::Surfaces, "surfaces", "main", 0..4, 0..1, 2, None, None);
        assert_eq!(recorder.draw_calls.len(), 1);
        assert!(
            recorder.touched_buffers.is_empty(),
            "Surfaces draws a per-call uniform buffer, not a fixed WgpuContext buffer"
        );
    }

    /// Creates a headless (surface-less) `wgpu::Device`/`Queue` for testing
    /// buffer readback, the same way `WgpuContext::new` does minus the
    /// window-bound `Surface` (which needs a real window handle this test
    /// doesn't have). Returns `None` rather than panicking when no adapter is
    /// available -- this repo has no headless-GPU CI fixture (see
    /// `render_context.rs`'s test module doc comment for the same caveat on
    /// phase 3's buffer/texture-size accessors), so a sandbox/CI environment
    /// with no GPU or software rasterizer skips the readback assertions below
    /// instead of failing the whole suite.
    fn create_headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        });
        // Mirrors `WgpuContext::new`'s `enumerate_adapters` + pick-first
        // approach (`render_context.rs`) rather than `request_adapter`, for
        // the same reason: no `compatible_surface` is available here (there
        // is no window), and this crate's established pattern already covers
        // that case.
        let adapter = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
            .into_iter()
            .next()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    /// End-to-end readback test: records a command stream referencing a
    /// single fake "quads" buffer, runs it through `finish` (stages + copies)
    /// and `begin_readback`/`poll` (maps + harvests), and asserts the
    /// harvested `DeepCapture`'s buffer contents match what was written --
    /// the concrete "resource contents at time of use" claim from the issue,
    /// covering the fixed-buffer slice of it this phase implements.
    #[test]
    fn finish_and_poll_read_back_real_buffer_contents() {
        let Some((device, queue)) = create_headless_device() else {
            eprintln!(
                "skipping finish_and_poll_read_back_real_buffer_contents: no wgpu adapter available in this environment"
            );
            return;
        };

        let source_bytes: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let source_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test_quads_buffer"),
            size: source_bytes.len() as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&source_buffer, 0, &source_bytes);

        let mut recorder = DeepCaptureRecorder::new();
        recorder.record_draw_call(
            DrawCallKind::Quads,
            "quads",
            "main",
            0..4,
            0..1,
            2,
            Some(DeepCaptureBufferKind::Quads),
            None,
        );

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut pending = recorder.finish(&device, &mut encoder, &[(DeepCaptureBufferKind::Quads, &source_buffer)]);
        queue.submit(Some(encoder.finish()));
        pending.begin_readback();

        let capture = loop {
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            if let Some(capture) = pending.poll(&device) {
                break capture;
            }
        };

        assert_eq!(capture.draw_calls.len(), 1);
        assert!(capture.resources_finalized, "the one touched buffer's readback should have succeeded");
        let contents = capture
            .buffer_contents(DeepCaptureBufferKind::Quads)
            .expect("Quads buffer should have been read back");
        assert_eq!(
            contents.bytes, source_bytes,
            "readback bytes should match exactly what was written to the source buffer"
        );
    }
}
