//! CPU-side flamegraph capture engine (Phase 1 of the profiling epic, see issue #57).
//!
//! This module owns the data model, per-thread span recorder, capture session
//! lifecycle, and binary trace export. It absorbs and replaces the old
//! `profiler.rs`, whose flat merge-by-location buffer could not represent
//! call-stack nesting; `CpuSpan::depth` fixes that by recording stack position
//! at push time.
//!
//! GPU timestamp capture lives in `flamegraph_gpu.rs` so that this module never
//! needs to depend on `wgpu`. The two modules communicate through the plain
//! (non-wgpu) types defined here: `GpuSpan`, `GpuPassKind`, `GpuClockCalibration`.

use std::{
    cell::{LazyCell, RefCell},
    collections::{HashMap, VecDeque},
    hash::{DefaultHasher, Hash, Hasher},
    io::Write,
    sync::{
        Arc, Weak,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread::ThreadId,
    time::{Instant, SystemTime},
};

use serde::Serialize;
use smallvec::SmallVec;

// The data model below derives `Serialize` only, not `Deserialize`. Several
// fields hold `&'static str` (`SpanName::Static`, `ElementAttribution`'s
// fields), and a genuinely `'static`-bound `Deserialize` impl can't be derived
// for those without leaking memory. `export_trace` is a write-only format this
// round (see module docs); the inline round-trip test in `tests` below
// decodes into separate owned mirror types instead of these live-process
// types, which is also what a future out-of-process viewer would do.

/// Coarse-grained category for a [`CpuSpan`] or [`GpuSpan`], used by a future
/// viewer to color/group spans without needing to parse span names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SpanCategory {
    /// A whole-window `Window::draw`/`Window::draw_roots` span.
    WindowFrame,
    /// `Drawable::request_layout`.
    ElementRequestLayout,
    /// `Drawable::prepaint`.
    ElementPrepaint,
    /// `Drawable::paint`.
    ElementPaint,
    /// A background-executor task.
    BackgroundTask,
    /// A GPU render pass bracketed by timestamp writes.
    GpuRenderPass,
    /// The whole-encoder submit-to-present bracket.
    GpuSubmitPresent,
    /// A span created through the public `flamegraph_span!` macro.
    UserDefined,
}

/// The name of a span. This round only produces `Static`; `Interned` exists so
/// that a future dynamic-name mechanism (e.g. per-entity type names built at
/// runtime) does not require a trace-format break.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SpanName {
    /// A `&'static str`, the common case for instrumentation call sites.
    Static(&'static str),
    /// An index into a future string-interning table. Unused in Phase 1.
    Interned(u32),
}

/// Attribution of a [`CpuSpan`] to the element that produced it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ElementAttribution {
    /// `type_name::<E>()` of the element.
    pub type_name: &'static str,
    /// A `seahash` hash of the element's `GlobalElementId`, computed only when
    /// capture is active (never unconditionally, per the zero-overhead rule).
    /// Zero when the element has no stable id.
    pub global_id_hash: u64,
    /// Source file and line, when available (reuses the existing
    /// `Component::source_location()` / `InspectorElementId` plumbing).
    pub source_location: Option<(&'static str, u32)>,
}

/// A hashed, serde-friendly stand-in for `std::thread::ThreadId`, which does
/// not implement `Serialize`/`Deserialize` itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct ThreadKey(u64);

impl ThreadKey {
    fn current() -> Self {
        let mut hasher = DefaultHasher::new();
        std::thread::current().id().hash(&mut hasher);
        ThreadKey(hasher.finish())
    }
}

/// A single completed CPU-side span.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct CpuSpan {
    /// Span name.
    pub name: SpanName,
    /// Span category.
    pub category: SpanCategory,
    /// Nesting depth, taken from the span stack's length at push time. This is
    /// what lets a viewer reconstruct call-stack nesting in O(1) per span,
    /// without walking a parent-pointer tree.
    pub depth: u16,
    /// Start time in nanoseconds relative to the owning `Capture`'s anchor.
    pub start_ns: u64,
    /// Duration in nanoseconds, saturating at `u32::MAX` (~4.29s) rather than
    /// wrapping, since a span that long is already pathological for a frame
    /// profiler and silent wraparound would be worse than a clamped value.
    pub duration_ns: u32,
    /// Which thread recorded this span.
    pub thread_id: ThreadKey,
    /// Element attribution, when this span was produced by element drawing.
    pub element: Option<ElementAttribution>,
}

/// Which of the renderer's timestamp-bracketed GPU passes a [`GpuSpan`] covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum GpuPassKind {
    /// The main render pass.
    Main,
    /// The main render pass, resumed after a filter-group interruption.
    MainResumed,
    /// A `with_filter_layer` offscreen group render pass.
    FilterGroup,
    /// A filter-group pass, resumed after nested filter groups.
    FilterGroupResumed,
    /// The fast (no-compositor) surface blit pass.
    FastSurfaceBlit,
    /// The whole-encoder submit-to-present bracket.
    SubmitPresent,
}

/// A single completed GPU-side span, resolved from a wgpu timestamp query pair.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct GpuSpan {
    /// Span name.
    pub name: SpanName,
    /// Start time in nanoseconds relative to the owning `Capture`'s anchor,
    /// converted from GPU ticks using the session's [`GpuClockCalibration`].
    pub start_ns: u64,
    /// Duration in nanoseconds.
    pub duration_ns: u32,
    /// Which pass this span covers.
    pub pass_kind: GpuPassKind,
    /// The `GpuQueryManager` generation this span's query pair came from, for
    /// diagnosing readback ordering issues in a future viewer.
    pub query_set_generation: u64,
}

/// One-time CPU/GPU clock calibration for a capture session, computed from a
/// bracketing pair of `write_timestamp` calls and `queue.get_timestamp_period()`.
/// Stored as plain data (no wgpu types) so it can live in this module.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize)]
pub struct GpuClockCalibration {
    /// CPU-side anchor, in nanoseconds since the `Capture`'s anchor `Instant`.
    pub cpu_anchor_ns: u64,
    /// GPU-side anchor, in raw timestamp ticks.
    pub gpu_anchor_ticks: u64,
    /// Nanoseconds per GPU timestamp tick (`queue.get_timestamp_period()`).
    pub ns_per_tick: f32,
    /// Whether calibration actually ran (false when no capture with
    /// `capture_gpu: true` has ever started).
    pub calibrated: bool,
}

/// All spans captured for one `Window::draw` cycle.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct FrameCapture {
    /// Monotonically increasing index, unique within a capture session.
    pub frame_index: u64,
    /// Hashed `WindowId` this frame belongs to (multi-window captures filter
    /// on this field rather than needing separate per-window captures).
    pub window_id: u64,
    /// Spans recorded on the thread that drew this frame (the foreground
    /// thread), in completion order.
    pub cpu_spans: Vec<CpuSpan>,
    /// Spans recorded on other threads whose `start_ns` fell within this
    /// frame's window.
    pub background_spans: Vec<CpuSpan>,
    /// GPU spans, attached asynchronously after query readback. May be empty
    /// even for an otherwise-complete frame; check `gpu_spans_finalized`.
    pub gpu_spans: Vec<GpuSpan>,
    /// Whether `gpu_spans` is done being populated. GPU spans typically arrive
    /// 1-2 frames after the CPU side closes, because of the resolve/map_async
    /// readback latency.
    pub gpu_spans_finalized: bool,
    /// Set when this frame recorded more GPU passes than
    /// `MAX_GPU_SPANS_PER_FRAME` could hold, so `gpu_spans` is incomplete.
    pub gpu_spans_truncated: bool,
    /// Frame start, in nanoseconds relative to the capture's anchor.
    pub frame_start_ns: u64,
    /// Frame end, in nanoseconds relative to the capture's anchor.
    pub frame_end_ns: u64,
}

/// A stopped capture session: a ring-buffered (by frame count) sequence of
/// [`FrameCapture`]s, ready to be inspected or exported.
#[derive(Debug)]
pub struct Capture {
    anchor: Instant,
    frames: VecDeque<FrameCapture>,
    /// Frame-count ring-buffer bound this capture was configured with.
    pub max_frames: usize,
    enabled: AtomicBool,
}

impl Capture {
    /// Iterate over captured frames, oldest first.
    pub fn frames(&self) -> impl Iterator<Item = &FrameCapture> {
        self.frames.iter()
    }

    /// Number of frames currently held.
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Write this capture out in WGPUI's versioned binary trace format:
    /// an 8-byte magic, a `u32` format version, a fixed `bytemuck::Pod` header,
    /// then a sequence of independently length-prefixed, bincode-encoded
    /// [`FrameCapture`] chunks (not one big `Vec<FrameCapture>` blob), so the
    /// format is streaming-friendly from day one. There is no public reader
    /// this round; a future viewer defines its own.
    pub fn export_trace(&self, writer: &mut impl Write) -> anyhow::Result<()> {
        let anchor_system_time = SystemTime::now()
            .checked_sub(self.anchor.elapsed())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let anchor_unix_nanos = anchor_system_time
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
            .min(u64::MAX as u128) as u64;

        let calibration = gpu_calibration();
        let header = TraceHeader {
            anchor_unix_nanos,
            gpu_cpu_anchor_ns: calibration.cpu_anchor_ns,
            gpu_anchor_ticks: calibration.gpu_anchor_ticks,
            frame_count: self.frames.len() as u32,
            cpu_clock_source: CPU_CLOCK_SOURCE_STD_INSTANT,
            gpu_ns_per_tick: calibration.ns_per_tick,
            gpu_calibrated: calibration.calibrated as u32,
            reserved: [0; 16],
        };

        writer.write_all(&TRACE_MAGIC)?;
        writer.write_all(&TRACE_FORMAT_VERSION.to_le_bytes())?;
        writer.write_all(bytemuck::bytes_of(&header))?;

        for frame in &self.frames {
            let encoded = bincode::serde::encode_to_vec(frame, bincode::config::standard())
                .map_err(|error| anyhow::anyhow!("failed to encode frame capture: {error}"))?;
            let length = u32::try_from(encoded.len())
                .map_err(|_| anyhow::anyhow!("frame capture too large to encode"))?;
            writer.write_all(&length.to_le_bytes())?;
            writer.write_all(&encoded)?;
        }

        Ok(())
    }
}

const TRACE_MAGIC: [u8; 8] = *b"WGPUIFLG";
const TRACE_FORMAT_VERSION: u32 = 1;
const CPU_CLOCK_SOURCE_STD_INSTANT: u32 = 0;

/// Fixed-size POD trace header. Field order is chosen so that no implicit
/// padding is inserted (three `u64`s, then four `u32`-sized fields, then a
/// byte array), which `bytemuck::Pod`'s derive requires.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TraceHeader {
    anchor_unix_nanos: u64,
    gpu_cpu_anchor_ns: u64,
    gpu_anchor_ticks: u64,
    frame_count: u32,
    cpu_clock_source: u32,
    gpu_ns_per_tick: f32,
    gpu_calibrated: u32,
    reserved: [u8; 16],
}

/// Error returned by [`start_capture`] when a capture session is already active.
#[derive(Debug, thiserror::Error)]
#[error("a flamegraph capture is already in progress")]
pub struct AlreadyCapturingError;

/// Options for [`start_capture`].
#[derive(Debug, Clone, Copy)]
pub struct CaptureOptions {
    /// Ring-buffer bound, in frames. Older frames are evicted once exceeded.
    pub max_frames: usize,
    /// Whether to also record GPU timestamp spans. When false, `flamegraph_gpu`
    /// never allocates a `QuerySet`, so GPU capture is also zero-cost when
    /// unused.
    pub capture_gpu: bool,
}

impl Default for CaptureOptions {
    fn default() -> Self {
        Self {
            max_frames: 600,
            capture_gpu: true,
        }
    }
}

/// A handle to an in-progress capture session. Dropping this without calling
/// [`CaptureHandle::stop`] leaves the capture running; there is intentionally
/// no `Drop`-based auto-stop, so that capture lifetime is explicit.
pub struct CaptureHandle {
    state: Arc<CaptureState>,
}

impl CaptureHandle {
    /// Stop the capture session and return the accumulated frames.
    pub fn stop(self) -> Capture {
        CAPTURE_ENABLED.store(false, Ordering::Release);
        *ACTIVE_CAPTURE.lock() = None;
        self.state.finalize_open_frames();

        Capture {
            anchor: self.state.anchor,
            frames: self.state.finished_frames.lock().clone(),
            max_frames: self.state.max_frames,
            enabled: AtomicBool::new(false),
        }
    }

    /// Whether a capture session is still active. Always true until `stop` is
    /// called (single-session invariant enforced by [`start_capture`]).
    pub fn is_recording(&self) -> bool {
        CAPTURE_ENABLED.load(Ordering::Relaxed)
    }
}

/// Whether a capture is currently active. Checked with `Ordering::Relaxed` by
/// every `enter_span` call; this is the entire cost of instrumentation when a
/// flamegraph-enabled build is not actively capturing.
static CAPTURE_ENABLED: AtomicBool = AtomicBool::new(false);

static ACTIVE_CAPTURE: parking_lot::Mutex<Option<Arc<CaptureState>>> = parking_lot::Mutex::new(None);

static GPU_CALIBRATION: parking_lot::Mutex<GpuClockCalibration> = parking_lot::Mutex::new(GpuClockCalibration {
    cpu_anchor_ns: 0,
    gpu_anchor_ticks: 0,
    ns_per_tick: 0.0,
    calibrated: false,
});

/// Start a new capture session. Only one session may be active at a time;
/// starting a second one while one is already running is a explicit error,
/// never a silent implicit restart of the previous session.
pub fn start_capture(options: CaptureOptions) -> Result<CaptureHandle, AlreadyCapturingError> {
    if CAPTURE_ENABLED
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return Err(AlreadyCapturingError);
    }

    let state = Arc::new(CaptureState::new(options));
    *ACTIVE_CAPTURE.lock() = Some(state.clone());
    Ok(CaptureHandle { state })
}

/// Whether a capture is currently recording. Cheap (`Ordering::Relaxed` atomic
/// load); call sites that need to do non-trivial attribution work (e.g.
/// hashing a `GlobalElementId`) should check this before doing that work,
/// rather than doing it unconditionally inside `enter_span`.
pub fn capture_enabled() -> bool {
    CAPTURE_ENABLED.load(Ordering::Relaxed)
}

/// Current session's GPU clock calibration, or a zeroed/`calibrated: false`
/// value if no GPU-capturing session has ever run. Used by
/// `present_synced_traced` to let embedding apps put their own wgpu
/// submissions on the same unified timeline.
pub fn gpu_calibration() -> GpuClockCalibration {
    *GPU_CALIBRATION.lock()
}

pub(crate) fn set_gpu_calibration(calibration: GpuClockCalibration) {
    *GPU_CALIBRATION.lock() = calibration;
}

/// Whether the active capture (if any) wants GPU spans. `flamegraph_gpu` uses
/// this to decide whether to lazily allocate its `QuerySet`s at all.
pub(crate) fn active_capture_wants_gpu() -> bool {
    ACTIVE_CAPTURE
        .lock()
        .as_ref()
        .map(|state| state.capture_gpu)
        .unwrap_or(false)
}

/// The most recently opened CPU-side frame index, used by `flamegraph_gpu` to
/// tag a `GpuQueryManager` generation with the frame it belongs to. See the
/// doc comment on `CaptureState::last_opened_frame_index` for why this is
/// safe to read this way instead of threading a frame index through
/// `PlatformWindow::draw`.
pub(crate) fn current_gpu_correlation_frame_index() -> Option<u64> {
    ACTIVE_CAPTURE.lock().as_ref().and_then(|state| {
        let index = state.last_opened_frame_index.load(Ordering::Acquire);
        (index != u64::MAX).then_some(index)
    })
}

/// Attach resolved GPU spans to the frame they belong to, if that frame is
/// still held by the active capture's ring buffer (it may have been evicted
/// already, given GPU readback latency of 1-2 frames).
pub(crate) fn attach_gpu_spans(frame_index: u64, spans: Vec<GpuSpan>, truncated: bool) {
    if let Some(state) = ACTIVE_CAPTURE.lock().as_ref() {
        let mut finished = state.finished_frames.lock();
        if let Some(frame) = finished.iter_mut().find(|frame| frame.frame_index == frame_index) {
            frame.gpu_spans = spans;
            frame.gpu_spans_finalized = true;
            frame.gpu_spans_truncated = truncated;
        }
    }
}

/// Open a new frame for CPU-side span bucketing. Returns `None` when no
/// capture is active, so callers can thread an `Option<u64>` through and skip
/// the matching `close_frame_cpu_side` call for free.
pub(crate) fn open_frame_cpu_side(window_id: u64) -> Option<u64> {
    if !capture_enabled() {
        return None;
    }
    ACTIVE_CAPTURE
        .lock()
        .as_ref()
        .map(|state| state.open_frame(window_id))
}

/// Close a frame opened with `open_frame_cpu_side`, bucketing completed spans
/// from this thread (as `cpu_spans`) and from other threads whose `start_ns`
/// falls in this frame's window (as `background_spans`), then finalizing it
/// into the capture's ring buffer. `gpu_spans` is left empty/unfinalized;
/// `attach_gpu_spans` fills it in later.
pub(crate) fn close_frame_cpu_side(frame_index: Option<u64>) {
    let Some(frame_index) = frame_index else {
        return;
    };
    if let Some(state) = ACTIVE_CAPTURE.lock().as_ref() {
        state.close_frame(frame_index);
    }
}

struct OpenFrame {
    window_id: u64,
    frame_start_ns: u64,
}

struct CaptureState {
    anchor: Instant,
    max_frames: usize,
    capture_gpu: bool,
    next_frame_index: AtomicU64,
    /// The most recently opened frame's index, or `u64::MAX` if none has been
    /// opened yet. `flamegraph_gpu`'s `GpuQueryManager` reads this (via
    /// `current_gpu_correlation_frame_index`) to tag its own timestamp-query
    /// generation with the CPU-side frame it belongs to, rather than needing
    /// a frame index threaded through the `PlatformWindow::draw` trait. This
    /// relies on GPUI's single-foreground-thread draw model (AGENTS.md: "All
    /// use of entities and UI rendering occurs on a single foreground
    /// thread."): `Window::draw` opens a frame and, still on that thread,
    /// synchronously drives rendering down into `WgpuRenderer::draw` before
    /// any other frame can be opened.
    last_opened_frame_index: AtomicU64,
    open_frames: parking_lot::Mutex<HashMap<u64, OpenFrame>>,
    finished_frames: parking_lot::Mutex<VecDeque<FrameCapture>>,
    /// Background-thread spans drained but not yet claimed by a frame window.
    /// Partitioned into a frame's `background_spans` at `close_frame` time.
    pending_background_spans: parking_lot::Mutex<Vec<CpuSpan>>,
}

impl CaptureState {
    fn new(options: CaptureOptions) -> Self {
        Self {
            anchor: Instant::now(),
            max_frames: options.max_frames.max(1),
            capture_gpu: options.capture_gpu,
            next_frame_index: AtomicU64::new(0),
            last_opened_frame_index: AtomicU64::new(u64::MAX),
            open_frames: parking_lot::Mutex::new(HashMap::new()),
            finished_frames: parking_lot::Mutex::new(VecDeque::new()),
            pending_background_spans: parking_lot::Mutex::new(Vec::new()),
        }
    }

    fn now_ns(&self) -> u64 {
        Instant::now().duration_since(self.anchor).as_nanos().min(u64::MAX as u128) as u64
    }

    fn open_frame(&self, window_id: u64) -> u64 {
        let frame_index = self.next_frame_index.fetch_add(1, Ordering::Relaxed);
        self.last_opened_frame_index.store(frame_index, Ordering::Release);
        self.open_frames.lock().insert(
            frame_index,
            OpenFrame {
                window_id,
                frame_start_ns: self.now_ns(),
            },
        );
        frame_index
    }

    fn close_frame(&self, frame_index: u64) {
        let Some(open_frame) = self.open_frames.lock().remove(&frame_index) else {
            return;
        };
        let frame_end_ns = self.now_ns();

        let cpu_spans = drain_current_thread_spans();
        collect_other_thread_spans_into(&self.pending_background_spans);
        let background_spans = {
            let mut pending = self.pending_background_spans.lock();
            let taken = std::mem::take(&mut *pending);
            let (in_window, remaining): (Vec<_>, Vec<_>) = taken.into_iter().partition(|span| {
                span.start_ns >= open_frame.frame_start_ns && span.start_ns < frame_end_ns
            });
            *pending = remaining;
            in_window
        };

        let frame = FrameCapture {
            frame_index,
            window_id: open_frame.window_id,
            cpu_spans,
            background_spans,
            gpu_spans: Vec::new(),
            gpu_spans_finalized: !self.capture_gpu,
            gpu_spans_truncated: false,
            frame_start_ns: open_frame.frame_start_ns,
            frame_end_ns,
        };

        let mut finished = self.finished_frames.lock();
        finished.push_back(frame);
        while finished.len() > self.max_frames {
            finished.pop_front();
        }
    }

    fn finalize_open_frames(&self) {
        let remaining: Vec<u64> = self.open_frames.lock().keys().copied().collect();
        for frame_index in remaining {
            self.close_frame(frame_index);
        }
    }
}

struct PendingSpan {
    name: SpanName,
    category: SpanCategory,
    element: Option<ElementAttribution>,
    start: Instant,
    depth: u16,
}

// Per-thread completed-span budget, mirroring `profiler.rs`'s flat 20MB
// budget but scoped per-thread and smaller, since frame-count ring-buffering
// on `Capture` is now the primary bound and this buffer is just a bridge
// until the next `close_frame_cpu_side` drains it.
const THREAD_SPAN_BUDGET_BYTES: usize = 2 * 1024 * 1024;
const MAX_THREAD_SPANS: usize = THREAD_SPAN_BUDGET_BYTES / core::mem::size_of::<CpuSpan>();

type ThreadSpans = circular_buffer::CircularBuffer<MAX_THREAD_SPANS, CpuSpan>;
type GuardedThreadRecorder = spin::Mutex<ThreadRecorder>;

struct ThreadRecorder {
    thread_id: ThreadId,
    completed: Box<ThreadSpans>,
}

struct GlobalThreadRecorderEntry {
    thread_id: ThreadId,
    recorder: Weak<GuardedThreadRecorder>,
}

static GLOBAL_THREAD_RECORDERS: spin::Mutex<Vec<GlobalThreadRecorderEntry>> = spin::Mutex::new(Vec::new());

impl Drop for ThreadRecorder {
    fn drop(&mut self) {
        let mut recorders = GLOBAL_THREAD_RECORDERS.lock();
        if let Some(index) = recorders.iter().position(|entry| entry.thread_id == self.thread_id) {
            recorders.swap_remove(index);
        }
    }
}

thread_local! {
    static SPAN_STACK: RefCell<SmallVec<[PendingSpan; 32]>> = RefCell::new(SmallVec::new());
    static THREAD_RECORDER: LazyCell<Arc<GuardedThreadRecorder>> = LazyCell::new(register_thread_recorder);
}

fn register_thread_recorder() -> Arc<GuardedThreadRecorder> {
    let thread_id = std::thread::current().id();
    let recorder = Arc::new(spin::Mutex::new(ThreadRecorder {
        thread_id,
        completed: ThreadSpans::boxed(),
    }));
    GLOBAL_THREAD_RECORDERS.lock().push(GlobalThreadRecorderEntry {
        thread_id,
        recorder: Arc::downgrade(&recorder),
    });
    recorder
}

fn drain_current_thread_spans() -> Vec<CpuSpan> {
    THREAD_RECORDER.with(|recorder| {
        let mut recorder = recorder.lock();
        let (first, second) = recorder.completed.as_slices();
        let mut spans = Vec::with_capacity(first.len() + second.len());
        spans.extend_from_slice(first);
        spans.extend_from_slice(second);
        recorder.completed.clear();
        spans
    })
}

fn collect_other_thread_spans_into(pending: &parking_lot::Mutex<Vec<CpuSpan>>) {
    let current_thread_id = std::thread::current().id();
    let recorders = GLOBAL_THREAD_RECORDERS.lock();
    let mut pending = pending.lock();
    for entry in recorders.iter() {
        if entry.thread_id == current_thread_id {
            continue;
        }
        let Some(recorder) = entry.recorder.upgrade() else {
            continue;
        };
        let mut recorder = recorder.lock();
        let (first, second) = recorder.completed.as_slices();
        pending.extend_from_slice(first);
        pending.extend_from_slice(second);
        recorder.completed.clear();
    }
}

/// RAII guard returned by [`enter_span`]. Dropping it closes the span. When
/// capture is disabled at the time `enter_span` was called, this is a no-op
/// guard that costs nothing to construct or drop.
pub struct SpanGuard {
    handle: Option<()>,
}

impl Drop for SpanGuard {
    fn drop(&mut self) {
        if self.handle.take().is_none() {
            return;
        }
        let completed = SPAN_STACK.with(|stack| stack.borrow_mut().pop());
        let Some(pending) = completed else {
            return;
        };
        let Some(anchor) = active_capture_anchor() else {
            return;
        };
        let now = Instant::now();
        let start_ns = pending.start.duration_since(anchor).as_nanos().min(u64::MAX as u128) as u64;
        let duration_ns = now
            .duration_since(pending.start)
            .as_nanos()
            .min(u32::MAX as u128) as u32;

        let span = CpuSpan {
            name: pending.name,
            category: pending.category,
            depth: pending.depth,
            start_ns,
            duration_ns,
            thread_id: ThreadKey::current(),
            element: pending.element,
        };

        THREAD_RECORDER.with(|recorder| {
            recorder.lock().completed.push_back(span);
        });
    }
}

fn active_capture_anchor() -> Option<Instant> {
    ACTIVE_CAPTURE.lock().as_ref().map(|state| state.anchor)
}

/// The active capture's CPU clock anchor, in the same `Instant` used to
/// compute `CpuSpan::start_ns`. `flamegraph_gpu` uses this during calibration
/// so `GpuSpan::start_ns` lands on the same timeline as CPU spans.
pub(crate) fn capture_anchor() -> Option<Instant> {
    active_capture_anchor()
}

/// Open a new CPU span. Returns a [`SpanGuard`] whose `Drop` closes the span.
///
/// When capture is disabled, this does a single `Ordering::Relaxed` atomic
/// load and returns immediately without touching `Instant::now()`, the
/// thread-local span stack, or allocating — the entire cost of instrumenting
/// a call site in a flamegraph-enabled-but-idle build.
pub fn enter_span(name: SpanName, category: SpanCategory, element: Option<ElementAttribution>) -> SpanGuard {
    if !capture_enabled() {
        return SpanGuard { handle: None };
    }

    SPAN_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        let depth = stack.len().min(u16::MAX as usize) as u16;
        stack.push(PendingSpan {
            name,
            category,
            element,
            start: Instant::now(),
            depth,
        });
    });

    SpanGuard { handle: Some(()) }
}

/// Open a CPU span with a category of [`SpanCategory::UserDefined`] and no
/// element attribution. Intended for use both by internal call sites and by
/// embedding applications.
#[macro_export]
macro_rules! flamegraph_span {
    ($name:expr) => {
        $crate::flamegraph_span!($name, $crate::SpanCategory::UserDefined)
    };
    ($name:expr, $category:expr) => {
        let _flamegraph_span_guard = $crate::enter_span($crate::SpanName::Static($name), $category, None);
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    // Owned mirror types used only to verify the wire format round-trips.
    // The live-process types above are intentionally `Serialize`-only (see
    // the module-level comment near their definitions): a real reader, in or
    // out of process, would define its own owned types like these rather
    // than deserialize directly into structs holding `&'static str`.

    #[derive(Debug, PartialEq, Deserialize)]
    enum DecodedSpanCategory {
        WindowFrame,
        ElementRequestLayout,
        ElementPrepaint,
        ElementPaint,
        BackgroundTask,
        GpuRenderPass,
        GpuSubmitPresent,
        UserDefined,
    }

    #[derive(Debug, PartialEq, Deserialize)]
    enum DecodedSpanName {
        Static(String),
        Interned(u32),
    }

    #[derive(Debug, PartialEq, Deserialize)]
    struct DecodedElementAttribution {
        type_name: String,
        global_id_hash: u64,
        source_location: Option<(String, u32)>,
    }

    #[derive(Debug, PartialEq, Deserialize)]
    struct DecodedCpuSpan {
        name: DecodedSpanName,
        category: DecodedSpanCategory,
        depth: u16,
        start_ns: u64,
        duration_ns: u32,
        thread_id: u64,
        element: Option<DecodedElementAttribution>,
    }

    #[derive(Debug, PartialEq, Deserialize)]
    enum DecodedGpuPassKind {
        Main,
        MainResumed,
        FilterGroup,
        FilterGroupResumed,
        FastSurfaceBlit,
        SubmitPresent,
    }

    #[derive(Debug, PartialEq, Deserialize)]
    struct DecodedGpuSpan {
        name: DecodedSpanName,
        start_ns: u64,
        duration_ns: u32,
        pass_kind: DecodedGpuPassKind,
        query_set_generation: u64,
    }

    #[derive(Debug, PartialEq, Deserialize)]
    struct DecodedFrameCapture {
        frame_index: u64,
        window_id: u64,
        cpu_spans: Vec<DecodedCpuSpan>,
        background_spans: Vec<DecodedCpuSpan>,
        gpu_spans: Vec<DecodedGpuSpan>,
        gpu_spans_finalized: bool,
        gpu_spans_truncated: bool,
        frame_start_ns: u64,
        frame_end_ns: u64,
    }

    fn read_trace(bytes: &[u8]) -> anyhow::Result<(TraceHeader, Vec<DecodedFrameCapture>)> {
        anyhow::ensure!(
            bytes.len() >= TRACE_MAGIC.len() + 4 + core::mem::size_of::<TraceHeader>(),
            "trace too short"
        );
        anyhow::ensure!(bytes[..8] == TRACE_MAGIC, "bad magic");
        let version = u32::from_le_bytes(bytes[8..12].try_into()?);
        anyhow::ensure!(version == TRACE_FORMAT_VERSION, "unsupported trace version");

        let header_size = core::mem::size_of::<TraceHeader>();
        let header_bytes = &bytes[12..12 + header_size];
        // `header_bytes` is a slice into a `Vec<u8>` starting at a fixed but
        // arbitrary byte offset, so it isn't guaranteed to satisfy
        // `TraceHeader`'s 8-byte alignment; read it unaligned instead of
        // reinterpreting the slice in place.
        let header: TraceHeader = bytemuck::pod_read_unaligned(header_bytes);

        let mut offset = 12 + header_size;
        let mut frames = Vec::with_capacity(header.frame_count as usize);
        for _ in 0..header.frame_count {
            anyhow::ensure!(bytes.len() >= offset + 4, "truncated frame length prefix");
            let length = u32::from_le_bytes(bytes[offset..offset + 4].try_into()?) as usize;
            offset += 4;
            anyhow::ensure!(bytes.len() >= offset + length, "truncated frame body");
            let (frame, _): (DecodedFrameCapture, usize) =
                bincode::serde::decode_from_slice(&bytes[offset..offset + length], bincode::config::standard())
                    .map_err(|error| anyhow::anyhow!("failed to decode frame capture: {error}"))?;
            offset += length;
            frames.push(frame);
        }

        Ok((header, frames))
    }

    fn decode(span: &CpuSpan) -> DecodedCpuSpan {
        DecodedCpuSpan {
            name: match span.name {
                SpanName::Static(name) => DecodedSpanName::Static(name.to_string()),
                SpanName::Interned(index) => DecodedSpanName::Interned(index),
            },
            category: decode_category(span.category),
            depth: span.depth,
            start_ns: span.start_ns,
            duration_ns: span.duration_ns,
            thread_id: span.thread_id.0,
            element: span.element.map(|element| DecodedElementAttribution {
                type_name: element.type_name.to_string(),
                global_id_hash: element.global_id_hash,
                source_location: element
                    .source_location
                    .map(|(file, line)| (file.to_string(), line)),
            }),
        }
    }

    fn decode_category(category: SpanCategory) -> DecodedSpanCategory {
        match category {
            SpanCategory::WindowFrame => DecodedSpanCategory::WindowFrame,
            SpanCategory::ElementRequestLayout => DecodedSpanCategory::ElementRequestLayout,
            SpanCategory::ElementPrepaint => DecodedSpanCategory::ElementPrepaint,
            SpanCategory::ElementPaint => DecodedSpanCategory::ElementPaint,
            SpanCategory::BackgroundTask => DecodedSpanCategory::BackgroundTask,
            SpanCategory::GpuRenderPass => DecodedSpanCategory::GpuRenderPass,
            SpanCategory::GpuSubmitPresent => DecodedSpanCategory::GpuSubmitPresent,
            SpanCategory::UserDefined => DecodedSpanCategory::UserDefined,
        }
    }

    fn decode_frame(frame: &FrameCapture) -> DecodedFrameCapture {
        DecodedFrameCapture {
            frame_index: frame.frame_index,
            window_id: frame.window_id,
            cpu_spans: frame.cpu_spans.iter().map(decode).collect(),
            background_spans: frame.background_spans.iter().map(decode).collect(),
            gpu_spans: Vec::new(),
            gpu_spans_finalized: frame.gpu_spans_finalized,
            gpu_spans_truncated: frame.gpu_spans_truncated,
            frame_start_ns: frame.frame_start_ns,
            frame_end_ns: frame.frame_end_ns,
        }
    }

    #[test]
    fn nesting_and_trace_round_trip() {
        let handle = start_capture(CaptureOptions {
            max_frames: 8,
            capture_gpu: false,
        })
        .expect("no other capture should be active in this test process at this point");

        for _ in 0..3 {
            let frame_index = open_frame_cpu_side(1);
            {
                let _draw = enter_span(SpanName::Static("Window::draw"), SpanCategory::WindowFrame, None);
                {
                    let _draw_roots =
                        enter_span(SpanName::Static("Window::draw_roots"), SpanCategory::WindowFrame, None);
                }
            }
            close_frame_cpu_side(frame_index);
        }

        let capture = handle.stop();
        assert_eq!(capture.frame_count(), 3);

        let frame = capture.frames().next().expect("at least one frame");
        let draw = frame
            .cpu_spans
            .iter()
            .find(|span| span.name == SpanName::Static("Window::draw"))
            .expect("Window::draw span recorded");
        let draw_roots = frame
            .cpu_spans
            .iter()
            .find(|span| span.name == SpanName::Static("Window::draw_roots"))
            .expect("Window::draw_roots span recorded");
        assert_eq!(draw.depth, 0, "Window::draw should be the outermost span");
        assert_eq!(draw_roots.depth, 1, "Window::draw_roots should nest one level under Window::draw");

        let mut buffer = Vec::new();
        capture.export_trace(&mut buffer).expect("export_trace should succeed");

        let (header, frames) = read_trace(&buffer).expect("round trip decode should succeed");
        assert_eq!(header.frame_count, 3);
        assert_eq!(frames.len(), 3);
        let expected: Vec<DecodedFrameCapture> = capture.frames().map(decode_frame).collect();
        assert_eq!(frames, expected);
    }
}
