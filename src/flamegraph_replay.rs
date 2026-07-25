//! Replay engine (Phase 6 of the profiling epic, see issue #62): reconstructs
//! and re-drives a previously triggered frame outside the code path that
//! originally produced it, building on Phase 4's [`crate::DeepCapture`] (GPU
//! command stream + fixed-buffer resource contents) and Phase 5's
//! [`crate::UiTreeCapture`] (element tree/layout/style/Scene snapshot).
//!
//! # Scope decisions (see the progress comments on issue #62 for the full
//! reasoning)
//!
//! 1. **In-process replay only, not from-disk trace deserialization.** Phase
//!    1's trace types (`FrameCapture` and friends, `flamegraph.rs`) are
//!    `Serialize`-only -- several fields hold `&'static str`, which blocks a
//!    clean `'static`-bound `Deserialize` impl (see that module's own
//!    doc comment). Building real owned-mirror `Deserialize` support for the
//!    binary trace format is a separate, harder problem than replay itself,
//!    so this module replays directly against a live [`crate::DeepCapture`]/
//!    [`crate::UiTreeCapture`] value already sitting in the calling process
//!    (typically just taken from [`crate::take_completed_deep_capture`]/
//!    [`crate::take_completed_ui_tree_capture`]), not a file loaded from
//!    disk days later. That is also what an in-app viewer (phase 7,
//!    `WGPUI-Component`) primarily wants: scrub what was just captured.
//! 2. **In-crate module behind `feature = "flamegraph"`, not a separate
//!    crate.** The viewer lives in a different repo but already depends on
//!    `gpui` directly, so a gated module costs it nothing extra to consume,
//!    versus standing up and validating a new workspace member.
//!
//! # GPU replay fidelity
//!
//! [`render_deep_capture_step`] re-submits a captured draw call's raw,
//! fixed-buffer resource bytes to a real [`wgpu::Device`]/[`wgpu::Queue`],
//! through the *actual* production `quads.wgsl` shader
//! (`platform/cross/shaders/quads.wgsl`, `include_str!`'d verbatim) --
//! `Quads` is the only shader-level primitive kind wired up to a real
//! pipeline this round. `quads.wgsl`'s vertex shader reads every field of
//! `Quad` directly out of a raw storage buffer indexed by
//! `@builtin(instance_index)`, so the exact bytes Phase 4 read back from the
//! live `quads_buffer` (`WgpuContext`, `platform/cross/render_context.rs`)
//! can be uploaded to a fresh buffer and drawn with unchanged pipeline state
//! -- no host-side decoding of `Quad`'s fields is needed to reproduce the
//! GPU's fragment output, only to know *which* instance range within the
//! buffer a given draw call covers (already recorded on
//! [`crate::DeepCaptureDrawCall`]).
//!
//! `Shadows`/`Underlines`/`BackdropFilters`/`Paths` reuse the same
//! raw-buffer-round-trip recipe in principle (all read a fixed
//! `WgpuContext` buffer the same way `Quads` does), but this round only
//! wires up `Quads`'s own pipeline; [`render_deep_capture_step`] returns
//! [`ReplayError::UnsupportedDrawCallKind`] for the others. Extending
//! coverage to them is mechanical (same recipe, different
//! `include_str!`'d shader and bind group layout) and is left for a
//! follow-up rather than multiplying this phase's shader-plumbing surface
//! area five-fold for a first cut -- consistent with Phase 4's own
//! documented cut of texture readback to #72.
//!
//! `MonoSprites`/`PolySprites` (atlas-textured) and `Surfaces` (embedded
//! surface content) have no real texture bytes to replay at all -- Phase 4's
//! atlas/surface texture readback was deferred to issue #72 and still
//! doesn't exist. [`DeepCaptureReplay::resource_status`] reports
//! [`DrawCallResourceStatus::TextureContentUnavailable`] for those draw
//! calls, and [`render_deep_capture_step`] degrades gracefully by returning
//! a generated checkerboard placeholder image
//! ([`placeholder_checkerboard_rgba`]) instead of erroring, so a viewer can
//! still show *something* at that step rather than getting stuck.

use crate::flamegraph::{DeepCapture, DeepCaptureBufferKind, DeepCaptureDrawCall, DrawCallKind};
use crate::flamegraph_ui_capture::{SceneSnapshot, UiElementNode, UiTreeCapture};

// ---------------------------------------------------------------------------
// CPU/UI-tree replay.

/// Reconstructs parent/child relationships from a [`UiTreeCapture`]'s flat,
/// depth-annotated DFS node list (`UiElementNode::depth`), so a caller can
/// walk the tree shape without re-deriving it from scratch every time. Pure
/// data transform, no live app state and no GPU involved -- can be built,
/// queried, and torn down entirely independent of the app that produced the
/// capture, and independent of GPU replay's device requirements.
#[derive(Debug, Clone)]
pub struct UiTreeReplay {
    capture: UiTreeCapture,
    parents: Vec<Option<usize>>,
    children: Vec<Vec<usize>>,
    roots: Vec<usize>,
}

impl UiTreeReplay {
    /// Build a replay view over `capture`. `O(n)` in the number of captured
    /// nodes: one pass with a depth-ordered stack, the same "keep popping
    /// while the top of the stack is not shallower than the current node"
    /// trick used to reconstruct a tree from a preorder depth list.
    pub fn new(capture: UiTreeCapture) -> Self {
        let node_count = capture.nodes.len();
        let mut parents: Vec<Option<usize>> = vec![None; node_count];
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); node_count];
        let mut roots: Vec<usize> = Vec::new();
        let mut stack: Vec<usize> = Vec::new();

        for (index, node) in capture.nodes.iter().enumerate() {
            while let Some(&top) = stack.last() {
                if capture.nodes[top].depth < node.depth {
                    break;
                }
                stack.pop();
            }
            match stack.last() {
                Some(&parent_index) => {
                    parents[index] = Some(parent_index);
                    children[parent_index].push(index);
                }
                None => roots.push(index),
            }
            stack.push(index);
        }

        Self {
            capture,
            parents,
            children,
            roots,
        }
    }

    /// The underlying capture this replay was built from.
    pub fn capture(&self) -> &UiTreeCapture {
        &self.capture
    }

    /// Number of nodes in the captured tree.
    pub fn node_count(&self) -> usize {
        self.capture.nodes.len()
    }

    /// The node at `index`, if `index` is in range.
    pub fn node(&self, index: usize) -> Option<&UiElementNode> {
        self.capture.nodes.get(index)
    }

    /// `index`'s parent, if it has one (root nodes return `None`).
    pub fn parent(&self, index: usize) -> Option<usize> {
        self.parents.get(index).copied().flatten()
    }

    /// `index`'s direct children, in the original DFS (paint) order.
    pub fn children(&self, index: usize) -> &[usize] {
        self.children.get(index).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Indices of every node with no captured parent (usually just one, the
    /// window's root element, but a capture that started mid-tree for any
    /// reason could have more).
    pub fn roots(&self) -> &[usize] {
        &self.roots
    }

    /// Every node index, in the same DFS order [`UiTreeCapture::nodes`]
    /// already stores them in -- a named accessor so callers don't need to
    /// know that detail of the underlying representation.
    pub fn depth_first_indices(&self) -> impl Iterator<Item = usize> + '_ {
        0..self.capture.nodes.len()
    }

    /// The frame's captured paint primitive list.
    pub fn scene(&self) -> &SceneSnapshot {
        &self.capture.scene
    }
}

// ---------------------------------------------------------------------------
// GPU replay: stepping over a `DeepCapture`'s command stream.

/// Whether the resource(s) a given draw call in a [`DeepCaptureReplay`]
/// depends on are actually available to replay against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawCallResourceStatus {
    /// The fixed buffer this draw call reads was touched by the capture and
    /// its readback completed -- [`render_deep_capture_step`] can use real
    /// captured bytes for it (though it may still not have a wired-up
    /// pipeline this round, see [`ReplayError::UnsupportedDrawCallKind`]).
    Available,
    /// This draw call references a fixed buffer
    /// ([`DeepCaptureDrawCall::buffer_kind`]) that the capture touched but
    /// whose readback did not complete (`DeepCapture::resources_finalized`
    /// was false, or this specific buffer's map failed).
    BufferReadbackMissing,
    /// This draw call reads atlas/surface texture content
    /// (`MonoSprites`/`PolySprites`/`Surfaces`), which Phase 4 never reads
    /// back (deferred to issue #72). There is nothing to replay faithfully;
    /// [`render_deep_capture_step`] substitutes a placeholder.
    TextureContentUnavailable,
    /// This draw call has no associated fixed-buffer resource at all (should
    /// not occur for a well-formed capture, but reported rather than
    /// panicking if the step index is out of range or otherwise
    /// unrecognized).
    NoResource,
}

/// Stepping cursor over one [`DeepCapture`]'s command stream --
/// draw-call-level granularity within a replayed frame, mirroring
/// RenderDoc's "step to next draw call" model. Pure bookkeeping over
/// already-captured data; does not itself touch a GPU device (see
/// [`render_deep_capture_step`] for the part that does).
#[derive(Debug, Clone)]
pub struct DeepCaptureReplay {
    capture: DeepCapture,
    cursor: usize,
}

impl DeepCaptureReplay {
    /// Build a replay session positioned at the first draw call (index `0`),
    /// mirroring RenderDoc opening a capture already stopped on a draw call
    /// rather than before/after the whole stream. Call
    /// [`Self::step_to_next_draw_call`]/[`Self::step_to_previous_draw_call`]/
    /// [`Self::seek`] to move the cursor.
    pub fn new(capture: DeepCapture) -> Self {
        Self { capture, cursor: 0 }
    }

    /// The underlying capture this replay was built from.
    pub fn capture(&self) -> &DeepCapture {
        &self.capture
    }

    /// Number of draw calls in the captured command stream.
    pub fn draw_call_count(&self) -> usize {
        self.capture.draw_calls.len()
    }

    /// The cursor's current position -- the index [`Self::current_draw_call`]
    /// reads from. Always `0` when the capture has no draw calls.
    pub fn current_step(&self) -> usize {
        self.cursor
    }

    /// The draw call at the cursor's current position, if any (`None` only
    /// when the capture has no draw calls at all).
    pub fn current_draw_call(&self) -> Option<&DeepCaptureDrawCall> {
        self.capture.draw_calls.get(self.cursor)
    }

    /// Move the cursor back to the first draw call.
    pub fn reset(&mut self) {
        self.cursor = 0;
    }

    /// Move the cursor to the next draw call and return it. Returns `None`
    /// without moving the cursor when already at the last draw call (or when
    /// the capture is empty) -- the cursor never runs off the end, so it is
    /// always safe to read [`Self::current_draw_call`] afterward.
    pub fn step_to_next_draw_call(&mut self) -> Option<&DeepCaptureDrawCall> {
        if self.cursor + 1 >= self.capture.draw_calls.len() {
            return None;
        }
        self.cursor += 1;
        self.capture.draw_calls.get(self.cursor)
    }

    /// Move the cursor to the previous draw call and return it. Returns
    /// `None` without moving the cursor when already at the first draw call
    /// (or when the capture is empty).
    pub fn step_to_previous_draw_call(&mut self) -> Option<&DeepCaptureDrawCall> {
        if self.cursor == 0 {
            return None;
        }
        self.cursor -= 1;
        self.capture.draw_calls.get(self.cursor)
    }

    /// Move the cursor directly to `step` and return the draw call there.
    /// Does nothing and returns `None` if `step` is out of range, leaving
    /// the cursor at its previous position.
    pub fn seek(&mut self, step: usize) -> Option<&DeepCaptureDrawCall> {
        if step >= self.capture.draw_calls.len() {
            return None;
        }
        self.cursor = step;
        self.capture.draw_calls.get(step)
    }

    /// Resource availability for the draw call at `step`, independent of the
    /// cursor's current position -- see [`DrawCallResourceStatus`].
    pub fn resource_status(&self, step: usize) -> DrawCallResourceStatus {
        let Some(call) = self.capture.draw_calls.get(step) else {
            return DrawCallResourceStatus::NoResource;
        };

        match call.kind {
            DrawCallKind::MonoSprites | DrawCallKind::PolySprites | DrawCallKind::Surfaces => {
                DrawCallResourceStatus::TextureContentUnavailable
            }
            _ => match call.buffer_kind {
                Some(kind) => {
                    if self.capture.buffer_contents(kind).is_some() {
                        DrawCallResourceStatus::Available
                    } else {
                        DrawCallResourceStatus::BufferReadbackMissing
                    }
                }
                None => DrawCallResourceStatus::NoResource,
            },
        }
    }
}

// ---------------------------------------------------------------------------
// GPU replay: real re-submission against a `wgpu::Device`/`wgpu::Queue`.

/// Errors from [`render_deep_capture_step`].
#[derive(Debug, thiserror::Error)]
pub enum ReplayError {
    /// `step` is not a valid index into the capture's command stream.
    #[error("draw call step {0} is out of range")]
    StepOutOfRange(usize),
    /// This draw call's `kind` has no wired-up replay pipeline this round.
    /// See this module's doc comment for which kinds are covered.
    #[error("no replay pipeline is wired up for draw call kind {0:?} yet")]
    UnsupportedDrawCallKind(DrawCallKind),
    /// The draw call references a fixed buffer whose readback never
    /// completed (see [`DrawCallResourceStatus::BufferReadbackMissing`]).
    #[error("buffer contents for {0:?} were not available in this capture")]
    MissingBufferContents(DeepCaptureBufferKind),
    /// `viewport_width`/`viewport_height` passed to
    /// [`render_deep_capture_step`] was zero.
    #[error("replay viewport must be non-zero in both dimensions")]
    EmptyViewport,
    /// A `wgpu` device-side operation failed (buffer map, adapter request,
    /// etc.). Carries a message rather than the original error type, since
    /// `wgpu`'s own error types are not uniformly `Clone`/`'static`-friendly
    /// across the operations this module performs.
    #[error("GPU replay failed: {0}")]
    Device(String),
}

/// The result of replaying one draw call's GPU work: the rendered contents
/// of an offscreen `viewport_width` x `viewport_height` target, as tightly
/// packed (no row padding) 8-bit RGBA.
#[derive(Debug, Clone)]
pub struct ReplayRenderOutput {
    /// Rendered target width, in pixels.
    pub width: u32,
    /// Rendered target height, in pixels.
    pub height: u32,
    /// Tightly packed 8-bit RGBA pixel data, `width * height * 4` bytes,
    /// row-major from the top-left.
    pub rgba8: Vec<u8>,
    /// `true` when this output is a generated placeholder (see
    /// [`placeholder_checkerboard_rgba`]) because the draw call's real
    /// texture content is unavailable
    /// ([`DrawCallResourceStatus::TextureContentUnavailable`]), rather than
    /// an actual GPU render of captured resource bytes.
    pub texture_unavailable: bool,
}

impl ReplayRenderOutput {
    /// The RGBA8 pixel at `(x, y)`, if in bounds.
    pub fn pixel(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let offset = ((y * self.width + x) * 4) as usize;
        let slice = self.rgba8.get(offset..offset + 4)?;
        Some([slice[0], slice[1], slice[2], slice[3]])
    }
}

/// Generate a placeholder checkerboard image, used by
/// [`render_deep_capture_step`] in place of real pixels for draw calls whose
/// texture content is unavailable (see this module's doc comment). `cell` is
/// the checkerboard square size in pixels (clamped to at least `1`).
pub fn placeholder_checkerboard_rgba(width: u32, height: u32, cell: u32) -> Vec<u8> {
    let cell = cell.max(1);
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    const LIGHT: [u8; 4] = [200, 200, 200, 255];
    const DARK: [u8; 4] = [120, 120, 120, 255];
    for y in 0..height {
        for x in 0..width {
            let checker = ((x / cell) + (y / cell)) % 2 == 0;
            pixels.extend_from_slice(if checker { &LIGHT } else { &DARK });
        }
    }
    pixels
}

/// Matches `quads.wgsl`'s `Globals` uniform layout exactly (`vec2<f32>` +
/// two `u32`s, 16 bytes, no implicit padding).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ReplayGlobals {
    viewport_size: [f32; 2],
    premultiplied_alpha: u32,
    pad: u32,
}

const REPLAY_TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Re-submit the draw call at `step` in `replay`'s command stream against a
/// real `device`/`queue`, reproducing its GPU output outside the live app
/// that originally recorded it. Renders into a fresh, offscreen
/// `viewport_width` x `viewport_height` target and reads the result back to
/// host memory -- this is a diagnostic/replay operation, not a hot-path one,
/// so it blocks on the GPU (`PollType::wait_indefinitely`) rather than
/// integrating with a caller's own per-frame poll loop, and builds its
/// pipeline fresh on every call rather than caching it across steps.
///
/// See this module's doc comment for exactly which [`DrawCallKind`]s have a
/// real pipeline wired up this round (`Quads` only) versus which degrade to
/// a placeholder (`MonoSprites`/`PolySprites`/`Surfaces`, via
/// [`placeholder_checkerboard_rgba`]) versus which return
/// [`ReplayError::UnsupportedDrawCallKind`].
pub fn render_deep_capture_step(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    replay: &DeepCaptureReplay,
    step: usize,
    viewport_width: u32,
    viewport_height: u32,
) -> Result<ReplayRenderOutput, ReplayError> {
    if viewport_width == 0 || viewport_height == 0 {
        return Err(ReplayError::EmptyViewport);
    }
    let call = replay
        .capture()
        .draw_calls
        .get(step)
        .ok_or(ReplayError::StepOutOfRange(step))?;

    match call.kind {
        DrawCallKind::MonoSprites | DrawCallKind::PolySprites | DrawCallKind::Surfaces => {
            return Ok(ReplayRenderOutput {
                width: viewport_width,
                height: viewport_height,
                rgba8: placeholder_checkerboard_rgba(viewport_width, viewport_height, 16),
                texture_unavailable: true,
            });
        }
        DrawCallKind::Quads => {}
        other => return Err(ReplayError::UnsupportedDrawCallKind(other)),
    }

    let buffer_kind = call.buffer_kind.ok_or(ReplayError::UnsupportedDrawCallKind(call.kind))?;
    let contents = replay
        .capture()
        .buffer_contents(buffer_kind)
        .ok_or(ReplayError::MissingBufferContents(buffer_kind))?;

    render_quads_step(device, queue, &contents.bytes, call, viewport_width, viewport_height)
}

fn render_quads_step(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    quads_bytes: &[u8],
    call: &DeepCaptureDrawCall,
    viewport_width: u32,
    viewport_height: u32,
) -> Result<ReplayRenderOutput, ReplayError> {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("flamegraph_replay_quads_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("platform/cross/shaders/quads.wgsl").into()),
    });

    let globals_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("flamegraph_replay_globals_bind_group_layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let quads_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("flamegraph_replay_quads_bind_group_layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("flamegraph_replay_quads_pipeline_layout"),
        bind_group_layouts: &[Some(&globals_bind_group_layout), Some(&quads_bind_group_layout)],
        immediate_size: 0,
    });

    let color_targets = [Some(wgpu::ColorTargetState {
        format: REPLAY_TARGET_FORMAT,
        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
        write_mask: wgpu::ColorWrites::ALL,
    })];

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("flamegraph_replay_quads_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_quad"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_quad"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &color_targets,
        }),
        multiview_mask: None,
        cache: None,
    });

    let globals = ReplayGlobals {
        viewport_size: [viewport_width as f32, viewport_height as f32],
        premultiplied_alpha: 0,
        pad: 0,
    };
    let globals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flamegraph_replay_globals_buffer"),
        size: core::mem::size_of::<ReplayGlobals>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&globals_buffer, 0, bytemuck::bytes_of(&globals));

    let quads_buffer_size = quads_bytes.len().max(1) as u64;
    let quads_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flamegraph_replay_quads_buffer"),
        size: quads_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !quads_bytes.is_empty() {
        queue.write_buffer(&quads_buffer, 0, quads_bytes);
    }

    let globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("flamegraph_replay_globals_bind_group"),
        layout: &globals_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: globals_buffer.as_entire_binding(),
        }],
    });
    let quads_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("flamegraph_replay_quads_bind_group"),
        layout: &quads_bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: quads_buffer.as_entire_binding(),
        }],
    });

    let target_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("flamegraph_replay_target_texture"),
        size: wgpu::Extent3d {
            width: viewport_width,
            height: viewport_height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: REPLAY_TARGET_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("flamegraph_replay_encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("flamegraph_replay_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &globals_bind_group, &[]);
        pass.set_bind_group(1, &quads_bind_group, &[]);
        pass.draw(call.vertex_range.0..call.vertex_range.1, call.instance_range.0..call.instance_range.1);
    }

    let bytes_per_pixel = 4u32;
    let unpadded_bytes_per_row = viewport_width * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = unpadded_bytes_per_row.div_ceil(align) * align;
    let staging_size = (padded_bytes_per_row as u64) * (viewport_height as u64);
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("flamegraph_replay_staging_buffer"),
        size: staging_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(viewport_height),
            },
        },
        wgpu::Extent3d {
            width: viewport_width,
            height: viewport_height,
            depth_or_array_layers: 1,
        },
    );

    queue.submit(Some(encoder.finish()));

    let map_ready = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let map_ok = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    {
        let map_ready = map_ready.clone();
        let map_ok = map_ok.clone();
        staging_buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            map_ok.store(result.is_ok(), std::sync::atomic::Ordering::Release);
            map_ready.store(true, std::sync::atomic::Ordering::Release);
        });
    }

    // Blocking wait: this is a diagnostic/replay call, not a per-frame hot
    // path -- see this function's doc comment.
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| ReplayError::Device(error.to_string()))?;
    if !map_ready.load(std::sync::atomic::Ordering::Acquire) || !map_ok.load(std::sync::atomic::Ordering::Acquire) {
        return Err(ReplayError::Device("staging buffer map_async did not complete successfully".to_string()));
    }

    let rgba8 = {
        let slice = staging_buffer.slice(..);
        let view = slice
            .get_mapped_range()
            .map_err(|error| ReplayError::Device(error.to_string()))?;
        let mut packed = Vec::with_capacity((unpadded_bytes_per_row as usize) * (viewport_height as usize));
        for row in 0..viewport_height as usize {
            let start = row * padded_bytes_per_row as usize;
            let end = start + unpadded_bytes_per_row as usize;
            packed.extend_from_slice(&view[start..end]);
        }
        packed
    };
    staging_buffer.unmap();

    Ok(ReplayRenderOutput {
        width: viewport_width,
        height: viewport_height,
        rgba8,
        texture_unavailable: false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flamegraph::{DeepCaptureBufferContents, DeepCaptureDrawCall};
    use crate::flamegraph_ui_capture::{SceneSnapshot, UiBounds, UiElementNode, UiTreeCapture};

    fn sample_ui_tree_capture() -> UiTreeCapture {
        // Depth-DFS shape:
        //   0: root (depth 0)
        //     1: child-a (depth 1)
        //       2: grandchild (depth 2)
        //     3: child-b (depth 1)
        UiTreeCapture {
            window_id: 7,
            nodes: vec![
                UiElementNode {
                    type_name: "Root",
                    global_id_hash: 1,
                    depth: 0,
                    bounds: UiBounds::default(),
                    style: None,
                },
                UiElementNode {
                    type_name: "ChildA",
                    global_id_hash: 2,
                    depth: 1,
                    bounds: UiBounds::default(),
                    style: None,
                },
                UiElementNode {
                    type_name: "Grandchild",
                    global_id_hash: 3,
                    depth: 2,
                    bounds: UiBounds::default(),
                    style: None,
                },
                UiElementNode {
                    type_name: "ChildB",
                    global_id_hash: 4,
                    depth: 1,
                    bounds: UiBounds::default(),
                    style: None,
                },
            ],
            scene: SceneSnapshot::default(),
        }
    }

    #[test]
    fn ui_tree_replay_reconstructs_parent_child_relationships() {
        let replay = UiTreeReplay::new(sample_ui_tree_capture());

        assert_eq!(replay.node_count(), 4);
        assert_eq!(replay.roots(), &[0]);
        assert_eq!(replay.parent(0), None);
        assert_eq!(replay.parent(1), Some(0));
        assert_eq!(replay.parent(2), Some(1));
        assert_eq!(replay.parent(3), Some(0), "child-b should reattach to root, not the grandchild");
        assert_eq!(replay.children(0), &[1, 3]);
        assert_eq!(replay.children(1), &[2]);
        assert!(replay.children(2).is_empty());
        assert!(replay.children(3).is_empty());

        assert_eq!(replay.node(1).map(|node| node.type_name), Some("ChildA"));
        assert_eq!(replay.depth_first_indices().collect::<Vec<_>>(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn ui_tree_replay_handles_empty_capture() {
        let replay = UiTreeReplay::new(UiTreeCapture::default());
        assert_eq!(replay.node_count(), 0);
        assert!(replay.roots().is_empty());
    }

    fn sample_deep_capture() -> DeepCapture {
        DeepCapture {
            draw_calls: vec![
                DeepCaptureDrawCall {
                    sequence: 0,
                    kind: DrawCallKind::Quads,
                    pipeline_label: "quads",
                    pass_label: "main",
                    vertex_range: (0, 4),
                    instance_range: (0, 1),
                    bind_group_count: 2,
                    buffer_kind: Some(DeepCaptureBufferKind::Quads),
                    atlas_texture_id: None,
                },
                DeepCaptureDrawCall {
                    sequence: 1,
                    kind: DrawCallKind::MonoSprites,
                    pipeline_label: "mono_sprites",
                    pass_label: "main",
                    vertex_range: (0, 4),
                    instance_range: (0, 2),
                    bind_group_count: 4,
                    buffer_kind: Some(DeepCaptureBufferKind::MonoSprites),
                    atlas_texture_id: Some(9),
                },
                DeepCaptureDrawCall {
                    sequence: 2,
                    kind: DrawCallKind::Shadows,
                    pipeline_label: "shadows",
                    pass_label: "main",
                    vertex_range: (0, 4),
                    instance_range: (0, 1),
                    bind_group_count: 2,
                    buffer_kind: Some(DeepCaptureBufferKind::Shadows),
                    atlas_texture_id: None,
                },
            ],
            buffer_contents: vec![DeepCaptureBufferContents {
                kind: DeepCaptureBufferKind::Quads,
                bytes: vec![0; 64],
            }],
            resources_finalized: false,
        }
    }

    #[test]
    fn deep_capture_replay_steps_through_command_stream() {
        let mut replay = DeepCaptureReplay::new(sample_deep_capture());

        assert_eq!(replay.draw_call_count(), 3);
        assert_eq!(replay.current_step(), 0);
        assert_eq!(replay.current_draw_call().map(|call| call.sequence), Some(0));

        let next = replay.step_to_next_draw_call().expect("should advance to the second draw call");
        assert_eq!(next.sequence, 1);
        assert_eq!(replay.current_step(), 1);

        let next = replay.step_to_next_draw_call().expect("should advance to the third draw call");
        assert_eq!(next.sequence, 2);
        assert_eq!(replay.current_step(), 2);

        assert!(
            replay.step_to_next_draw_call().is_none(),
            "already at the last draw call, stepping forward should report None"
        );
        assert_eq!(replay.current_step(), 2, "cursor should stay parked at the last valid draw call");

        let back = replay.step_to_previous_draw_call().expect("should step back to the second draw call");
        assert_eq!(back.sequence, 1);
        assert_eq!(replay.current_step(), 1);

        let seeked = replay.seek(0).expect("seeking within range should succeed");
        assert_eq!(seeked.sequence, 0);
        assert_eq!(replay.current_step(), 0);

        assert!(replay.seek(99).is_none(), "seeking out of range should report None");
        assert_eq!(replay.current_step(), 0, "cursor should not move on an out-of-range seek");

        replay.step_to_previous_draw_call();
        assert_eq!(replay.current_step(), 0, "stepping back from the first draw call should be a no-op");

        replay.seek(2);
        replay.reset();
        assert_eq!(replay.current_step(), 0);
    }

    #[test]
    fn deep_capture_replay_reports_resource_status_per_kind() {
        let replay = DeepCaptureReplay::new(sample_deep_capture());

        assert_eq!(replay.resource_status(0), DrawCallResourceStatus::Available);
        assert_eq!(replay.resource_status(1), DrawCallResourceStatus::TextureContentUnavailable);
        assert_eq!(
            replay.resource_status(2),
            DrawCallResourceStatus::BufferReadbackMissing,
            "Shadows buffer was never added to buffer_contents in this fixture"
        );
        assert_eq!(replay.resource_status(99), DrawCallResourceStatus::NoResource);
    }

    #[test]
    fn placeholder_checkerboard_has_expected_size_and_alternates() {
        let pixels = placeholder_checkerboard_rgba(4, 2, 1);
        assert_eq!(pixels.len(), 4 * 2 * 4);
        // Adjacent cells (cell size 1) should differ.
        let pixel_at = |x: usize, y: usize| -> [u8; 4] {
            let offset = (y * 4 + x) * 4;
            [pixels[offset], pixels[offset + 1], pixels[offset + 2], pixels[offset + 3]]
        };
        assert_ne!(pixel_at(0, 0), pixel_at(1, 0));
    }

    /// Creates a headless (surface-less) `wgpu::Device`/`Queue`, mirroring
    /// `flamegraph_gpu.rs`'s own test helper of the same shape (see that
    /// module's doc comment on why `enumerate_adapters` + pick-first is used
    /// instead of `request_adapter`, and why a missing adapter skips rather
    /// than fails the test in this sandbox).
    fn create_headless_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            backend_options: wgpu::BackendOptions::default(),
            memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
            display: None,
        });
        let adapter = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
            .into_iter()
            .next()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    /// Builds the raw bytes for one `Quad` the exact same way
    /// `WgpuRenderer::draw` does when uploading `scene.quads` to the GPU
    /// (`unsafe { as_bytes(&scene.quads) }`, `platform/cross/renderer.rs`) --
    /// a raw `#[repr(C)]` reinterpret, not a `bytemuck` cast, mirroring the
    /// production write path exactly so this test's captured bytes are
    /// genuinely representative of what Phase 4 would have read back from a
    /// live frame.
    fn quad_bytes(quad: &crate::scene::Quad) -> Vec<u8> {
        unsafe { std::slice::from_raw_parts(quad as *const crate::scene::Quad as *const u8, core::mem::size_of::<crate::scene::Quad>()) }
            .to_vec()
    }

    /// End-to-end GPU replay test: builds one opaque red `Quad` covering the
    /// whole replay viewport, captures its raw bytes the way a real
    /// `DeepCapture` would hold them, replays that single draw call against
    /// a real headless device, and asserts the rendered output's center
    /// pixel is red -- the concrete "reproduce the frame outside the live
    /// app" claim from issue #62, for the one primitive kind this phase
    /// wires up a real pipeline for.
    #[test]
    fn render_deep_capture_step_reproduces_a_captured_quad() {
        let Some((device, queue)) = create_headless_device() else {
            eprintln!("skipping render_deep_capture_step_reproduces_a_captured_quad: no wgpu adapter available");
            return;
        };

        let width = 32u32;
        let height = 32u32;
        let bounds = crate::Bounds {
            origin: crate::Point {
                x: crate::ScaledPixels::from(0.0),
                y: crate::ScaledPixels::from(0.0),
            },
            size: crate::Size {
                width: crate::ScaledPixels::from(width as f32),
                height: crate::ScaledPixels::from(height as f32),
            },
        };
        let red: crate::Hsla = crate::rgb(0xff0000).into();
        let quad = crate::scene::Quad {
            order: 0,
            border_style: crate::BorderStyle::default(),
            bounds,
            content_mask: crate::ContentMask { bounds },
            background: crate::solid_background(red),
            border_color: crate::Hsla { h: 0.0, s: 0.0, l: 0.0, a: 0.0 },
            corner_radii: Default::default(),
            border_widths: Default::default(),
        };

        let capture = DeepCapture {
            draw_calls: vec![DeepCaptureDrawCall {
                sequence: 0,
                kind: DrawCallKind::Quads,
                pipeline_label: "quads",
                pass_label: "main",
                vertex_range: (0, 4),
                instance_range: (0, 1),
                bind_group_count: 2,
                buffer_kind: Some(DeepCaptureBufferKind::Quads),
                atlas_texture_id: None,
            }],
            buffer_contents: vec![DeepCaptureBufferContents {
                kind: DeepCaptureBufferKind::Quads,
                bytes: quad_bytes(&quad),
            }],
            resources_finalized: true,
        };

        let replay = DeepCaptureReplay::new(capture);
        assert_eq!(replay.resource_status(0), DrawCallResourceStatus::Available);

        let output = render_deep_capture_step(&device, &queue, &replay, 0, width, height)
            .expect("replaying the captured quad draw call should succeed");

        assert!(!output.texture_unavailable);
        assert_eq!(output.width, width);
        assert_eq!(output.height, height);

        let pixel = output.pixel(width / 2, height / 2).expect("center pixel should be in bounds");
        assert!(pixel[0] > 200, "expected a strongly red center pixel, got {pixel:?}");
        assert!(pixel[1] < 60, "expected little green in the center pixel, got {pixel:?}");
        assert!(pixel[2] < 60, "expected little blue in the center pixel, got {pixel:?}");
        assert!(pixel[3] > 200, "expected an opaque center pixel, got {pixel:?}");
    }

    #[test]
    fn render_deep_capture_step_reports_texture_unavailable_placeholder_for_sprites() {
        let Some((device, queue)) = create_headless_device() else {
            eprintln!(
                "skipping render_deep_capture_step_reports_texture_unavailable_placeholder_for_sprites: no wgpu adapter available"
            );
            return;
        };

        let replay = DeepCaptureReplay::new(sample_deep_capture());
        let output = render_deep_capture_step(&device, &queue, &replay, 1, 8, 8)
            .expect("sprite draw calls should degrade gracefully rather than error");
        assert!(output.texture_unavailable);
        assert_eq!(output.rgba8.len(), 8 * 8 * 4);
    }

    #[test]
    fn render_deep_capture_step_rejects_unwired_kinds() {
        let Some((device, queue)) = create_headless_device() else {
            eprintln!("skipping render_deep_capture_step_rejects_unwired_kinds: no wgpu adapter available");
            return;
        };

        let replay = DeepCaptureReplay::new(sample_deep_capture());
        let error = render_deep_capture_step(&device, &queue, &replay, 2, 8, 8)
            .expect_err("Shadows has no wired-up replay pipeline this round");
        assert!(matches!(error, ReplayError::UnsupportedDrawCallKind(DrawCallKind::Shadows)));
    }
}
