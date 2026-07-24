// Blits the persistent framebuffer onto the acquired swapchain image.
//
// The compositor renders every pass into `persistent_framebuffer`, which retains
// the full previous frame. That makes `LoadOp::Load` meaningful for incremental
// paths (see `blit_surfaces_direct`), which cannot rely on swapchain contents:
// the swapchain rotates through several images, so a "previous" image is two or
// three frames stale.
//
// The framebuffer and the swapchain share a format, so `textureSample` performs
// no colour conversion here and the blit is an exact 1:1 passthrough.

@group(0) @binding(0) var t_framebuffer: texture_2d<f32>;
@group(0) @binding(1) var s_framebuffer: sampler;

struct BlitVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
}

// Oversized fullscreen triangle: cheaper than a quad and avoids the diagonal
// seam two triangles can produce.
@vertex
fn vs_blit(@builtin(vertex_index) vertex_id: u32) -> BlitVarying {
    let uv = vec2<f32>(f32((vertex_id << 1u) & 2u), f32(vertex_id & 2u));

    var out: BlitVarying;
    out.position = vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    out.tex_coord = uv;
    return out;
}

@fragment
fn fs_blit(input: BlitVarying) -> @location(0) vec4<f32> {
    return textureSample(t_framebuffer, s_framebuffer, input.tex_coord);
}
