use gpui::{
    div, prelude::*, px, rgb, size, App, Application, Bounds, Context, SharedString, Window,
    WindowBounds, WindowOptions,
};
use wasm_bindgen::prelude::*;

fn elapsed_secs() -> f64 {
    web_sys::window()
        .and_then(|w| w.performance())
        .map(|p| p.now() / 1000.0)
        .unwrap_or(0.0)
}

struct HelloWasm {
    start: f64,
    text: SharedString,
}

impl gpui::Render for HelloWasm {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl gpui::IntoElement {
        cx.notify();
        let elapsed = (elapsed_secs() - self.start) as f32;
        let view = window.viewport_size();
        let max_w = (view.width - px(48.0)).max(px(0.0));
        let max_h = (view.height - px(48.0)).max(px(0.0));
        let tx = (elapsed * 1.2) % 2.0;
        let ty = (elapsed * 0.9) % 2.0;
        let prog_x = if tx < 1.0 { tx } else { 2.0 - tx };
        let prog_y = if ty < 1.0 { ty } else { 2.0 - ty };
        let x = max_w * prog_x;
        let y = max_h * prog_y;

        div()
            .relative()
            .size_full()
            .bg(rgb(0x1a1a2e))
            .child(
                div()
                    .absolute()
                    .left(x)
                    .top(y)
                    .size_12()
                    .bg(rgb(0xe94560))
                    .rounded_full()
                    .shadow_lg(),
            )
            .child(
                div()
                    .flex()
                    .absolute()
                    .bottom(px(16.0))
                    .left(px(16.0))
                    .bg(rgb(0x16213e))
                    .px_3()
                    .py_1()
                    .rounded_md()
                    .text_color(rgb(0xaaaaaa))
                    .child(format!("{}", &self.text)),
            )
    }
}

#[wasm_bindgen(start)]
pub fn start() {
    console_error_panic_hook::set_once();
    Application::new().run(|cx: &mut App| {
        let bounds = Bounds::centered(None, size(px(800.), px(600.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| {
                cx.new(|_| HelloWasm {
                    start: elapsed_secs(),
                    text: "WGPUI on WASM".into(),
                })
            },
        )
        .unwrap();
        cx.activate(true);
    });
}
