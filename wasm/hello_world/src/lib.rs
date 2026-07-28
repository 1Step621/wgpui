use gpui::{
    App, Application, Bounds, Context, Window, WindowBounds, WindowOptions, div, prelude::*, px,
    rgb, size,
};
use wasm_bindgen::prelude::*;

struct HelloWasm;

impl gpui::Render for HelloWasm {
    fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl gpui::IntoElement {
        div()
            .flex()
            .flex_col()
            .gap_3()
            .bg(rgb(0x2a6f97))
            .size_full()
            .justify_center()
            .items_center()
            .p_4()
            .child(
                div()
                    .flex()
                    .flex_row()
                    .gap_2()
                    .child(div().size_16().bg(gpui::red()).rounded_md())
                    .child(div().size_16().bg(gpui::green()).rounded_md())
                    .child(div().size_16().bg(gpui::blue()).rounded_md())
                    .child(div().size_16().bg(gpui::yellow()).rounded_md())
                    .child(
                        div()
                            .size_16()
                            .bg(rgb(0xffffff))
                            .border_2()
                            .border_color(gpui::black())
                            .rounded_md(),
                    ),
            )
            .child(
                div()
                    .mt_4()
                    .bg(rgb(0x1b3a5c))
                    .px_6()
                    .py_3()
                    .rounded_lg()
                    .border_1()
                    .border_color(rgb(0x4a9eff)),
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
            |_, cx| cx.new(|_| HelloWasm),
        )
        .unwrap();
        cx.activate(true);
    });
}
