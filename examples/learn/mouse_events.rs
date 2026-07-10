//! Mouse Events & Drag Hover Example
//!
//! Demonstrates the new on_mouse_enter, on_mouse_leave, and on_drag_hover callbacks.
//!
//! ## How to test
//!
//! ### on_mouse_enter / on_mouse_leave
//! 1. Move your mouse into the blue box — "Entered" lights up, box turns green
//! 2. Move your mouse out of the blue box — "Left" lights up, box turns blue again
//! 3. These fire even while a drag is active (unlike on_hover)
//!
//! ### on_drag_hover
//! 1. Click and drag the orange "Drag Me" box
//! 2. While dragging, move the cursor over the blue box — "Dragging Over" lights up
//! 3. Move the cursor out of the blue box while still dragging — clears
//! 4. Release to drop

#[path = "../prelude.rs"]
mod example_prelude;
use example_prelude::init_example;

use gpui::{
    App, Application, Bounds, Context, Entity, Hsla, IntoElement, MouseButton,
    Pixels, Point, Render, Styled, Window, WindowBounds, WindowOptions, div, prelude::*, px, size,
};

struct DragPayload;

struct MouseEventsExample {
    /// True while the cursor is inside the hover target (any state, including drags)
    hovered: bool,
    /// True while a drag of DragPayload is inside the hover target
    drag_hovered: bool,
}

impl Render for MouseEventsExample {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let bg = if self.drag_hovered {
            gpui::green()
        } else if self.hovered {
            gpui::green()
        } else {
            gpui::blue()
        };

        let on_enter = cx.entity().downgrade();
        let on_leave = cx.entity().downgrade();
        let on_drag_enter = cx.entity().downgrade();
        let on_drag_leave = cx.entity().downgrade();

        div()
            .size_full()
            .p_12()
            .flex()
            .flex_col()
            .gap_8()
            .bg(gpui::rgb(0x1a1a2e))
            .child(
                div()
                    .text_color(gpui::white())
                    .text_xl()
                    .child("Mouse Events Playground"),
            )
            .child(
                div()
                    .text_color(gpui::rgb(0x8888aa))
                    .text_sm()
                    .child("1. Hover the blue box — on_mouse_enter/leave fire")
                    .child("2. Drag the orange box over the blue box — on_drag_hover fires"),
            )
            .child(
                div()
                    .flex()
                    .flex_row()
                    .gap_6()
                    .items_start()
                    .child(self.render_status_panel())
                    .child(
                        div()
                            .w(px(200.))
                            .h(px(200.))
                            .rounded_lg()
                            .bg(bg)
                            .flex()
                            .items_center()
                            .justify_center()
                            .text_color(gpui::white())
                            .text_lg()
                            .child(
                                if self.drag_hovered {
                                    "DROP HERE"
                                } else if self.hovered {
                                    "HOVERED"
                                } else {
                                    "TARGET"
                                },
                            )
                            .on_mouse_enter(move |_, cx| {
                                let _ = on_enter.update(cx, |this, _| this.hovered = true);
                            })
                            .on_mouse_leave(move |_, cx| {
                                let _ = on_leave.update(cx, |this, _| this.hovered = false);
                            })
                            .on_drag_hover::<DragPayload>(move |&hovered, _, cx| {
                                if hovered {
                                    let _ = on_drag_enter.update(cx, |this, _| this.drag_hovered = true);
                                } else {
                                    let _ = on_drag_leave.update(cx, |this, _| this.drag_hovered = false);
                                }
                            }),
                    )
                    .child(
                        div()
                            .w(px(120.))
                            .h(px(120.))
                            .rounded_lg()
                            .bg(gpui::rgb(0xe85d04))
                            .flex()
                            .items_center()
                            .justify_center()
                            .text_color(gpui::white())
                            .child("Drag Me")
                            .on_drag(DragPayload, |_, _, _, cx| {
                                cx.new(|_| gpui::Empty)
                            }),
                    ),
            )
    }
}

impl MouseEventsExample {
    fn render_status_panel(&self) -> impl IntoElement {
        let hovered = self.hovered;
        let drag_hovered = self.drag_hovered;
        div()
            .flex()
            .flex_col()
            .gap_3()
            .p_6()
            .rounded_lg()
            .bg(gpui::rgb(0x16213e))
            .border_1()
            .border_color(gpui::rgb(0x0f3460))
            .child(
                div()
                    .text_color(gpui::white())
                    .child("Status"),
            )
            .child(status_row("Mouse Entered", hovered))
            .child(status_row("Mouse Left", !hovered))
            .child(status_row("Dragging Over", drag_hovered))
    }
}

fn status_row(label: &str, active: bool) -> impl IntoElement {
    let color: Hsla = if active { gpui::green() } else { gpui::rgb(0x333333).into() };
    let text_color: Hsla = if active { gpui::white() } else { gpui::rgb(0x666666).into() };
    div()
        .flex()
        .items_center()
        .gap_2()
        .child(div().w(px(12.)).h(px(12.)).rounded_full().bg(color))
        .child(
            div()
                .text_color(text_color)
                .child(label.to_string()),
        )
}

fn main() {
    Application::new().run(|cx: &mut App| {
        init_example(cx, "Mouse Events");
        let bounds = Bounds::centered(None, size(px(800.), px(600.)), cx);
        cx.open_window(
            WindowOptions {
                window_bounds: Some(WindowBounds::Windowed(bounds)),
                ..Default::default()
            },
            |_, cx| {
                cx.new(|_| MouseEventsExample {
                    hovered: false,
                    drag_hovered: false,
                })
            },
        )
        .unwrap();
    });
}
