//! A horizontally-scrollable virtual list. Each item is measured individually so
//! that items with different natural widths (e.g. tab labels of varying length)
//! are all positioned correctly without clipping.  Suitable for small-to-medium
//! sized lists (tens to low hundreds of items).

use super::ListHorizontalSizingBehavior;
use crate::elements::smooth_scroll::SmoothScrollState;
use crate::{
    AnyElement, App, AvailableSpace, Bounds, ContentMask, Element, ElementId, Entity,
    GlobalElementId, Hitbox, InspectorElementId, InteractiveElement, Interactivity, IntoElement,
    IsZero, LayoutId, ListSizingBehavior, Overflow, Pixels, Point, ScrollHandle, Size,
    StyleRefinement, Styled, Window, point, px, size,
};
use smallvec::SmallVec;
use std::{cell::RefCell, cmp, ops::Range, rc::Rc, usize};

/// Create a new horizontal virtual list.  Items are given [`AvailableSpace::MinContent`]
/// width so each one reports its natural size; positions are calculated from the
/// cumulative sum of all measured widths.
#[track_caller]
pub fn h_list<R>(
    id: impl Into<ElementId>,
    item_count: usize,
    f: impl 'static + Fn(Range<usize>, &mut Window, &mut App) -> Vec<R>,
) -> HList
where
    R: IntoElement,
{
    let id = id.into();
    let mut base_style = StyleRefinement::default();
    base_style.overflow.x = Some(Overflow::Scroll);
    base_style.overflow.y = Some(Overflow::Hidden);

    let render_range = move |range: Range<usize>, window: &mut Window, cx: &mut App| {
        f(range, window, cx)
            .into_iter()
            .map(|component| component.into_any_element())
            .collect()
    };

    HList {
        item_count,
        render_items: Box::new(render_range),
        item_widths: Vec::new(),
        interactivity: Interactivity {
            element_id: Some(id),
            base_style: Box::new(base_style),
            ..Interactivity::new()
        },
        scroll_handle: None,
        sizing_behavior: ListSizingBehavior::default(),
        horizontal_sizing_behavior: ListHorizontalSizingBehavior::default(),
    }
}

/// A horizontally-scrollable virtual list with per-item width measurement.
pub struct HList {
    item_count: usize,
    render_items: Box<
        dyn for<'a> Fn(Range<usize>, &'a mut Window, &'a mut App) -> SmallVec<[AnyElement; 64]>,
    >,
    /// Per-item widths, populated lazily on the first prepaint.
    item_widths: Vec<Pixels>,
    interactivity: Interactivity,
    scroll_handle: Option<HListScrollHandle>,
    sizing_behavior: ListSizingBehavior,
    horizontal_sizing_behavior: ListHorizontalSizingBehavior,
}

/// Per-frame rendering state for an [`HList`].
pub struct HListFrameState {
    items: SmallVec<[AnyElement; 32]>,
}

/// A handle for controlling the scroll position of a horizontal list.
#[derive(Clone, Debug, Default)]
pub struct HListScrollHandle(pub Rc<RefCell<HListScrollState>>);

/// Scroll and animation state for an [`HList`].
#[derive(Debug, Default)]
pub struct HListScrollState {
    /// The underlying GPUI scroll handle.
    pub base_handle: ScrollHandle,
    /// A deferred scroll-to-item request to be consumed during prepaint.
    pub deferred_scroll_to_item: Option<HListDeferredScroll>,
    /// The size of the list and its contents from the last layout pass.
    pub last_item_size: Option<HListItemSize>,
    /// Smooth scrolling animation state.
    pub smooth_scroll: SmoothScrollState,
}

/// A deferred request to scroll an [`HList`] to a specific item.
#[derive(Clone, Copy, Debug)]
pub struct HListDeferredScroll {
    /// The index of the item to scroll to.
    pub item_index: usize,
    /// The scroll strategy.
    pub strategy: HListScrollStrategy,
    /// Number of items offset from the viewport edge.
    pub offset: usize,
    /// If true, always scroll even if the item is already visible.
    pub scroll_strict: bool,
}

/// Measured size of an item and its container during layout.
#[derive(Copy, Clone, Debug, Default)]
pub struct HListItemSize {
    /// The size of the item.
    pub item: Size<Pixels>,
    /// The size of the item's contents.
    pub contents: Size<Pixels>,
}

/// Scroll strategy for bringing an item into view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HListScrollStrategy {
    /// Scroll so the item is at the start (left) of the viewport.
    Start,
    /// Scroll so the item is centered in the viewport.
    Center,
    /// Scroll so the item is at the end (right) of the viewport.
    End,
    /// Scroll only as much as needed to make the item visible.
    Nearest,
}

impl HListScrollHandle {
    /// Create a new scroll handle.
    pub fn new() -> Self {
        Self(Rc::new(RefCell::new(HListScrollState {
            base_handle: ScrollHandle::new(),
            deferred_scroll_to_item: None,
            last_item_size: None,
            smooth_scroll: SmoothScrollState::default(),
        })))
    }

    /// Request a non-strict scroll to bring the given item into view.
    pub fn scroll_to_item(&self, ix: usize, strategy: HListScrollStrategy) {
        self.0.borrow_mut().deferred_scroll_to_item = Some(HListDeferredScroll {
            item_index: ix,
            strategy,
            offset: 0,
            scroll_strict: false,
        });
    }

    /// Request a strict scroll to bring the given item into view.
    pub fn scroll_to_item_strict(&self, ix: usize, strategy: HListScrollStrategy) {
        self.0.borrow_mut().deferred_scroll_to_item = Some(HListDeferredScroll {
            item_index: ix,
            strategy,
            offset: 0,
            scroll_strict: true,
        });
    }
}

impl Styled for HList {
    fn style(&mut self) -> &mut StyleRefinement {
        &mut self.interactivity.base_style
    }
}

impl HList {
    /// Ensure all items have been measured.
    fn ensure_widths(&mut self, window: &mut Window, cx: &mut App) {
        if self.item_widths.len() == self.item_count || self.item_count == 0 {
            return;
        }
        self.item_widths.clear();
        for ix in 0..self.item_count {
            let mut items = (self.render_items)(ix..ix + 1, window, cx);
            let width = items
                .pop()
                .and_then(|mut item| {
                    let sz = item.layout_as_root(
                        size(AvailableSpace::MinContent, AvailableSpace::MinContent),
                        window,
                        cx,
                    );
                    (sz.width > Pixels::ZERO).then_some(sz.width)
                })
                .unwrap_or(px(80.));
            self.item_widths.push(width);
        }
    }
}

impl Element for HList {
    type RequestLayoutState = HListFrameState;
    type PrepaintState = Option<Hitbox>;

    fn id(&self) -> Option<ElementId> {
        self.interactivity.element_id.clone()
    }

    fn source_location(&self) -> Option<&'static core::panic::Location<'static>> {
        None
    }

    fn request_layout(
        &mut self,
        global_id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        window: &mut Window,
        cx: &mut App,
    ) -> (LayoutId, Self::RequestLayoutState) {
        let max_items = self.item_count;
        let item_size = self
            .item_widths
            .first()
            .copied()
            .map(|w| Size { width: w, height: px(24.) })
            .unwrap_or_else(|| self.measure_one(None, window, cx));

        let layout_id = self.interactivity.request_layout(
            global_id,
            inspector_id,
            window,
            cx,
            |style, window, cx| match self.sizing_behavior {
                ListSizingBehavior::Infer => {
                    window.with_text_style(style.text_style().cloned(), |window| {
                        window.request_measured_layout(
                            style,
                            move |known_dimensions, available_space, _window, _cx| {
                                let desired_width = item_size.width * max_items as f32;
                                let height = known_dimensions.height.unwrap_or(
                                    match available_space.height {
                                        AvailableSpace::Definite(x) => x,
                                        AvailableSpace::MinContent | AvailableSpace::MaxContent => {
                                            item_size.height
                                        }
                                    },
                                );
                                let width = match available_space.width {
                                    AvailableSpace::Definite(w) => desired_width.min(w),
                                    AvailableSpace::MinContent | AvailableSpace::MaxContent => {
                                        desired_width
                                    }
                                };
                                size(width, height)
                            },
                        )
                    })
                }
                ListSizingBehavior::Auto => window
                    .with_text_style(style.text_style().cloned(), |window| {
                        window.request_layout(style, None, cx)
                    }),
            },
        );

        (
            layout_id,
            HListFrameState {
                items: SmallVec::new(),
            },
        )
    }

    fn prepaint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        frame_state: &mut Self::RequestLayoutState,
        window: &mut Window,
        cx: &mut App,
    ) -> Option<Hitbox> {
        let style = self
            .interactivity
            .compute_style(global_id, None, window, cx);
        let border = style.border_widths.to_pixels(window.rem_size());
        let padding = style
            .padding
            .to_pixels(bounds.size.into(), window.rem_size());

        let padded_bounds = Bounds::from_corners(
            bounds.origin + point(border.left + padding.left, border.top + padding.top),
            bounds.bottom_right()
                - point(border.right + padding.right, border.bottom + padding.bottom),
        );

        // Lazily measure all items so we have per-item widths.
        self.ensure_widths(window, cx);

        let content_width: Pixels = self
            .item_widths
            .iter()
            .fold(Pixels::ZERO, |acc, &w| acc + w);
        let content_height = padded_bounds.size.height.max(px(24.));
        let content_size = Size {
            width: content_width,
            height: content_height,
        };

        let shared_scroll_offset = self
            .interactivity
            .scroll_offset
            .clone()
            .unwrap_or_else(|| Rc::new(RefCell::new(Point::default())));
        let mut logical_scroll_offset = *shared_scroll_offset.borrow();
        let shared_scroll_to_item = self.scroll_handle.as_mut().and_then(|handle| {
            let mut handle = handle.0.borrow_mut();
            handle.last_item_size = Some(HListItemSize {
                item: padded_bounds.size,
                contents: content_size,
            });
            handle.deferred_scroll_to_item.take()
        });

        self.interactivity.prepaint(
            global_id,
            inspector_id,
            bounds,
            content_size,
            window,
            cx,
            |_style, mut logical_scroll_offset, hitbox, window, cx| {
                if self.item_count > 0 {
                    let is_scrolled = !logical_scroll_offset.x.is_zero();
                    let max_scroll_offset = padded_bounds.size.width - content_width;
                    if is_scrolled && logical_scroll_offset.x < max_scroll_offset {
                        shared_scroll_offset.borrow_mut().x = max_scroll_offset;
                    }

                    let applied_deferred_scroll = shared_scroll_to_item.is_some();

                    // --- deferred scroll-to-item ------------------------------------------------
                    if let Some(HListDeferredScroll {
                        mut item_index,
                        mut strategy,
                        offset,
                        scroll_strict,
                    }) = shared_scroll_to_item
                    {
                        let list_width = padded_bounds.size.width;
                        let mut updated_scroll_offset = shared_scroll_offset.borrow_mut();
                        let item_left: Pixels = self.item_widths[..item_index].iter().fold(Pixels::ZERO, |a, &w| a + w);
                        let item_width = self.item_widths[item_index];
                        let item_right = item_left + item_width;
                        let scroll_left = -updated_scroll_offset.x;
                        let offset_pixels: Pixels = self.item_widths[..offset]
                            .iter()
                            .fold(Pixels::ZERO, |a, &w| a + w);

                        let is_before = item_left < scroll_left + offset_pixels;
                        let is_after = item_right > scroll_left + list_width;

                        if scroll_strict || is_before || is_after {
                            if strategy == HListScrollStrategy::Nearest {
                                if is_before {
                                    strategy = HListScrollStrategy::Start;
                                } else if is_after {
                                    strategy = HListScrollStrategy::End;
                                }
                            }

                            let max_scroll_offset =
                                (content_width - list_width).max(Pixels::ZERO);
                            match strategy {
                                HListScrollStrategy::Start => {
                                    updated_scroll_offset.x = -(item_left - offset_pixels)
                                        .clamp(Pixels::ZERO, max_scroll_offset);
                                }
                                HListScrollStrategy::Center => {
                                    let item_center = item_left + item_width / 2.0;
                                    let viewport_width = list_width - offset_pixels;
                                    let viewport_center = offset_pixels + viewport_width / 2.0;
                                    let target_scroll_left = item_center - viewport_center;
                                    updated_scroll_offset.x = -target_scroll_left
                                        .clamp(Pixels::ZERO, max_scroll_offset);
                                }
                                HListScrollStrategy::End => {
                                    updated_scroll_offset.x = -(item_right - list_width)
                                        .clamp(Pixels::ZERO, max_scroll_offset);
                                }
                                HListScrollStrategy::Nearest => {}
                            }
                        }
                        logical_scroll_offset = *updated_scroll_offset;
                    }

                    // --- smooth scroll ---------------------------------------------------------
                    let mut visual_scroll_offset = logical_scroll_offset;
                    if let Some(scroll_handle) = &self.scroll_handle {
                        let mut scroll_state = scroll_handle.0.borrow_mut();
                        scroll_state
                            .smooth_scroll
                            .set_target(logical_scroll_offset.x);
                        if applied_deferred_scroll {
                            scroll_state.smooth_scroll.visual_offset = logical_scroll_offset.x;
                            scroll_state.smooth_scroll.target_offset = logical_scroll_offset.x;
                            scroll_state.smooth_scroll.animating = false;
                        } else if scroll_state.smooth_scroll.update() {
                            window.refresh();
                        }
                        visual_scroll_offset.x = scroll_state.smooth_scroll.current();
                    }

                    // --- find visible range via cumulative offsets -------------------------------
                    let scroll_left = -visual_scroll_offset.x;
                    let view_right = scroll_left + padded_bounds.size.width;

                    let mut cum = Pixels::ZERO;
                    let first_visible = self
                        .item_widths
                        .iter()
                        .position(|w| {
                            let end = cum + *w;
                            let visible = end > scroll_left;
                            cum = end;
                            visible
                        })
                        .unwrap_or(0);

                    // Reset cum and find last visible with overscan.
                    cum = self.item_widths[..first_visible].iter().fold(Pixels::ZERO, |a, &w| a + w);
                    let mut last_visible = first_visible;
                    let overscan_target = view_right + padded_bounds.size.width * 0.5;
                    for w in &self.item_widths[first_visible..] {
                        cum += *w;
                        last_visible += 1;
                        if cum > overscan_target {
                            break;
                        }
                    }
                    let visible_range = cmp::min(last_visible, self.item_count);

                    // --- render visible items ----------------------------------------------------
                    let visible_count = visible_range - first_visible;
                    if visible_count > 0 {
                        let items = (self.render_items)(
                            first_visible..visible_range,
                            window,
                            cx,
                        );

                        let content_mask = ContentMask { bounds };
                        let base_x = padded_bounds.origin.x + visual_scroll_offset.x;
                        let base_y = padded_bounds.origin.y + visual_scroll_offset.y;

                        window.with_content_mask(Some(content_mask), |window| {
                            let mut cum_offset = self.item_widths[..first_visible]
                                .iter()
                                .fold(Pixels::ZERO, |a, &w| a + w);
                            for mut item in items {
                                let item_w = self.item_widths[first_visible + frame_state.items.len()];
                                let item_origin = point(base_x + cum_offset, base_y);
                                let available_space = size(
                                    AvailableSpace::Definite(item_w),
                                    AvailableSpace::Definite(padded_bounds.size.height),
                                );
                                item.layout_as_root(available_space, window, cx);
                                item.prepaint_at(item_origin, window, cx);
                                frame_state.items.push(item);
                                cum_offset += item_w;
                            }
                        });
                    }
                }

                hitbox
            },
        )
    }

    fn paint(
        &mut self,
        global_id: Option<&GlobalElementId>,
        inspector_id: Option<&InspectorElementId>,
        bounds: Bounds<Pixels>,
        request_layout: &mut Self::RequestLayoutState,
        hitbox: &mut Option<Hitbox>,
        window: &mut Window,
        cx: &mut App,
    ) {
        self.interactivity.paint(
            global_id,
            inspector_id,
            bounds,
            hitbox.as_ref(),
            window,
            cx,
            |_, window, cx| {
                for item in &mut request_layout.items {
                    item.paint(window, cx);
                }
            },
        )
    }
}

impl IntoElement for HList {
    type Element = Self;

    fn into_element(self) -> Self::Element {
        self
    }
}

impl HList {
    /// Measure a single item (used before per-item widths are populated).
    fn measure_one(
        &self,
        list_height: Option<Pixels>,
        window: &mut Window,
        cx: &mut App,
    ) -> Size<Pixels> {
        if self.item_count == 0 {
            return Size::default();
        }
        let mut items = (self.render_items)(0..1, window, cx);
        let Some(mut item) = items.pop() else {
            return Size::default();
        };
        let available = size(
            AvailableSpace::MinContent,
            list_height.map_or(AvailableSpace::MinContent, AvailableSpace::Definite),
        );
        item.layout_as_root(available, window, cx)
    }

    /// Track and render scroll state of this list with reference to the given scroll handle.
    pub fn track_scroll(mut self, handle: &HListScrollHandle) -> Self {
        self.interactivity.tracked_scroll_handle = Some(handle.0.borrow().base_handle.clone());
        self.scroll_handle = Some(handle.clone());
        self
    }

    /// Sets the sizing behavior for the list height.
    pub fn with_sizing_behavior(mut self, behavior: ListSizingBehavior) -> Self {
        self.sizing_behavior = behavior;
        self
    }

    /// Sets the horizontal sizing behavior.
    pub fn with_horizontal_sizing_behavior(
        mut self,
        behavior: ListHorizontalSizingBehavior,
    ) -> Self {
        self.horizontal_sizing_behavior = behavior;
        match behavior {
            ListHorizontalSizingBehavior::FitList => {
                self.interactivity.base_style.overflow.x = None;
            }
            ListHorizontalSizingBehavior::Unconstrained => {
                self.interactivity.base_style.overflow.x = Some(Overflow::Scroll);
            }
        }
        self
    }
}

impl InteractiveElement for HList {
    fn interactivity(&mut self) -> &mut Interactivity {
        &mut self.interactivity
    }
}
