use crate::{PlatformDispatcher, Priority, PriorityQueueSender, RealtimePriority, RunnableVariant};
use priority_threadpool::ThreadPool;
use std::thread::ThreadId;
use winit::event_loop::EventLoopProxy;

pub enum CrossEvent {
    WakeUp,
    SurfacePresent(winit::window::WindowId),
    SingleInstanceActivated,
    /// Sent by CrossWindow when GPUI programmatically removes a window,
    /// so the platform layer removes it from AppState.windows and the OS window is destroyed.
    CloseWindow(winit::window::WindowId),
}

pub struct Dispatcher {
    main_thread_id: ThreadId,
    main_tx: PriorityQueueSender<RunnableVariant>,
    threadpool: ThreadPool<Priority>,
    proxy: EventLoopProxy<CrossEvent>,
}

impl Dispatcher {
    pub fn new(
        main_tx: PriorityQueueSender<RunnableVariant>,
        proxy: EventLoopProxy<CrossEvent>,
    ) -> Self {
        Self {
            main_thread_id: std::thread::current().id(),
            main_tx,
            threadpool: ThreadPool::new(num_cpus::get() * 8),
            proxy,
        }
    }
}

impl PlatformDispatcher for Dispatcher {
    fn is_main_thread(&self) -> bool {
        std::thread::current().id() == self.main_thread_id
    }

    fn dispatch(
        &self,
        runnable: RunnableVariant,
        _label: Option<crate::TaskLabel>,
        priority: Priority,
    ) {
        match runnable {
            RunnableVariant::Meta(runnable) => self.threadpool.queue(&priority, runnable),
            RunnableVariant::Compat(runnable) => self.threadpool.queue(&priority, runnable),
        }
    }

    fn dispatch_on_main_thread(&self, runnable: RunnableVariant, priority: Priority) {
        match self.main_tx.send(priority, runnable) {
            Ok(_) => {
                let _ = self.proxy.send_event(CrossEvent::WakeUp);
            }
            Err(runnable) => {
                std::mem::forget(runnable);
            }
        }
    }

    fn dispatch_after(&self, duration: std::time::Duration, runnable: RunnableVariant) {
        match runnable {
            RunnableVariant::Meta(runnable) => {
                self.threadpool
                    .queue_delayed(&Priority::Low, duration, runnable);
            }
            RunnableVariant::Compat(runnable) => {
                self.threadpool
                    .queue_delayed(&Priority::Low, duration, runnable);
            }
        }
    }

    fn spawn_realtime(&self, _priority: RealtimePriority, f: Box<dyn FnOnce() + Send>) {
        // TODO(mdeand): There's a crate (thread-priority) that implements thread
        // TODO(mdeand): priorities, but I don't want to add it right now.

        std::thread::spawn(move || {
            f();
        });
    }
}

impl priority_threadpool::Priority for Priority {
    const COUNT: usize = 3;

    fn index(&self) -> usize {
        match self {
            Priority::High => 0,
            Priority::Medium => 1,
            Priority::Low => 2,
            _ => unreachable!(),
        }
    }
}
