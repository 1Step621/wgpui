//! A cheap "the UI thread is starting a frame" signal.
//!
//! An external render thread (the Helio viewport) uses it to begin its own
//! frame when the UI begins theirs, so its CPU work and queue submission hide
//! under the UI's layout/prepaint/paint instead of landing in the UI's present
//! phase, where both fight for the shared queue.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

static SEQ: AtomicU64 = AtomicU64::new(0);
static LOCK: Mutex<()> = Mutex::new(());
static WAKE: Condvar = Condvar::new();

/// Called by the window as it begins drawing a frame.
pub fn mark_ui_frame_begin() {
    SEQ.fetch_add(1, Ordering::Release);
    // Taking the lock orders the increment against a waiter's re-check, so a
    // begin can never slip between its check and its sleep.
    drop(LOCK.lock());
    WAKE.notify_all();
}

/// How many UI frames have begun so far.
pub fn ui_frame_seq() -> u64 {
    SEQ.load(Ordering::Acquire)
}

/// Block until a UI frame beyond `seen` has begun or `timeout` elapses.
/// Returns the sequence number at wake-up.
pub fn wait_for_ui_frame_after(seen: u64, timeout: Duration) -> u64 {
    let deadline = Instant::now() + timeout;
    let mut guard = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    loop {
        let now = ui_frame_seq();
        if now != seen {
            return now;
        }
        let Some(left) = deadline.checked_duration_since(Instant::now()) else {
            return now;
        };
        guard = match WAKE.wait_timeout(guard, left) {
            Ok((g, _)) => g,
            Err(e) => e.into_inner().0,
        };
    }
}
