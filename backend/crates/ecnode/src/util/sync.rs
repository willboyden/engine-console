//! Locks that survive a panic.
//!
//! A thread that panics while it holds a `Mutex` poisons it; `lock().unwrap()` would then panic in every later
//! user, so one bad round of a loop or one failed request would take a whole part of the node down with it. Every
//! lock and condition-variable wait in the node goes through these helpers, which take the guard out of the poison
//! error: the data a lock protects in this node is always left consistent between statements (counters, caches,
//! queues), so carrying on with it is safe, and far better than a dead node.

use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

/// `m.lock()`, recovering the guard of a poisoned mutex.
pub fn lock<T: ?Sized>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// `Mutex::into_inner`, recovering the value of a poisoned mutex.
pub fn into_inner<T>(m: Mutex<T>) -> T {
    m.into_inner().unwrap_or_else(PoisonError::into_inner)
}

/// `cv.wait(g)`, recovering the guard when the mutex was poisoned meanwhile.
pub fn wait<'a, T>(cv: &Condvar, g: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
    cv.wait(g).unwrap_or_else(PoisonError::into_inner)
}

/// `cv.wait_timeout(g, d)`, recovering the guard when the mutex was poisoned meanwhile.
pub fn wait_timeout<'a, T>(cv: &Condvar, g: MutexGuard<'a, T>, d: Duration) -> MutexGuard<'a, T> {
    cv.wait_timeout(g, d)
        .map_or_else(|e| e.into_inner().0, |(g, _)| g)
}

/// `cv.wait_timeout_while(g, d, cond)`, recovering the guard when the mutex was poisoned meanwhile.
pub fn wait_timeout_while<'a, T>(
    cv: &Condvar,
    g: MutexGuard<'a, T>,
    d: Duration,
    cond: impl FnMut(&mut T) -> bool,
) -> MutexGuard<'a, T> {
    cv.wait_timeout_while(g, d, cond)
        .map_or_else(|e| e.into_inner().0, |(g, _)| g)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn poisoned() -> Arc<Mutex<Vec<i32>>> {
        let m = Arc::new(Mutex::new(vec![1]));
        let m2 = Arc::clone(&m);
        let _ = std::thread::spawn(move || {
            let mut g = lock(&m2);
            g.push(2);
            #[allow(clippy::panic)]
            {
                panic!("a panic while the lock is held");
            }
        })
        .join();
        assert!(m.is_poisoned());
        m
    }

    #[test]
    fn a_poisoned_lock_still_works() {
        let m = poisoned();
        lock(&m).push(3);
        assert_eq!(*lock(&m), vec![1, 2, 3]);
        let cv = Condvar::new();
        let g = wait_timeout(&cv, lock(&m), Duration::from_millis(5));
        assert_eq!(g.len(), 3);
        let g = wait_timeout_while(&cv, g, Duration::from_millis(5), |v| v.len() < 10);
        assert_eq!(g.len(), 3);
        drop(g);
        let inner = Arc::try_unwrap(m).map(into_inner).unwrap_or_default();
        assert_eq!(inner, vec![1, 2, 3]);
    }
}
