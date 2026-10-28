//! Locks that survive a panic.
//!
//! A std `Mutex`/`RwLock` is "poisoned" when a thread panics while holding it, and every later `lock()` returns an
//! error. Python has no such state: an exception inside `with lock:` releases the lock and the next caller goes on.
//! The hub keeps Python's behaviour: one panic (a bug in one request, one tick, one node message) must not lock every
//! later caller out of a service. Every std lock in the hub is taken through these helpers, which hand back the guard
//! of a poisoned lock as well (the data a panicking holder left is still the service's best state; each service keeps
//! its invariants per operation, as the Python code does).

use std::sync::{Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// `Mutex::lock` that never fails: a poisoned lock yields its guard all the same.
pub trait Recover<T: ?Sized> {
    /// The exclusive guard, poisoned or not.
    fn guard(&self) -> MutexGuard<'_, T>;
}

impl<T: ?Sized> Recover<T> for Mutex<T> {
    fn guard(&self) -> MutexGuard<'_, T> {
        self.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The two guards of a `RwLock`, poisoned or not.
pub trait RecoverRw<T: ?Sized> {
    fn read_guard(&self) -> RwLockReadGuard<'_, T>;
    fn write_guard(&self) -> RwLockWriteGuard<'_, T>;
}

impl<T: ?Sized> RecoverRw<T> for RwLock<T> {
    fn read_guard(&self) -> RwLockReadGuard<'_, T> {
        self.read().unwrap_or_else(PoisonError::into_inner)
    }
    fn write_guard(&self) -> RwLockWriteGuard<'_, T> {
        self.write().unwrap_or_else(PoisonError::into_inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn test_a_poisoned_mutex_still_works() {
        let m = Arc::new(Mutex::new(1));
        let m2 = Arc::clone(&m);
        let _ = std::thread::spawn(move || {
            let mut g = m2.guard();
            *g = 2;
            #[allow(clippy::panic)]
            {
                panic!("a holder panics");
            }
        })
        .join();
        assert!(m.is_poisoned());
        assert_eq!(*m.guard(), 2);
        *m.guard() = 3;
        assert_eq!(*m.guard(), 3);
    }

    #[test]
    fn test_a_poisoned_rwlock_still_works() {
        let l = Arc::new(RwLock::new(vec![1]));
        let l2 = Arc::clone(&l);
        let _ = std::thread::spawn(move || {
            let _g = l2.write_guard();
            #[allow(clippy::panic)]
            {
                panic!("a writer panics");
            }
        })
        .join();
        assert!(l.is_poisoned());
        l.write_guard().push(2);
        assert_eq!(*l.read_guard(), vec![1, 2]);
    }
}
