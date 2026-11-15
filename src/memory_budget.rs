use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::decoder::DecodingError;

#[derive(Clone, Debug)]
pub(crate) struct MemoryBudget(Arc<State>);

#[derive(Debug)]
struct State {
    limit: AtomicUsize,
    used: AtomicUsize,
}

#[derive(Debug)]
pub(crate) struct Reservation {
    budget: MemoryBudget,
    bytes: usize,
}

#[derive(Debug)]
pub(crate) struct MemoryAccount {
    budget: MemoryBudget,
    used: usize,
}

impl MemoryBudget {
    pub(crate) fn new(limit: usize) -> Self {
        Self(Arc::new(State {
            limit: AtomicUsize::new(limit),
            used: AtomicUsize::new(0),
        }))
    }

    pub(crate) fn set_limit(&self, limit: usize) {
        self.0.limit.store(limit, Ordering::Relaxed);
    }

    pub(crate) fn reserve(&self, bytes: usize) -> Result<Reservation, DecodingError> {
        self.charge(bytes)?;
        Ok(Reservation {
            budget: self.clone(),
            bytes,
        })
    }

    fn charge(&self, bytes: usize) -> Result<(), DecodingError> {
        let mut used = self.0.used.load(Ordering::Relaxed);
        loop {
            let new_used = used
                .checked_add(bytes)
                .filter(|&new_used| new_used <= self.0.limit.load(Ordering::Relaxed))
                .ok_or(DecodingError::MemoryLimitExceeded)?;
            match self.0.used.compare_exchange_weak(
                used,
                new_used,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(()),
                Err(current) => used = current,
            }
        }
    }

    pub(crate) fn reserve_array<T>(&self, capacity: usize) -> Result<Reservation, DecodingError> {
        let bytes = capacity
            .checked_mul(std::mem::size_of::<T>())
            .ok_or(DecodingError::MemoryLimitExceeded)?;
        self.reserve(bytes)
    }
}

impl MemoryAccount {
    pub(crate) fn new(budget: MemoryBudget) -> Self {
        Self { budget, used: 0 }
    }

    pub(crate) fn allocate(&mut self, bytes: usize) -> Result<(), DecodingError> {
        let used = self
            .used
            .checked_add(bytes)
            .ok_or(DecodingError::MemoryLimitExceeded)?;
        self.budget.charge(bytes)?;
        self.used = used;
        Ok(())
    }

    pub(crate) fn release(&mut self, bytes: usize) {
        self.used -= bytes;
        self.budget.0.used.fetch_sub(bytes, Ordering::Relaxed);
    }

    pub(crate) fn used(&self) -> usize {
        self.used
    }
}

impl Drop for MemoryAccount {
    fn drop(&mut self) {
        self.budget.0.used.fetch_sub(self.used, Ordering::Relaxed);
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.budget.0.used.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

pub(crate) fn grow<T>(
    vec: &mut Vec<T>,
    capacity: usize,
    budget: &MemoryBudget,
    reservation: &mut Option<Reservation>,
) -> Result<(), DecodingError> {
    if capacity > vec.capacity() {
        let new_reservation = budget.reserve_array::<T>(capacity)?;
        vec.try_reserve_exact(capacity - vec.len())
            .map_err(|_| DecodingError::MemoryLimitExceeded)?;
        *reservation = Some(new_reservation);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::MemoryBudget;

    #[test]
    fn reservations_share_limit_and_release_on_drop() {
        let budget = MemoryBudget::new(10);
        let first = budget.reserve(6).expect("first reservation fits");
        assert!(budget.reserve(5).is_err());
        let second = budget.reserve(4).expect("second reservation fits");
        drop(first);
        assert!(budget.reserve(6).is_ok());
        drop(second);
        assert!(budget.reserve(10).is_ok());
    }

    #[test]
    fn overflow_and_lowered_limit_reject_reservations() {
        let budget = MemoryBudget::new(usize::MAX);
        let _reservation = budget.reserve(1).expect("reservation fits");
        assert!(budget.reserve(usize::MAX).is_err());
        assert!(budget.reserve_array::<u16>(usize::MAX).is_err());
        budget.set_limit(0);
        assert!(budget.reserve(0).is_err());
    }

    #[test]
    fn growth_charges_old_and_new_capacities_together() {
        let budget = MemoryBudget::new(11);
        let mut data = Vec::<u8>::new();
        let mut reservation = None;
        super::grow(&mut data, 4, &budget, &mut reservation).expect("first growth fits");
        assert!(super::grow(&mut data, 8, &budget, &mut reservation).is_err());
        budget.set_limit(12);
        super::grow(&mut data, 8, &budget, &mut reservation).expect("second growth fits");
        assert!(budget.reserve(4).is_ok());
    }

    #[test]
    fn account_releases_temporary_allocations_and_drops_remaining() {
        let budget = MemoryBudget::new(8);
        let mut account = super::MemoryAccount::new(budget.clone());
        account.allocate(6).expect("allocation fits");
        assert!(account.allocate(3).is_err());
        account.release(2);
        assert!(budget.reserve(4).is_ok());
        drop(account);
        assert!(budget.reserve(8).is_ok());
    }
}
