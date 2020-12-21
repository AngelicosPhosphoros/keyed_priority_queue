use std::borrow::Borrow;
use std::hash::{BuildHasher, Hash};

use indexmap::map::{IndexMap, OccupiedEntry as IMOccupiedEntry, VacantEntry as IMVacantEntry};

use crate::editable_binary_heap::HeapIndex;

/// Wrapper around possible outer vec index
/// Used to avoid mux up with heap index
/// And to make sure that `Mediator` indexed only with MediatorIndex
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub(crate) struct MediatorIndex(pub(crate) usize);

/// This is wrapper over over indexmap that uses `MediatorIndex` as index.
/// Also it centralized checking for panics
#[derive(Clone, Debug)]
pub(crate) struct Mediator<TKey: Hash + Eq, S: BuildHasher> {
    map: IndexMap<TKey, HeapIndex, S>,
}

#[inline(always)]
fn with_copied_heap_index<'a, T>((k, &i): (&'a T, &HeapIndex)) -> (&'a T, HeapIndex) {
    (k, i)
}

pub(crate) struct VacantEntry<'a, TKey: 'a + Hash + Eq, S: BuildHasher> {
    internal: IMVacantEntry<'a, TKey, HeapIndex>,
    // look `insert` definition for this
    map: *mut Mediator<TKey, S>,
}
pub(crate) struct OccupiedEntry<'a, TKey: 'a + Hash + Eq, S: BuildHasher> {
    internal: IMOccupiedEntry<'a, TKey, HeapIndex>,
    // look `insert` definition for this
    map: *mut Mediator<TKey, S>,
}

pub(crate) enum MediatorEntry<'a, TKey: 'a + Hash + Eq, S: BuildHasher> {
    Vacant(VacantEntry<'a, TKey, S>),
    Occupied(OccupiedEntry<'a, TKey, S>),
}

impl<TKey, S> Mediator<TKey, S>
where
    TKey: Hash + Eq,
    S: BuildHasher,
{
    pub(crate) fn with_capacity_and_hasher(capacity: usize, hasher: S) -> Self {
        Self {
            map: IndexMap::with_capacity_and_hasher(capacity, hasher),
        }
    }

    #[inline(always)]
    pub(crate) fn reserve(&mut self, additional: usize) {
        self.map.reserve(additional)
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.map.len()
    }

    #[inline(always)]
    pub(crate) fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    #[inline(always)]
    pub(crate) fn clear(&mut self) {
        self.map.clear()
    }

    #[inline(always)]
    pub(crate) fn get_index(&self, MediatorIndex(position): MediatorIndex) -> (&TKey, HeapIndex) {
        self.map
            .get_index(position)
            .map(with_copied_heap_index)
            .expect("All mediator indexes must be valid")
    }

    #[inline(always)]
    pub(crate) fn entry(&mut self, key: TKey) -> MediatorEntry<TKey, S> {
        // Pointer dereferenced only after internal entry dropped
        // This unsafe pointer dark magic is required because you cannot handle
        // enum that keep either Entry or Map inside:
        // Example: https://play.rust-lang.org/?version=stable&mode=debug&edition=2018&gist=aee5555275572e385350a786127f91ed
        //
        // We need to acquire mutable reference to Mediator in KeyedPriorityQueue Entry API implementation to keep consistency
        // but keeping multiple mutable references in entry disallowed by borrow checker.
        // We keep IndexMap entry in Mediator entry and pointer to Mediator, and allow use second one only after dropping first.
        // This references used in 3 places:
        // 1. keyed_priority_queue::OccupiedEntry::set_priority
        // 2. keyed_priority_queue::OccupiedEntry::remove
        // 3. keyed_priority_queue::VacantEntry::set_priority
        //
        // Also after stabilisation of `polonius` (https://rust-lang.github.io/polonius/) we would be able
        // to remove this pointer hack from OccupiedEntry and keep reference to Mediator and index in it instead.
        let map = self as *mut _;
        match self.map.entry(key) {
            indexmap::map::Entry::Occupied(internal) => {
                MediatorEntry::Occupied(OccupiedEntry { internal, map })
            }
            indexmap::map::Entry::Vacant(internal) => {
                MediatorEntry::Vacant(VacantEntry { internal, map })
            }
        }
    }

    #[inline(always)]
    pub(crate) fn get<Q>(&self, key: &Q) -> Option<HeapIndex>
    where
        TKey: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.get(key).copied()
    }

    #[inline(always)]
    pub(crate) fn get_full<'a, Q>(&'a self, key: &Q) -> Option<(MediatorIndex, &'a TKey, HeapIndex)>
    where
        TKey: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map
            .get_full(key)
            .map(|(idx, key, &val)| (MediatorIndex(idx), key, val))
    }

    #[inline(always)]
    pub(crate) fn swap_remove_index(
        &mut self,
        MediatorIndex(index): MediatorIndex,
    ) -> (TKey, HeapIndex) {
        self.map
            .swap_remove_index(index)
            .expect("All mediator indexes must be valid")
    }

    #[inline(always)]
    pub(crate) fn get_index_mut(&mut self, MediatorIndex(index): MediatorIndex) -> &mut HeapIndex {
        self.map
            .get_index_mut(index)
            .expect("All mediator indexes must be valid")
            .1
    }

    #[cfg(test)]
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&TKey, HeapIndex)> {
        self.map.iter().map(with_copied_heap_index)
    }
}

impl<'a, TKey: 'a + Hash + Eq, S: BuildHasher> VacantEntry<'a, TKey, S> {
    // Safety: make sure that nobody uses original mutable reference to mediator
    // when returned pointer are used
    // And the pointer never available longer than `Mediator` instance which created the VacantEntry
    // See `Mediator::entry` and KeyedPriorityQueue's `remove` and `set_priority` entry methods.
    #[inline]
    pub(crate) unsafe fn insert(
        self,
        value: HeapIndex,
    ) -> (&'a mut Mediator<TKey, S>, MediatorIndex) {
        let map = self.map;
        let result_index = MediatorIndex(self.internal.index());
        {
            self.internal.insert(value);
        }
        let mediator = map.as_mut().expect("Validated in entry method");
        (mediator, result_index)
    }

    #[inline]
    pub(crate) fn get_key(&self) -> &TKey {
        self.internal.key()
    }

    #[inline]
    pub(crate) fn index(&self) -> MediatorIndex {
        MediatorIndex(self.internal.index())
    }
}

impl<'a, TKey: 'a + Hash + Eq, S: BuildHasher> OccupiedEntry<'a, TKey, S> {
    #[inline]
    pub(crate) fn get_heap_idx(&self) -> HeapIndex {
        *self.internal.get()
    }

    #[inline]
    pub(crate) fn get_key(&self) -> &TKey {
        self.internal.key()
    }

    // Safety: make sure that nobody uses original mutable reference to mediator
    // when returned reference are used
    // And the pointer never available longer than `Mediator` instance which created the VacantEntry
    // See `Mediator::entry` and KeyedPriorityQueue's `set_priority` entry method.
    #[inline]
    pub(crate) unsafe fn transform_to_map(self) -> &'a mut Mediator<TKey, S> {
        let map = self.map;
        std::mem::drop(self);
        let mediator = map.as_mut().expect("Validated in entry method");
        mediator
    }
}
