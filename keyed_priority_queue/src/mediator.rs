use crate::editable_binary_heap::HeapIndex;
use indexmap::map::{IndexMap, OccupiedEntry as IMOccupiedEntry, VacantEntry as IMVacantEntry};
use std::borrow::Borrow;
use std::hash::Hash;

/// Wrapper around possible outer vec index
/// Used to avoid mux up with heap index
/// And to make sure that `Mediator` indexed only with MediatorIndex
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub(crate) struct MediatorIndex(pub(crate) usize);

/// This is wrapper over over indexmap that uses `MediatorIndex` as index.
/// Also it centralized checking for panics
#[derive(Clone, Debug)]
pub(crate) struct Mediator<TKey: Hash + Eq> {
    map: IndexMap<TKey, HeapIndex>,
}

#[inline(always)]
fn with_copied_heap_index<'a, T>((k, &i): (&'a T, &HeapIndex)) -> (&'a T, HeapIndex) {
    (k, i)
}

pub(crate) struct VacantEntry<'a, TKey: 'a + Hash + Eq>(IMVacantEntry<'a, TKey, HeapIndex>);
pub(crate) struct OccupiedEntry<'a, TKey: 'a + Hash + Eq>(IMOccupiedEntry<'a, TKey, HeapIndex>);
pub(crate) enum MediatorEntry<'a, TKey: 'a + Hash + Eq> {
    Vacant(VacantEntry<'a, TKey>),
    Occupied(OccupiedEntry<'a, TKey>),
}

impl<TKey> Mediator<TKey>
where
    TKey: Hash + Eq,
{
    #[inline(always)]
    pub(crate) fn new() -> Self {
        Self {
            map: IndexMap::new(),
        }
    }

    #[inline(always)]
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            map: IndexMap::with_capacity(capacity),
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
    pub(crate) fn entry(&mut self, key: TKey) -> MediatorEntry<TKey> {
        match self.map.entry(key) {
            indexmap::map::Entry::Occupied(v) => MediatorEntry::Occupied(OccupiedEntry(v)),
            indexmap::map::Entry::Vacant(v) => MediatorEntry::Vacant(VacantEntry(v)),
        }
    }

    #[inline(always)]
    pub(crate) fn get<Q>(&self, key: &Q) -> Option<HeapIndex>
    where
        TKey: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.get(key).map(|&x| x)
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

impl<'a, TKey: 'a + Hash + Eq> VacantEntry<'a, TKey> {
    #[inline(always)]
    pub(crate) fn insert(self, value: HeapIndex) {
        self.0.insert(value);
    }

    #[inline(always)]
    pub(crate) fn index(&self) -> MediatorIndex {
        MediatorIndex(self.0.index())
    }
}

impl<'a, TKey: 'a + Hash + Eq> OccupiedEntry<'a, TKey> {
    #[inline(always)]
    pub(crate) fn index(&self) -> MediatorIndex {
        MediatorIndex(self.0.index())
    }

    #[inline(always)]
    pub(crate) fn get(&self) -> HeapIndex {
        *self.0.get()
    }
}
