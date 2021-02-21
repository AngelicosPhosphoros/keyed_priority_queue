use std::cmp::{Ord, Ordering};
use std::fmt::Debug;
use std::hash::BuildHasher;
use std::vec::Vec;

use crate::mediator::MediatorIndex;

/// Wrapper around usize that can be used only as index of `BinaryHeap`
/// Mostly needed to statically check that
/// Heap is not indexed by any other collection index
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub(crate) struct HeapIndex(usize);

#[derive(Copy, Clone)]
struct HeapEntry<TPriority> {
    outer_pos: MediatorIndex,
    priority: TPriority,
}

impl<TPriority> HeapEntry<TPriority> {
    // For usings as HeapEntry::as_pair instead of closures in map

    #[inline(always)]
    fn conv_pair(self) -> (MediatorIndex, TPriority) {
        (self.outer_pos, self.priority)
    }

    #[inline(always)]
    fn to_pair_ref(&self) -> (MediatorIndex, &TPriority) {
        (self.outer_pos, &self.priority)
    }

    #[inline(always)]
    fn to_outer(&self) -> MediatorIndex {
        self.outer_pos
    }
}

#[derive(Clone)]
pub(crate) struct BinaryHeap<TPriority>
where
    TPriority: Ord,
{
    data: Vec<HeapEntry<TPriority>>,
}

impl<TPriority: Ord> BinaryHeap<TPriority> {
    #[inline]
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.data.reserve(additional);
    }

    /// Puts outer index and priority in queue
    /// outer_pos is assumed to be unique but not validated
    /// because validation too expensive
    /// Calls change_handler for every move of old values
    #[inline]
    pub(crate) fn push<TChangeHandler: std::ops::FnMut(MediatorIndex, HeapIndex)>(
        &mut self,
        outer_pos: MediatorIndex,
        priority: TPriority,
        mut change_handler: TChangeHandler,
    ) {
        self.data.push(HeapEntry {
            outer_pos,
            priority,
        });
        self.heapify_up(HeapIndex(self.data.len() - 1), &mut change_handler);
    }

    /// Removes item at position and returns it
    /// Time complexity - O(log n) swaps and change_handler calls
    pub(crate) fn remove<TChangeHandler: std::ops::FnMut(MediatorIndex, HeapIndex)>(
        &mut self,
        position: HeapIndex,
        mut change_handler: TChangeHandler,
    ) -> Option<(MediatorIndex, TPriority)> {
        if position >= self.len() {
            return None;
        }
        if position.0 + 1 == self.len().0 {
            let result = self.data.pop().expect("At least 1 item");
            return Some(result.conv_pair());
        }

        let result = self.data.swap_remove(position.0);
        self.heapify_down(position, &mut change_handler);
        if position.0 > 0 {
            self.heapify_up(position, &mut change_handler);
        }
        Some(result.conv_pair())
    }

    #[inline]
    pub(crate) fn look_into(&self, position: HeapIndex) -> Option<(MediatorIndex, &TPriority)> {
        self.data.get(position.0).map(HeapEntry::to_pair_ref)
    }

    /// Changes priority of queue item
    /// Returns old priority
    pub(crate) fn change_priority<TChangeHandler: std::ops::FnMut(MediatorIndex, HeapIndex)>(
        &mut self,
        position: HeapIndex,
        updated: TPriority,
        mut change_handler: TChangeHandler,
    ) -> TPriority {
        debug_assert!(
            position < self.len(),
            "Out of index during changing priority"
        );

        let old = std::mem::replace(&mut self.data[position.0].priority, updated);
        match old.cmp(&self.data[position.0].priority) {
            Ordering::Less => {
                self.heapify_up(position, &mut change_handler);
            }
            Ordering::Equal => {}
            Ordering::Greater => {
                self.heapify_down(position, &mut change_handler);
            }
        }
        old
    }

    // Changes outer index for element and return old index
    pub(crate) fn change_outer_pos(
        &mut self,
        outer_pos: MediatorIndex,
        position: HeapIndex,
    ) -> MediatorIndex {
        debug_assert!(position < self.len(), "Out of index during changing key");

        let old_pos = self.data[position.0].outer_pos;
        self.data[position.0].outer_pos = outer_pos;
        old_pos
    }

    #[inline]
    pub(crate) fn most_prioritized_idx(&self) -> Option<(MediatorIndex, HeapIndex)> {
        self.data.get(0).map(|x| (x.outer_pos, HeapIndex(0)))
    }

    #[inline]
    pub(crate) fn len(&self) -> HeapIndex {
        HeapIndex(self.data.len())
    }

    #[inline]
    pub(crate) fn usize_len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    pub(crate) fn clear(&mut self) {
        self.data.clear()
    }

    #[inline]
    pub(crate) fn iter(&self) -> BinaryHeapIterator<TPriority> {
        BinaryHeapIterator {
            inner: self.data.iter(),
        }
    }

    pub(crate) fn produce_from_iter_hash<TKey, TIter, S>(
        iter: TIter,
    ) -> (Self, crate::mediator::Mediator<TKey, S>)
    where
        TKey: std::hash::Hash + Eq,
        TIter: IntoIterator<Item = (TKey, TPriority)>,
        S: BuildHasher + Default,
    {
        use crate::mediator::{Mediator, MediatorEntry};

        let iter = iter.into_iter();
        let (min_size, _) = iter.size_hint();

        let mut heap_base: Vec<HeapEntry<TPriority>> = Vec::with_capacity(min_size);
        let mut map: Mediator<TKey, S> = Mediator::with_capacity_and_hasher(min_size, S::default());

        for (key, priority) in iter {
            match map.entry(key) {
                MediatorEntry::Vacant(entry) => {
                    let outer_pos = entry.index();
                    unsafe {
                        // Safety: resulting reference never used
                        entry.insert(HeapIndex(heap_base.len()));
                    }
                    heap_base.push(HeapEntry {
                        outer_pos,
                        priority,
                    });
                }
                MediatorEntry::Occupied(entry) => {
                    let HeapIndex(heap_pos) = entry.get_heap_idx();
                    heap_base[heap_pos].priority = priority;
                }
            }
        }

        let heapify_start = std::cmp::min(heap_base.len() / 2 + 2, heap_base.len());
        let mut heap = BinaryHeap { data: heap_base };
        for pos in (0..heapify_start).rev().map(HeapIndex) {
            heap.heapify_down(pos, &mut |_, _| {});
        }

        for (i, pos) in heap.data.iter().map(HeapEntry::to_outer).enumerate() {
            let heap_idx = map.get_index_mut(pos);
            *heap_idx = HeapIndex(i);
        }

        (heap, map)
    }

    fn heapify_up<TChangeHandler: std::ops::FnMut(MediatorIndex, HeapIndex)>(
        &mut self,
        position: HeapIndex,
        change_handler: &mut TChangeHandler,
    ) {
        debug_assert!(position < self.len(), "Out of index in heapify_up");
        let HeapIndex(mut position) = position;
        while position > 0 {
            let parent_pos = (position - 1) / 2;
            if self.data[parent_pos].priority >= self.data[position].priority {
                break;
            }
            self.data.swap(parent_pos, position);
            change_handler(self.data[position].outer_pos, HeapIndex(position));
            position = parent_pos;
        }
        change_handler(self.data[position].outer_pos, HeapIndex(position));
    }

    fn heapify_down<TChangeHandler: std::ops::FnMut(MediatorIndex, HeapIndex)>(
        &mut self,
        position: HeapIndex,
        change_handler: &mut TChangeHandler,
    ) {
        debug_assert!(position < self.len(), "Out of index in heapify_down");
        let HeapIndex(mut position) = position;
        loop {
            let max_child_idx = {
                let child1 = position * 2 + 1;
                let child2 = child1 + 1;
                if child1 >= self.data.len() {
                    break;
                }
                if child2 < self.data.len()
                    && self.data[child1].priority <= self.data[child2].priority
                {
                    child2
                } else {
                    child1
                }
            };

            if self.data[position].priority >= self.data[max_child_idx].priority {
                break;
            }
            self.data.swap(position, max_child_idx);
            change_handler(self.data[position].outer_pos, HeapIndex(position));
            position = max_child_idx;
        }
        change_handler(self.data[position].outer_pos, HeapIndex(position));
    }
}

/// Useful to create iterator for outer struct
/// Does NOT guarantee any particular order
pub(crate) struct BinaryHeapIterator<'a, TPriority> {
    inner: std::slice::Iter<'a, HeapEntry<TPriority>>,
}

impl<'a, TPriority> Iterator for BinaryHeapIterator<'a, TPriority> {
    type Item = (MediatorIndex, &'a TPriority);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|entry: &'a HeapEntry<TPriority>| (entry.outer_pos, &entry.priority))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }

    #[inline]
    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.inner.count()
    }
}

// Default implementations

impl<TPriority: Debug> Debug for HeapEntry<TPriority> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            f,
            "{{outer: {:?}, priority: {:?}}}",
            &self.outer_pos, &self.priority
        )
    }
}

impl<TPriority: Debug + Ord> Debug for BinaryHeap<TPriority> {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        self.data.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use crate::mediator::Mediator;

    use super::*;
    use std::cmp::Reverse;
    use std::collections::hash_map::RandomState;
    use std::collections::{HashMap, HashSet};

    fn is_valid_heap<TP: Ord>(heap: &BinaryHeap<TP>) -> bool {
        for (i, current) in heap.data.iter().enumerate().skip(1) {
            let parent = &heap.data[(i - 1) / 2];
            if parent.priority < current.priority {
                return false;
            }
        }
        true
    }

    #[test]
    fn test_heap_fill() {
        let items = [
            70, 50, 0, 1, 2, 4, 6, 7, 9, 72, 4, 4, 87, 78, 72, 6, 7, 9, 2, -50, -72, -50, -42, -1,
            -3, -13,
        ];
        let mut maximum = std::i32::MIN;
        let mut heap = BinaryHeap::<i32>::with_capacity(0);
        assert!(heap.look_into(HeapIndex(0)).is_none());
        assert!(is_valid_heap(&heap), "Heap state is invalid");
        for (key, x) in items
            .iter()
            .enumerate()
            .map(|(i, &x)| (MediatorIndex(i), x))
        {
            if x > maximum {
                maximum = x;
            }
            heap.push(key, x, |_, _| {});
            assert!(
                is_valid_heap(&heap),
                "Heap state is invalid after pushing {}",
                x
            );
            assert!(heap.look_into(HeapIndex(0)).is_some());
            let (_, &heap_max) = heap.look_into(HeapIndex(0)).unwrap();
            assert_eq!(maximum, heap_max)
        }
    }

    #[test]
    fn test_change_logger() {
        let items = [
            2, 3, 21, 22, 25, 29, 36, 90, 89, 88, 87, 83, 48, 50, 52, 69, 65, 55, 73, 75, 76, -53,
            78, 81, -45, -41, 91, -34, -33, -31, -27, -22, -19, -8, -5, -3,
        ];
        let mut last_positions = HashMap::<MediatorIndex, HeapIndex>::new();
        let mut heap = BinaryHeap::<i32>::with_capacity(0);
        let mut on_pos_change = |outer_pos: MediatorIndex, position: HeapIndex| {
            last_positions.insert(outer_pos, position);
        };
        for (i, &x) in items.iter().enumerate() {
            heap.push(MediatorIndex(i), x, &mut on_pos_change);
        }
        assert_eq!(heap.usize_len(), last_positions.len());
        for i in 0..items.len() {
            let rem_idx = MediatorIndex(i);
            assert!(
                last_positions.contains_key(&rem_idx),
                "Not for all items change_handler called"
            );
            let position = last_positions[&rem_idx];
            assert_eq!(
                items[(heap.look_into(position).unwrap().0).0],
                *heap.look_into(position).unwrap().1
            );
            assert_eq!(heap.look_into(position).unwrap().0, rem_idx);
        }

        let mut removed = HashSet::<MediatorIndex>::new();
        loop {
            let mut on_pos_change = |key: MediatorIndex, position: HeapIndex| {
                last_positions.insert(key, position);
            };
            let popped = heap.remove(HeapIndex(0), &mut on_pos_change);
            if popped.is_none() {
                break;
            }
            let (key, _) = popped.unwrap();
            last_positions.remove(&key);
            removed.insert(key);
            assert_eq!(heap.usize_len(), last_positions.len());
            for i in (0..items.len())
                .into_iter()
                .filter(|i| !removed.contains(&MediatorIndex(*i)))
            {
                let rem_idx = MediatorIndex(i);
                assert!(
                    last_positions.contains_key(&rem_idx),
                    "Not for all items change_handler called"
                );
                let position = last_positions[&rem_idx];
                assert_eq!(
                    items[(heap.look_into(position).unwrap().0).0],
                    *heap.look_into(position).unwrap().1
                );
                assert_eq!(heap.look_into(position).unwrap().0, rem_idx);
            }
        }
    }

    #[test]
    fn test_pop() {
        let items = [
            -16, 5, 11, -1, -34, -42, -5, -6, 25, -35, 11, 35, -2, 40, 42, 40, -45, -48, 48, -38,
            -28, -33, -31, 34, -18, 25, 16, -33, -11, -6, -35, -38, 35, -41, -38, 31, -38, -23, 26,
            44, 38, 11, -49, 30, 7, 13, 12, -4, -11, -24, -49, 26, 42, 46, -25, -22, -6, -42, 28,
            45, -47, 8, 8, 21, 49, -12, -5, -33, -37, 24, -3, -26, 6, -13, 16, -40, -14, -39, -26,
            12, -44, 47, 45, -41, -22, -11, 20, 43, -44, 24, 47, 40, 43, 9, 19, 12, -17, 30, -36,
            -50, 24, -2, 1, 1, 5, -19, 21, -38, 47, 34, -14, 12, -30, 24, -2, -32, -10, 40, 34, 2,
            -33, 9, -31, -3, -15, 28, 50, -37, 35, 19, 35, 13, -2, 46, 28, 35, -40, -19, -1, -33,
            -42, -35, -12, 19, 29, 10, -31, -4, -9, 24, 15, -27, 13, 20, 15, 19, -40, -41, 40, -25,
            45, -11, -7, -19, 11, -44, -37, 35, 2, -49, 11, -37, -14, 13, 41, 10, 3, 19, -32, -12,
            -12, 33, -26, -49, -45, 24, 47, -29, -25, -45, -36, 40, 24, -29, 15, 36, 0, 47, 3, -45,
        ];

        let mut heap = BinaryHeap::<i32>::with_capacity(0);
        for (i, &x) in items.iter().enumerate() {
            heap.push(MediatorIndex(i), x, |_, _| {});
        }
        assert!(is_valid_heap(&heap), "Heap is invalid before pops");

        let mut sorted_items = items;
        sorted_items.sort_unstable_by_key(|&x| Reverse(x));
        for &x in sorted_items.iter() {
            let pop_res = heap.remove(HeapIndex(0), |_, _| {});
            assert!(pop_res.is_some());
            let (rem_idx, val) = pop_res.unwrap();
            assert_eq!(val, x);
            assert_eq!(items[rem_idx.0], val);
            assert!(is_valid_heap(&heap), "Heap is invalid after {}", x);
        }

        assert_eq!(heap.remove(HeapIndex(0), |_, _| {}), None);
    }

    #[test]
    fn test_remove() {
        let mut heap = BinaryHeap::with_capacity(16);
        for i in 0..16 {
            heap.push(MediatorIndex(i), i, |_, _| {});
        }
        assert!(is_valid_heap(&heap));
        for _ in 0..5 {
            heap.remove(HeapIndex(5), |_, _| {});
            assert!(is_valid_heap(&heap));
        }
    }

    #[test]
    fn test_change_priority() {
        let pairs = [
            (MediatorIndex(0), 0),
            (MediatorIndex(1), 1),
            (MediatorIndex(2), 2),
            (MediatorIndex(3), 3),
            (MediatorIndex(4), 4),
        ];

        let mut heap = BinaryHeap::with_capacity(0);
        for (key, priority) in pairs.iter().cloned() {
            heap.push(key, priority, |_, _| {});
        }
        assert!(is_valid_heap(&heap), "Invalid before change");
        heap.change_priority(HeapIndex(3), 10, |_, _| {});
        assert!(is_valid_heap(&heap), "Invalid after upping");
        heap.change_priority(HeapIndex(2), -10, |_, _| {});
        assert!(is_valid_heap(&heap), "Invalid after lowering");
    }

    #[test]
    fn create_heap_hash_test() {
        let priorities = [
            16i32, 16, 5, 20, 10, 12, 10, 8, 12, 2, 20, -1, -18, 5, -16, 1, 7, 3, 17, -20, -4, 3,
            -7, -5, -8, 19, -19, -16, 3, 4, 17, 13, 3, 11, -9, 0, -10, -2, 16, 19, -12, -4, 19, 7,
            16, -19, -9, -17, 6, -16, -3, 11, -14, -15, -10, 13, 11, -14, 18, -8, -9, -4, 5, -4,
            17, 6, -16, -5, 12, 12, -3, 8, 5, -4, 7, 10, 7, -11, 18, -16, 18, 4, -15, -4, -13, 7,
            -14, -16, -18, -10, 13, -1, -9, 0, -18, -4, -13, 16, 10, -20, 19, 20, 0, -9, -7, 14,
            19, -8, -18, -1, -17, -11, 13, 12, -15, 0, -18, 6, -13, -17, -3, 18, 2, 12, 12, 4, -14,
            -11, -10, -9, 3, 14, 8, 7, 13, 13, -17, -9, -4, -19, -6, 1, 9, 5, 20, -9, -19, -20,
            -18, -8, 7,
        ];
        let (heap, key_to_pos): (_, Mediator<_, RandomState>) =
            BinaryHeap::produce_from_iter_hash(priorities.iter().cloned().map(|x| (x, x)));
        assert!(is_valid_heap(&heap), "Must be valid heap");
        for (map_idx, (key, heap_idx)) in key_to_pos.iter().enumerate() {
            assert_eq!(
                Some((MediatorIndex(map_idx), key)),
                heap.look_into(heap_idx)
            );
        }
    }

    #[test]
    fn test_clear() {
        let mut heap = BinaryHeap::with_capacity(0);
        for x in 0..5 {
            heap.push(MediatorIndex(x), x, |_, _| {});
        }
        assert!(!heap.is_empty(), "Heap must be non empty");
        heap.data.clear();
        assert!(heap.is_empty(), "Heap must be empty");
        assert_eq!(heap.remove(HeapIndex(0), |_, _| {}), None);
    }

    #[test]
    fn test_change_change_outer_pos() {
        let mut heap = BinaryHeap::with_capacity(0);
        for x in 0..5 {
            heap.push(MediatorIndex(x), x, |_, _| {});
        }
        assert_eq!(heap.look_into(HeapIndex(0)), Some((MediatorIndex(4), &4)));
        assert_eq!(
            heap.change_outer_pos(MediatorIndex(10), HeapIndex(0)),
            MediatorIndex(4)
        );
        assert_eq!(heap.look_into(HeapIndex(0)), Some((MediatorIndex(10), &4)));
    }
}
