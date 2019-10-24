use std::cmp::{Ord, Ordering};
use std::collections::HashMap;
use std::fmt::Debug;
use std::iter::FromIterator;
use std::vec::Vec;

struct HeapEntry<TKey, TPriority> {
    key: TKey,
    priority: TPriority,
}

pub(crate) struct BinaryHeap<TKey, TPriority>
where
    TPriority: Ord,
{
    data: Vec<HeapEntry<TKey, TPriority>>,
}

impl<TKey, TPriority: Ord> BinaryHeap<TKey, TPriority> {
    pub(crate) fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }

    /// Puts key and priority in queue, returns its final position
    /// Calls change_handler for every move of old values
    #[inline(always)]
    pub(crate) fn push<TChangeHandler: std::ops::FnMut(&TKey, usize)>(
        &mut self,
        key: TKey,
        priority: TPriority,
        change_handler: TChangeHandler,
    ) {
        self.data.push(HeapEntry { key, priority });
        self.heapify_up(self.data.len() - 1, change_handler);
    }

    /// Removes item with the biggest priority
    /// Time complexity - O(log n) swaps and change_handler calls
    #[inline(always)]
    pub(crate) fn pop<TChangeHandler: std::ops::FnMut(&TKey, usize)>(
        &mut self,
        change_handler: TChangeHandler,
    ) -> Option<(TKey, TPriority)> {
        self.remove(0, change_handler)
    }

    #[inline(always)]
    pub(crate) fn peek(&self) -> Option<(&TKey, &TPriority)> {
        self.look_into(0)
    }

    /// Removes item at position and returns it
    /// Time complexity - O(log n) swaps and change_handler calls
    pub(crate) fn remove<TChangeHandler: std::ops::FnMut(&TKey, usize)>(
        &mut self,
        position: usize,
        change_handler: TChangeHandler,
    ) -> Option<(TKey, TPriority)> {
        if self.data.len() <= position {
            return None;
        }
        if position == self.data.len() - 1 {
            let result = self.data.pop().unwrap();
            return Some((result.key, result.priority));
        }
        self.swap_items(position, self.data.len() - 1);
        let result = self.data.pop().unwrap();
        self.heapify_down(position, change_handler);
        Some((result.key, result.priority))
    }

    #[inline(always)]
    pub(crate) fn look_into(&self, position: usize) -> Option<(&TKey, &TPriority)> {
        let entry = self.data.get(position)?;
        Some((&entry.key, &entry.priority))
    }

    /// Changes priority of queue item
    pub(crate) fn change_priority<TChangeHandler: std::ops::FnMut(&TKey, usize)>(
        &mut self,
        position: usize,
        updated: TPriority,
        change_handler: TChangeHandler,
    ) {
        if position >= self.data.len() {
            panic!("Out of index during changing priority");
        }

        let old = std::mem::replace(&mut self.data[position].priority, updated);
        match old.cmp(&self.data[position].priority) {
            Ordering::Less => {
                self.heapify_up(position, change_handler);
            }
            Ordering::Equal => {}
            Ordering::Greater => {
                self.heapify_down(position, change_handler);
            }
        }
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }

    fn heapify_up<TChangeHandler: std::ops::FnMut(&TKey, usize)>(
        &mut self,
        position: usize,
        mut change_handler: TChangeHandler,
    ) {
        debug_assert!(position < self.data.len(), "Out of index in heapify_up");
        let mut position = position;
        while position > 0 {
            let parent_pos = (position - 1) >> 1;
            if self.data[parent_pos].priority < self.data[position].priority {
                self.swap_items(parent_pos, position);
                change_handler(&self.data[position].key, position);
                position = parent_pos;
            } else {
                break;
            }
        }
        change_handler(&self.data[position].key, position);
    }

    fn heapify_down<TChangeHandler: std::ops::FnMut(&TKey, usize)>(
        &mut self,
        position: usize,
        mut change_handler: TChangeHandler,
    ) {
        debug_assert!(position < self.data.len(), "Out of index in heapify_down");
        let mut position = position;
        loop {
            let max_child_idx = {
                let child1 = (position << 1) + 1;
                let child2 = child1 + 1;
                if child1 >= self.data.len() {
                    break;
                }
                if child2 >= self.data.len()
                    || self.data[child2].priority < self.data[child1].priority
                {
                    child1
                } else {
                    child2
                }
            };

            if self.data[position].priority < self.data[max_child_idx].priority {
                self.swap_items(position, max_child_idx);
                change_handler(&self.data[position].key, position);
                position = max_child_idx;
            } else {
                break;
            }
        }
        change_handler(&self.data[position].key, position);
    }

    #[inline(always)]
    fn swap_items(&mut self, pos1: usize, pos2: usize) {
        debug_assert!(pos1 < self.data.len(), "Out of index in first pos in swap");
        debug_assert!(pos2 < self.data.len(), "Out of index in second pos in swap");
        self.data.swap(pos1, pos2);
    }
}

impl<TKey: std::hash::Hash + Clone + Eq, TPriority: Ord> BinaryHeap<TKey, TPriority> {
    pub(crate) fn generate_mapping(&self) -> HashMap<TKey, usize> {
        self.data
            .iter()
            .enumerate()
            .map(|(index, entry)| (entry.key.clone(), index))
            .collect()
    }
}

impl<TKey, TPriority: Ord> FromIterator<(TKey, TPriority)> for BinaryHeap<TKey, TPriority> {
    fn from_iter<T: IntoIterator<Item = (TKey, TPriority)>>(iter: T) -> Self {
        let data: Vec<HeapEntry<TKey, TPriority>> = iter
            .into_iter()
            .map(|(key, priority)| HeapEntry { key, priority })
            .collect();
        if data.len() < 2 {
            return Self { data };
        }
        let mut res = Self { data };
        let heapify_start = std::cmp::min(res.data.len() / 2 + 2, res.data.len());
        for i in (0..heapify_start).rev() {
            res.heapify_down(i, |_, _| {});
        }
        res
    }
}

// Default implementations

impl<TKey: Clone, TPriority: Clone> Clone for HeapEntry<TKey, TPriority> {
    fn clone(&self) -> Self {
        Self {
            key: self.key.clone(),
            priority: self.priority.clone(),
        }
    }
}

impl<TKey: Copy, TPriority: Copy> Copy for HeapEntry<TKey, TPriority> {}

unsafe impl<TKey: Sync, TPriority: Sync> Sync for HeapEntry<TKey, TPriority> {}

unsafe impl<TKey: Send, TPriority: Send> Send for HeapEntry<TKey, TPriority> {}

impl<TKey: Debug, TPriority: Debug> Debug for HeapEntry<TKey, TPriority> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            f,
            "{{key: {:?}, priority: {:?}}}",
            &self.key, &self.priority
        )
    }
}

impl<TKey: Clone, TPriority: Clone + Ord> Clone for BinaryHeap<TKey, TPriority> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
        }
    }
}

unsafe impl<TKey: Sync, TPriority: Sync + Ord> Sync for BinaryHeap<TKey, TPriority> {}

unsafe impl<TKey: Send, TPriority: Send + Ord> Send for BinaryHeap<TKey, TPriority> {}

impl<TKey: Debug, TPriority: Debug + Ord> Debug for BinaryHeap<TKey, TPriority> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        self.data.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Reverse;
    use std::collections::{HashMap, HashSet};

    fn is_valid_heap<TK, TP: Ord>(heap: &BinaryHeap<TK, TP>) -> bool {
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
        let mut heap = BinaryHeap::<(), i32>::new();
        assert!(heap.peek().is_none());
        assert!(is_valid_heap(&heap), "Heap state is invalid");
        for &x in items.iter() {
            if x > maximum {
                maximum = x;
            }
            heap.push((), x, |_, _| {});
            assert!(
                is_valid_heap(&heap),
                "Heap state is invalid after pushing {}",
                x
            );
            assert!(heap.peek().is_some());
            let (_, &heap_max) = heap.peek().unwrap();
            assert_eq!(maximum, heap_max)
        }
    }

    #[test]
    fn test_change_logger() {
        let items = [
            2, 3, 21, 22, 25, 29, 36, -90, -89, -88, -87, -83, 48, 50, 52, -69, -65, -55, 73, 75,
            76, -53, 78, 81, -45, -41, 91, -34, -33, -31, -27, -22, -19, -8, -5, -3,
        ];
        let mut last_positions = HashMap::<i32, usize>::new();
        let mut heap = BinaryHeap::<i32, i32>::new();
        let heap_ptr: *const BinaryHeap<i32, i32> = &heap;
        let mut on_pos_change = |key: &i32, position: usize| {
            // Hack to avoid borrow checker
            let heap_local = unsafe { &*heap_ptr };
            assert_eq!(*heap_local.look_into(position).unwrap().0, *key);
            assert_eq!(
                heap_local.look_into(position).unwrap().0,
                heap_local.look_into(position).unwrap().1
            );
            last_positions.insert(*key, position);
        };
        for &x in items.iter() {
            heap.push(x, x, &mut on_pos_change);
        }
        for &x in items.iter() {
            assert!(
                last_positions.contains_key(&x),
                "Not for all items change_handler called"
            );
            let position = last_positions[&x];
            assert_eq!(
                heap.look_into(position).unwrap().0,
                heap.look_into(position).unwrap().1
            );
            assert_eq!(*heap.look_into(position).unwrap().0, x);
        }

        let mut removed = HashSet::<i32>::new();
        loop {
            let mut on_pos_change = |key: &i32, position: usize| {
                // Hack to avoid borrow checker
                let heap_local = unsafe { &*heap_ptr };
                assert_eq!(*heap_local.look_into(position).unwrap().0, *key);
                assert_eq!(
                    heap_local.look_into(position).unwrap().0,
                    heap_local.look_into(position).unwrap().1
                );
                last_positions.insert(*key, position);
            };
            let popped = heap.pop(&mut on_pos_change);
            if popped.is_none() {
                break;
            }
            let (key, _) = popped.unwrap();
            last_positions.remove(&key);
            removed.insert(key);
            for x in items.iter().cloned().filter(|x| !removed.contains(x)) {
                assert!(
                    last_positions.contains_key(&x),
                    "Not for all items change_handler called"
                );
                let position = last_positions[&x];
                assert_eq!(
                    heap.look_into(position).unwrap().0,
                    heap.look_into(position).unwrap().1
                );
                assert_eq!(*heap.look_into(position).unwrap().0, x);
            }
        }
    }

    #[test]
    fn test_pop() {
        let mut items = [
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

        let mut heap = BinaryHeap::<i32, i32>::new();
        for &x in items.iter() {
            heap.push(x, x, |_, _| {});
        }
        assert!(is_valid_heap(&heap), "Heap is invalid before pops");

        items.sort_unstable_by_key(|&x| Reverse(x));
        for &x in items.iter() {
            assert_eq!(heap.pop(|_, _| {}), Some((x, x)));
            assert!(is_valid_heap(&heap), "Heap is invalid after {}", x);
        }

        assert_eq!(heap.pop(|_, _| {}), None);
    }

    #[test]
    fn test_change_priority() {
        let pairs = [
            ("first", 0),
            ("second", 1),
            ("third", 2),
            ("fourth", 3),
            ("fifth", 4),
        ];

        let mut heap: BinaryHeap<&str, i32> = pairs.iter().cloned().collect();
        assert!(is_valid_heap(&heap), "Invalid before change");
        heap.change_priority(3, 10, |_, _| {});
        assert!(is_valid_heap(&heap), "Invalid after upping");
        heap.change_priority(21, -10, |_, _| {});
        assert!(is_valid_heap(&heap), "Invalid after lowering");
    }

    #[test]
    fn test_from_iter() {
        let data = [
            16, 5, 20, 10, 12, 10, 8, 12, 2, 20, -1, -18, 5, -16, 1, 7, 3, 17, -20, -4, 3, -7, -5,
            -8, 19, -19, -16, 3, 4, 17, 13, 3, 11, -9, 0, -10, -2, 16, 19, -12, -4, 19, 7, 16, -19,
            -9, -17, 6, -16, -3, 11, -14, -15, -10, 13, 11, -14, 18, -8, -9, -4, 5, -4, 17, 6, -16,
            -5, 12, 12, -3, 8, 5, -4, 7, 10, 7, -11, 18, -16, 18, 4, -15, -4, -13, 7, -14, -16,
            -18, -10, 13, -1, -9, 0, -18, -4, -13, 16, 10, -20, 19, 20, 0, -9, -7, 14, 19, -8, -18,
            -1, -17, -11, 13, 12, -15, 0, -18, 6, -13, -17, -3, 18, 2, 12, 12, 4, -14, -11, -10,
            -9, 3, 14, 8, 7, 13, 13, -17, -9, -4, -19, -6, 1, 9, 5, 20, -9, -19, -20, -18, -8, 7,
        ];

        for len in 0..data.len() {
            let heap: BinaryHeap<i32, i32> = data.iter().take(len).map(|&x| (x, x)).collect();
            assert!(
                is_valid_heap(&heap),
                "Invalid heap from iterator on len {}\n{:?}",
                len,
                &heap,
            );
        }
    }

    #[test]
    fn test_generate_mapping() {
        let data = [
            16, 5, 20, 10, 12, 10, 8, 12, 2, 20, -1, -18, 5, -16, 1, 7, 3, 17, -20, -4, 3, -7, -5,
            -8, 19, -19, -16, 3, 4, 17, 13, 3, 11, -9, 0, -10, -2, 16, 19, -12, -4, 19, 7, 16, -19,
            -9, -17, 6, -16, -3, 11, -14, -15, -10, 13, 11, -14, 18, -8, -9, -4, 5, -4, 17, 6, -16,
            -5, 12, 12, -3, 8, 5, -4, 7, 10, 7, -11, 18, -16, 18, 4, -15, -4, -13, 7, -14, -16,
            -18, -10, 13, -1, -9, 0, -18, -4, -13, 16, 10, -20, 19, 20, 0, -9, -7, 14, 19, -8, -18,
            -1, -17, -11, 13, 12, -15, 0, -18, 6, -13, -17, -3, 18, 2, 12, 12, 4, -14, -11, -10,
            -9, 3, 14, 8, 7, 13, 13, -17, -9, -4, -19, -6, 1, 9, 5, 20, -9, -19, -20, -18, -8, 7,
        ];

        let heap: BinaryHeap<i32, i32> = data.iter().map(|&x| (x, x)).collect();
        let mapping = heap.generate_mapping();
        for (key, pos) in mapping {
            assert_eq!(heap.data[pos].key, key);
        }
    }
}
