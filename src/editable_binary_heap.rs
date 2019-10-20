use std::cmp::{Ord, Ordering};
use std::vec::Vec;

struct HeapEntry<TKey, TPriority>
where
    TPriority: Ord,
{
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
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
        }
    }

    /// Puts key and priority in queue, returns its final position
    /// Calls change_handler for every move of old values
    pub fn push(
        &mut self,
        key: TKey,
        priority: TPriority,
        change_handler: &mut dyn std::ops::FnMut(&TKey, usize),
    ) {
        self.data.push(HeapEntry { key, priority });
        change_handler(&self.data.last().unwrap().key, self.data.len() - 1);
        self.heapify_up(self.data.len() - 1, change_handler);
    }

    /// Removes item with the biggest priority
    /// Time complexity - O(log n) swaps and change_handler calls
    pub fn pop(
        &mut self,
        change_handler: &mut dyn std::ops::FnMut(&TKey, usize),
    ) -> Option<(TKey, TPriority)> {
        self.remove(0, change_handler)
    }

    pub fn peek(&self) -> Option<(&TKey, &TPriority)> {
        self.look_into(0)
    }

    /// Removes item at position and returns it
    /// Time complexity - O(log n) swaps and change_handler calls
    pub fn remove(
        &mut self,
        position: usize,
        change_handler: &mut dyn std::ops::FnMut(&TKey, usize),
    ) -> Option<(TKey, TPriority)> {
        if self.data.len() <= position {
            return None;
        }
        if position == self.data.len() - 1 {
            let result = self.data.pop().unwrap();
            return Some((result.key, result.priority));
        }
        self.swap_items(position, self.data.len() - 1, change_handler);
        let result = self.data.pop().unwrap();
        self.heapify_down(position, change_handler);
        Some((result.key, result.priority))
    }

    pub fn look_into(&self, position: usize)->Option<(&TKey, &TPriority)>{
        if self.data.len()<=position{
            None
        }
        else{
            let entry = &self.data[position];
            Some((&entry.key, &entry.priority))
        }
    }

    /// Changes priority of queue item
    pub fn change_priority(
        &mut self,
        position: usize,
        updated: TPriority,
        change_handler: &mut dyn std::ops::FnMut(&TKey, usize),
    ) {
        if position >= self.data.len() {
            return;
        }

        let old = std::mem::replace(&mut self.data[position].priority, updated);
        if old < self.data[position].priority {
            self.heapify_up(position, change_handler);
        } else {
            self.heapify_down(position, change_handler);
        }
    }

    fn heapify_up(
        &mut self,
        position: usize,
        change_handler: &mut dyn std::ops::FnMut(&TKey, usize),
    ) {
        debug_assert!(position < self.data.len(), "Out of index in heapify_up");
        let mut position = position;
        while position > 0 {
            let parent_pos = (position - 1) >> 1;
            if self.data[parent_pos].priority < self.data[position].priority {
                self.swap_items(parent_pos, position, change_handler);
                position = parent_pos;
            } else {
                break;
            }
        }
    }

    fn heapify_down(
        &mut self,
        position: usize,
        change_handler: &mut dyn std::ops::FnMut(&TKey, usize),
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
                self.swap_items(position, max_child_idx, change_handler);
                position = max_child_idx;
            } else {
                break;
            }
        }
    }

    fn swap_items(
        &mut self,
        pos1: usize,
        pos2: usize,
        change_handler: &mut dyn std::ops::FnMut(&TKey, usize),
    ) {
        debug_assert!(pos1 < self.data.len(), "Out of index in first pos in swap");
        debug_assert!(pos2 < self.data.len(), "Out of index in second pos in swap");
        self.data.swap(pos1, pos2);
        change_handler(&self.data[pos1].key, pos1);
        change_handler(&self.data[pos2].key, pos2);
    }
}

// Default implementations

impl<TKey: Clone, TPriority: Clone + Ord> Clone for HeapEntry<TKey, TPriority> {
    fn clone(&self) -> Self {
        Self {
            key: self.key.clone(),
            priority: self.priority.clone(),
        }
    }
}

impl<TKey: Copy, TPriority: Copy + Ord> Copy for HeapEntry<TKey, TPriority> {}
unsafe impl<TKey: Sync, TPriority: Sync + Ord> Sync for HeapEntry<TKey, TPriority> {}
unsafe impl<TKey: Send, TPriority: Send + Ord> Send for HeapEntry<TKey, TPriority> {}

impl<TKey: Clone, TPriority: Clone + Ord> Clone for BinaryHeap<TKey, TPriority> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
        }
    }
}
unsafe impl<TKey: Sync, TPriority: Sync + Ord> Sync for BinaryHeap<TKey, TPriority> {}
unsafe impl<TKey: Send, TPriority: Send + Ord> Send for BinaryHeap<TKey, TPriority> {}

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
        for (i, &x) in items.iter().enumerate() {
            if x > maximum {
                maximum = x;
            }
            heap.push((), x, &mut |_, _| {});
            assert!(is_valid_heap(&heap), "Heap state is invalid");
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
            assert_eq!(heap.look_into(position).unwrap().0, heap.look_into(position).unwrap().1);
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
                assert_eq!(heap.look_into(position).unwrap().0, heap.look_into(position).unwrap().1);
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
            heap.push(x, x, &mut |_, _| {});
        }
        assert!(is_valid_heap(&heap), "Heap is invalid before pops");

        items.sort_unstable_by_key(|&x| Reverse(x));
        for &x in items.iter() {
            assert_eq!(heap.pop(&mut |_, _| {}), Some((x, x)));
            assert!(is_valid_heap(&heap), format!("Heap is invalid after {}", x));
        }

        assert_eq!(heap.pop(&mut |_, _| {}), None);
    }

    #[test]
    fn test_change_priority(){
        let pairs = [
            ("first", 0),
            ("second", 1),
            ("third", 2),
            ("fourth", 3),
            ("fifth", 4),
        ];

        let mut heap = BinaryHeap::<&str, i32>::with_capacity(pairs.len());
        for &(key, value) in pairs.iter(){
            heap.push(key, value, &mut |_,_|{});
        }
        assert!(is_valid_heap(&heap), "Invalid before change");
        heap.change_priority(3, 10, &mut |_,_|{});
        assert!(is_valid_heap(&heap), "Invalid after upping");
        heap.change_priority(21, -10, &mut |_,_|{});
        assert!(is_valid_heap(&heap), "Invalid after lowering");
    }
}
