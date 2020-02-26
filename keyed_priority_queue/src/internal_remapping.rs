use crate::editable_binary_heap::{BinaryHeap, HeapIndex};

use std::fmt::Debug;

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Hash)]
pub(crate) struct RemapIndex(usize);

impl RemapIndex {
    #[cfg(test)]
    #[inline(always)]
    pub(crate) fn new(v: usize) -> Self {
        Self(v)
    }

    #[inline(always)]
    pub(crate) fn as_usize(self) -> usize {
        self.0
    }
}

struct RemappingEntry<TKey> {
    key: TKey,
    heap_idx: HeapIndex,
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub(crate) struct RemapKeyPositionUpdate<'a, TKey> {
    pub(crate) key: &'a TKey,
    pub(crate) new_pos: RemapIndex,
}

// Intermediate wrapper around heap to avoid log(n) updates of hash map
pub(crate) struct QueueWrapper<TKey, TPriority: Ord> {
    heap: BinaryHeap<TPriority>,
    remapping: Vec<RemappingEntry<TKey>>,
}

impl<TKey, TPriority> QueueWrapper<TKey, TPriority>
where
    TPriority: Ord,
{
    #[inline(always)]
    pub(crate) fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            remapping: Vec::new(),
        }
    }

    #[inline(always)]
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(capacity),
            remapping: Vec::with_capacity(capacity),
        }
    }

    #[inline(always)]
    pub(crate) fn reserve(&mut self, additional: usize) {
        self.heap.reserve(additional);
        self.remapping.reserve(additional);
    }

    // Adds new key and priority to queue
    // Doesn't check uniqueness of key
    // Returns remap index of inserted element
    #[inline(always)]
    pub(crate) fn push(&mut self, key: TKey, priority: TPriority) -> RemapIndex {
        let heap = &mut self.heap;
        let remapping = &mut self.remapping;

        let remap_idx = RemapIndex(remapping.len());
        remapping.push(RemappingEntry {
            key,
            heap_idx: heap.len(),
        });
        heap.push(remap_idx, priority, |changed_rem_idx, heap_idx| {
            remapping[changed_rem_idx.0].heap_idx = heap_idx;
        });
        remap_idx
    }

    // Removes the most prioritized element
    // Returns key, priority and possibly info about moved key position for key to pos mapping
    pub(crate) fn pop(
        &mut self,
    ) -> Option<(TKey, TPriority, Option<RemapKeyPositionUpdate<TKey>>)> {
        let heap = &mut self.heap;
        let remapping = &mut self.remapping;

        let (removed_idx, priority) = heap.pop(|changed_rem_idx, heap_idx| {
            remapping[changed_rem_idx.0].heap_idx = heap_idx;
        })?;

        let removed_entry = remapping.swap_remove(removed_idx.0);
        if removed_idx.0 == remapping.len() {
            return Some((removed_entry.key, priority, None));
        }
        // Update heap to use new remapping position
        let old_remap_index = heap.change_key(removed_idx, remapping[removed_idx.0].heap_idx);
        debug_assert_eq!(old_remap_index, RemapIndex(remapping.len()));

        // To update hashmap
        let update = RemapKeyPositionUpdate {
            key: &remapping[removed_idx.0].key,
            new_pos: removed_idx,
        };
        Some((removed_entry.key, priority, Some(update)))
    }

    #[inline(always)]
    pub(crate) fn peek(&self) -> Option<(&TKey, &TPriority)> {
        let (rem_idx, priority) = self.heap.peek()?;
        Some((&self.remapping[rem_idx.0].key, priority))
    }

    #[inline(always)]
    pub(crate) fn get_priority(&self, rem_idx: RemapIndex) -> &TPriority {
        let heap_idx = self.remapping[rem_idx.0].heap_idx;
        let (rem_idx_fh, priority) = self.heap.look_into(heap_idx).unwrap();
        debug_assert_eq!(rem_idx_fh, rem_idx);
        priority
    }

    #[inline(always)]
    pub(crate) fn set_priority(&mut self, rem_idx: RemapIndex, priority: TPriority) {
        let heap = &mut self.heap;
        let remapping = &mut self.remapping;

        let heap_idx = remapping[rem_idx.0].heap_idx;
        heap.change_priority(heap_idx, priority, |changed_rem_idx, heap_idx| {
            remapping[changed_rem_idx.0].heap_idx = heap_idx;
        });
    }

    pub(crate) fn remove_item(
        &mut self,
        rem_idx: RemapIndex,
    ) -> (TKey, TPriority, Option<RemapKeyPositionUpdate<TKey>>) {
        let heap = &mut self.heap;
        let remapping = &mut self.remapping;

        let heap_idx = remapping[rem_idx.0].heap_idx;
        let (removed_idx, priority) = heap
            .remove(heap_idx, |changed_rem_idx, heap_idx| {
                remapping[changed_rem_idx.0].heap_idx = heap_idx;
            })
            .unwrap();
        debug_assert_eq!(removed_idx, rem_idx);

        let removed_entry = remapping.swap_remove(removed_idx.0);
        if removed_idx.0 == remapping.len() {
            return (removed_entry.key, priority, None);
        }

        // Update heap to use new remapping position
        let old_remap_index = heap.change_key(removed_idx, remapping[removed_idx.0].heap_idx);
        debug_assert_eq!(old_remap_index, RemapIndex(remapping.len()));

        // To update hashmap
        let update = RemapKeyPositionUpdate {
            key: &remapping[removed_idx.0].key,
            new_pos: removed_idx,
        };
        (removed_entry.key, priority, Some(update))
    }

    #[inline(always)]
    pub(crate) fn len(&self) -> RemapIndex {
        debug_assert_eq!(self.remapping.len(), self.heap.len().as_usize());
        RemapIndex(self.remapping.len())
    }

    #[inline(always)]
    pub(crate) fn is_empty(&self) -> bool {
        debug_assert_eq!(self.heap.is_empty(), self.remapping.is_empty());
        self.remapping.is_empty()
    }

    #[inline(always)]
    pub(crate) fn clear(&mut self) {
        self.heap.clear();
        self.remapping.clear();
    }
}

impl<TKey: Clone> Clone for RemappingEntry<TKey> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self {
            key: self.key.clone(),
            heap_idx: self.heap_idx,
        }
    }
}

impl<TKey: Copy> Copy for RemappingEntry<TKey> {}

impl<TKey: Debug> Debug for RemappingEntry<TKey> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{{key: {:?}, heap_idx: {:?}}}", &self.key, self.heap_idx)
    }
}

impl<'a, TKey: Debug> Debug for RemapKeyPositionUpdate<'a, TKey> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{{key: {:?}, new_pos: {:?}}}", self.key, self.new_pos)
    }
}

impl<TKey, TPriority> Clone for QueueWrapper<TKey, TPriority>
where
    TPriority: Ord + Clone,
    TKey: Clone,
{
    #[inline(always)]
    fn clone(&self) -> Self {
        Self {
            heap: self.heap.clone(),
            remapping: self.remapping.clone(),
        }
    }
}

impl<TKey: Debug, TPriority: Debug + Ord> Debug for QueueWrapper<TKey, TPriority> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "{:?}\n{:?}", &self.remapping, &self.heap)
    }
}

impl<TKey: Clone + std::hash::Hash + Eq, TPriority: Ord> QueueWrapper<TKey, TPriority> {
    pub(crate) fn build_from_iterator<TIter>(
        iter: TIter,
    ) -> (
        QueueWrapper<TKey, TPriority>,
        std::collections::HashMap<TKey, RemapIndex>,
    )
    where
        TIter: Iterator<Item = (TKey, TPriority)>,
    {
        use crate::editable_binary_heap::for_iteration_construction;
        use crate::editable_binary_heap::for_iteration_construction::{
            create_heap, make_heap_entry, make_heap_index, set_entry_priority,
        };
        use crate::editable_binary_heap::HeapEntry;
        use std::collections::hash_map::Entry;
        use std::collections::HashMap;

        let min_size = iter.size_hint().0;
        let max_size = iter.size_hint().1.unwrap_or(min_size);
        let mut for_heap: Vec<HeapEntry<TPriority>> = Vec::with_capacity(min_size);
        let mut for_wrapper: Vec<RemappingEntry<TKey>> = Vec::with_capacity(min_size);
        let mut for_map: HashMap<TKey, RemapIndex> = HashMap::with_capacity(max_size);

        for (key, priority) in iter {
            match for_map.entry(key) {
                Entry::Vacant(entry) => {
                    let new_idx = for_heap.len();
                    for_heap.push(make_heap_entry(RemapIndex(new_idx), priority));
                    for_wrapper.push(RemappingEntry {
                        key: entry.key().clone(),
                        heap_idx: make_heap_index(new_idx),
                    });
                    entry.insert(RemapIndex(new_idx));
                }
                Entry::Occupied(entry) => {
                    let index = entry.get().0;
                    set_entry_priority(&mut for_heap[index], priority);
                }
            }
        }

        let heap = create_heap(for_heap);
        for (heap_idx, &RemapIndex(v)) in for_iteration_construction::reader_iterator(&heap) {
            for_wrapper[v].heap_idx = heap_idx;
        }

        let wrapper_queue = QueueWrapper {
            heap,
            remapping: for_wrapper,
        };
        (wrapper_queue, for_map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{HashMap, HashSet};

    fn is_valid_mapping<TK, TP>(wrapper: &QueueWrapper<TK, TP>)
    where
        TP: Ord,
    {
        let mut heap_indices = HashSet::with_capacity(wrapper.len().0);
        for remap_idx in 0..wrapper.len().0 {
            let heap_idx = wrapper.remapping[remap_idx].heap_idx;
            let (hp_remap, _) = wrapper.heap.look_into(heap_idx).unwrap();
            assert_eq!(RemapIndex(remap_idx), hp_remap);
            assert!(
                heap_indices.insert(heap_idx),
                "Duplicate heap_idx {:?} found in remapping",
                heap_idx
            );
        }
    }

    fn create_wrapped_queue_asc(n: usize) -> QueueWrapper<usize, usize> {
        let mut wrapper = QueueWrapper::with_capacity(n);
        for i in 0..n {
            wrapper.push(i, i);
        }
        wrapper
    }

    fn create_wrapped_queue_desc(n: usize) -> QueueWrapper<usize, usize> {
        let mut wrapper = QueueWrapper::with_capacity(n);
        for i in (0..n).rev() {
            wrapper.push(i, i);
        }
        wrapper
    }

    fn create_wrapped_queue_shuffle() -> QueueWrapper<usize, usize> {
        let mut wrapper = QueueWrapper::with_capacity(5);
        for &i in &[1, 0, 4, 7, 5] {
            wrapper.push(i, i);
        }
        wrapper
    }

    fn is_valid_hash_mapping<TK, TP>(map: &HashMap<TK, RemapIndex>, wrapper: &QueueWrapper<TK, TP>)
    where
        TK: std::hash::Hash + Eq + Debug,
        TP: Ord + Debug,
    {
        assert_eq!(map.len(), wrapper.len().0);
        for remap_idx in 0..wrapper.len().0 {
            let wentry = &wrapper.remapping[remap_idx];
            assert_eq!(map[&wentry.key], RemapIndex(remap_idx));
        }
        for (key, remap) in map.iter() {
            assert_eq!(&wrapper.remapping[remap.0].key, key);
        }
    }

    #[test]
    fn test_push() {
        let mut queue = QueueWrapper::<usize, usize>::new();
        is_valid_mapping(&queue);
        assert_eq!(queue.push(0, 0), RemapIndex(0));
        is_valid_mapping(&queue);
        assert_eq!(queue.push(1, 1), RemapIndex(1));
        is_valid_mapping(&queue);
        assert_eq!(queue.push(2, 2), RemapIndex(2));
        is_valid_mapping(&queue);
        assert_eq!(queue.push(3, 3), RemapIndex(3));
        is_valid_mapping(&queue);
        assert_eq!(queue.push(4, 4), RemapIndex(4));
        is_valid_mapping(&queue);
    }

    #[test]
    fn test_pop_asc() {
        let mut queue = create_wrapped_queue_asc(5);
        assert_eq!(queue.pop(), Some((4, 4, None)));
        is_valid_mapping(&queue);
        assert_eq!(queue.pop(), Some((3, 3, None)));
        is_valid_mapping(&queue);
        assert_eq!(queue.pop(), Some((2, 2, None)));
        is_valid_mapping(&queue);
        assert_eq!(queue.pop(), Some((1, 1, None)));
        is_valid_mapping(&queue);
        assert_eq!(queue.pop(), Some((0, 0, None)));
        is_valid_mapping(&queue);
        assert_eq!(queue.pop(), None);
        is_valid_mapping(&queue);
    }

    #[test]
    fn test_pop_desc() {
        let mut queue = create_wrapped_queue_desc(5);
        let mut map: HashMap<usize, RemapIndex> = queue
            .remapping
            .iter()
            .enumerate()
            .map(|(i, x)| (x.key.clone(), RemapIndex(i)))
            .collect();
        is_valid_hash_mapping(&map, &queue);

        for i in (0..5).rev() {
            let taken = queue.pop();
            assert!(taken.is_some());
            let taken = taken.unwrap();
            assert_eq!(taken.0, i);
            assert_eq!(taken.1, i);
            assert!(
                map.remove(&taken.0).is_some(),
                "Tried to remove unknown key {}",
                taken.0
            );
            if let Some(mv) = taken.2 {
                *map.get_mut(mv.key).unwrap() = mv.new_pos;
            }
            is_valid_mapping(&queue);
            is_valid_hash_mapping(&map, &queue);
        }

        assert_eq!(queue.pop(), None);
        is_valid_mapping(&queue);
    }

    #[test]
    fn test_pop_shuffle() {
        let mut queue = create_wrapped_queue_shuffle();
        let mut map: HashMap<usize, RemapIndex> = queue
            .remapping
            .iter()
            .enumerate()
            .map(|(i, x)| (x.key.clone(), RemapIndex(i)))
            .collect();
        is_valid_hash_mapping(&map, &queue);

        for &i in &[7, 5, 4, 1, 0] {
            let taken = queue.pop();
            assert!(taken.is_some());
            let taken = taken.unwrap();
            assert_eq!(taken.0, i);
            assert_eq!(taken.1, i);
            assert!(
                map.remove(&taken.0).is_some(),
                "Tried to remove unknown key {}",
                taken.0
            );
            if let Some(mv) = taken.2 {
                *map.get_mut(mv.key).unwrap() = mv.new_pos;
            }
            is_valid_mapping(&queue);
            is_valid_hash_mapping(&map, &queue);
        }

        assert_eq!(queue.pop(), None);
        is_valid_mapping(&queue);
    }

    #[test]
    fn test_set_priority() {
        let mut queue = create_wrapped_queue_asc(10);
        queue.set_priority(RemapIndex(7), 20);
        let taken = queue.pop().unwrap();
        assert_eq!(taken.0, 7);
        for x in (0..10).rev().filter(|&x| x != 7) {
            let taken = queue.pop().unwrap();
            assert_eq!(taken.0, x)
        }
    }

    #[test]
    fn test_remove_item() {
        let mut queue = create_wrapped_queue_desc(5);
        let mut map: HashMap<usize, RemapIndex> = queue
            .remapping
            .iter()
            .enumerate()
            .map(|(i, x)| (x.key.clone(), RemapIndex(i)))
            .collect();
        is_valid_hash_mapping(&map, &queue);

        let i = 2;
        let taken = queue.remove_item(map[&i]);
        assert_eq!(taken.0, i);
        assert_eq!(taken.1, i);
        assert!(
            map.remove(&taken.0).is_some(),
            "Tried to remove unknown key {}",
            taken.0
        );
        if let Some(mv) = taken.2 {
            *map.get_mut(mv.key).unwrap() = mv.new_pos;
        }

        let i = 0;
        let taken = queue.remove_item(map[&i]);
        assert_eq!(taken.0, i);
        assert_eq!(taken.1, i);
        assert!(
            map.remove(&taken.0).is_some(),
            "Tried to remove unknown key {}",
            taken.0
        );
        if let Some(mv) = taken.2 {
            *map.get_mut(mv.key).unwrap() = mv.new_pos;
        }

        let i = 3;
        let taken = queue.remove_item(map[&i]);
        assert_eq!(taken.0, i);
        assert_eq!(taken.1, i);
        assert!(
            map.remove(&taken.0).is_some(),
            "Tried to remove unknown key {}",
            taken.0
        );
        if let Some(mv) = taken.2 {
            *map.get_mut(mv.key).unwrap() = mv.new_pos;
        }

        let i = 4;
        let taken = queue.remove_item(map[&i]);
        assert_eq!(taken.0, i);
        assert_eq!(taken.1, i);
        assert!(
            map.remove(&taken.0).is_some(),
            "Tried to remove unknown key {}",
            taken.0
        );
        if let Some(mv) = taken.2 {
            *map.get_mut(mv.key).unwrap() = mv.new_pos;
        }
    }
}
