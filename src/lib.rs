use std::collections::HashMap;
use std::hash::Hash;

mod editable_binary_heap;

use editable_binary_heap::BinaryHeap;
use std::fmt::Debug;
use std::iter::FromIterator;

pub struct KeyedPriorityQueue<TKey, TPriority>
where
    TKey: Hash + Clone + Eq,
    TPriority: Ord,
{
    heap: BinaryHeap<TKey, TPriority>,
    key_to_pos: HashMap<TKey, usize>,
}

impl<TKey: Hash + Clone + Eq, TPriority: Ord> KeyedPriorityQueue<TKey, TPriority> {
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            key_to_pos: HashMap::new(),
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(capacity),
            key_to_pos: HashMap::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, key: TKey, priority: TPriority) {
        // Borrow checker treats borrowing a field as borrowing whole structure
        // so we need to avoid this by getting new references.
        let heap = &mut self.heap;
        let key_to_pos = &mut self.key_to_pos;
        match key_to_pos.get(&key) {
            None => {
                // It will be rewritten during binary heap rebalancing.
                key_to_pos.insert(key.clone(), std::usize::MAX);
                heap.push(key, priority, |changed, pos| {
                    *key_to_pos.get_mut(changed).unwrap() = pos;
                })
            }
            Some(&index) => {
                heap.change_priority(index, priority, |changed, pos| {
                    *key_to_pos.get_mut(changed).unwrap() = pos;
                });
            }
        };
    }

    pub fn pop(&mut self) -> Option<(TKey, TPriority)> {
        let heap = &mut self.heap;
        let key_to_pos = &mut self.key_to_pos;
        let key_priority = heap.pop(|changed, pos| {
            *key_to_pos.get_mut(changed).unwrap() = pos;
        })?;
        key_to_pos.remove(&key_priority.0);
        Some(key_priority)
    }

    pub fn peek(&self) -> Option<(&TKey, &TPriority)> {
        self.heap.peek()
    }

    pub fn get_priority(&self, key: &TKey) -> Option<&TPriority> {
        let index = *self.key_to_pos.get(key)?;
        let (_, priority) = self.heap.look_into(index).unwrap();
        Some(priority)
    }

    pub fn set_priority(&mut self, key: &TKey, priority: TPriority) {
        let index = match self.key_to_pos.get(key) {
            None => panic!("Tried to set_priority with unknown key"),
            Some(&idx) => idx,
        };
        let heap = &mut self.heap;
        let key_to_pos = &mut self.key_to_pos;
        heap.change_priority(index, priority, |changed, pos| {
            *key_to_pos.get_mut(changed).unwrap() = pos;
        });
    }

    pub fn remove_item(&mut self, key: &TKey) -> Option<TPriority> {
        let index = *self.key_to_pos.get(key)?;
        let heap = &mut self.heap;
        let key_to_pos = &mut self.key_to_pos;
        let (_, old_val) = heap
            .remove(index, |changed, pos| {
                *key_to_pos.get_mut(changed).unwrap() = pos;
            })
            .unwrap();
        self.key_to_pos.remove(key);
        Some(old_val)
    }

    pub fn len(&self) -> usize {
        debug_assert_eq!(self.key_to_pos.len(), self.heap.len());
        self.key_to_pos.len()
    }
}

impl<TKey: Hash + Clone + Eq, TPriority: Ord + Clone> Clone
    for KeyedPriorityQueue<TKey, TPriority>
{
    fn clone(&self) -> Self {
        Self {
            heap: self.heap.clone(),
            key_to_pos: self.key_to_pos.clone(),
        }
    }
}

unsafe impl<TKey: Hash + Clone + Eq + Sync, TPriority: Ord + Sync> Sync
    for KeyedPriorityQueue<TKey, TPriority>
{
}

unsafe impl<TKey: Hash + Clone + Eq + Send, TPriority: Ord + Send> Send
    for KeyedPriorityQueue<TKey, TPriority>
{
}

impl<TKey: Hash + Clone + Eq + Debug, TPriority: Ord + Debug> Debug
    for KeyedPriorityQueue<TKey, TPriority>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        self.heap.fmt(f)
    }
}

impl<TKey: Hash + Clone + Eq, TPriority: Ord> FromIterator<(TKey, TPriority)>
    for KeyedPriorityQueue<TKey, TPriority>
{
    fn from_iter<T: IntoIterator<Item = (TKey, TPriority)>>(iter: T) -> Self {
        let heap: BinaryHeap<TKey, TPriority> = iter.into_iter().collect();
        let key_to_pos = heap.generate_mapping();
        Self { heap, key_to_pos }
    }
}

impl<TKey: Hash + Clone + Eq, TPriority: Ord> IntoIterator for KeyedPriorityQueue<TKey, TPriority> {
    type Item = (TKey, TPriority);
    type IntoIter = KeyedPriorityQueueIterator<TKey, TPriority>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter { queue: self }
    }
}

pub struct KeyedPriorityQueueIterator<TKey, TPriority>
where
    TKey: Hash + Clone + Eq,
    TPriority: Ord,
{
    queue: KeyedPriorityQueue<TKey, TPriority>,
}

impl<TKey: Hash + Clone + Eq, TPriority: Ord> Iterator
    for KeyedPriorityQueueIterator<TKey, TPriority>
{
    type Item = (TKey, TPriority);

    fn next(&mut self) -> Option<Self::Item> {
        self.queue.pop()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.queue.len(), Some(self.queue.len()))
    }

    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.queue.len()
    }
}

#[cfg(test)]
mod tests {
    use crate::KeyedPriorityQueue;

    #[test]
    fn test_priority() {
        let mut items = [1, 4, 5, 2, 3];
        let mut queue = KeyedPriorityQueue::<i32, i32>::with_capacity(items.len());
        for (i, &x) in items.iter().enumerate() {
            queue.push(x, x);
            assert_eq!(queue.len(), i + 1);
        }
        assert_eq!(queue.len(), items.len());
        items.sort_unstable_by_key(|&x| -x);
        for &x in items.iter() {
            assert_eq!(queue.pop(), Some((x, x)));
        }
        assert_eq!(queue.pop(), None);
    }

    #[test]
    fn test_peek() {
        let items = [
            ("first", 5),
            ("second", 4),
            ("third", 3),
            ("fourth", 2),
            ("fifth", 1),
        ];

        let mut queue: KeyedPriorityQueue<&str, i32> = items.iter().cloned().collect();

        while queue.len() > 0 {
            let (&key, &priority) = queue.peek().unwrap();
            let (key1, priority1) = queue.pop().unwrap();
            assert_eq!(key, key1);
            assert_eq!(priority, priority1);
        }
        assert_eq!(queue.peek(), None);
    }

    #[test]
    fn test_get_priority() {
        let items = [
            ("first", 5),
            ("second", 4),
            ("third", 3),
            ("fourth", 2),
            ("fifth", 1),
        ];

        let queue: KeyedPriorityQueue<&str, i32> = items.iter().cloned().collect();
        for &(key, priority) in items.iter() {
            let &real = queue.get_priority(&key).unwrap();
            assert_eq!(real, priority);
        }
        let mut queue = queue;
        while let Some(_) = queue.pop() {}
        for &(key, _) in items.iter() {
            assert_eq!(queue.get_priority(&key), None);
        }
    }

    #[test]
    fn test_change_priority() {
        let items = [
            ("first", 5),
            ("second", 4),
            ("third", 3),
            ("fourth", 2),
            ("fifth", 1),
        ];

        let mut queue: KeyedPriorityQueue<&str, i32> = items.iter().cloned().collect();
        let old_priority = *queue.get_priority(&"fifth").unwrap();
        queue.set_priority(&"fifth", old_priority + 10);
        assert_eq!(queue.get_priority(&"fifth"), Some(&11));
        assert_eq!(queue.pop(), Some(("fifth", 11)));

        let old_priority = *queue.get_priority(&"first").unwrap();
        queue.set_priority(&"first", old_priority - 10);
        assert_eq!(queue.get_priority(&"first"), Some(&-5));
        queue.pop();
        queue.pop();
        queue.pop();
        assert_eq!(queue.pop(), Some(("first", -5)));
    }

    #[test]
    fn test_remove_items() {
        let mut items = [1, 4, 5, 2, 3];
        let mut queue: KeyedPriorityQueue<i32, i32> = items.iter().map(|&x| (x, x)).collect();
        queue.remove_item(&3);
        assert_eq!(queue.len(), items.len() - 1);
        items.sort_unstable_by_key(|&x| -x);
        for x in items.iter().cloned().filter(|&x| x != 3) {
            assert_eq!(queue.pop(), Some((x, x)));
        }
        assert_eq!(queue.pop(), None);
    }

    #[test]
    fn test_iteration() {
        let items = [
            ("first", 5),
            ("second", 4),
            ("third", 3),
            ("fourth", 2),
            ("fifth", 1),
        ];

        let queue: KeyedPriorityQueue<&str, i32> = items.iter().rev().cloned().collect();
        let mut iter = queue.into_iter();
        assert_eq!(iter.next(), Some(("first", 5)));
        assert_eq!(iter.next(), Some(("second", 4)));
        assert_eq!(iter.next(), Some(("third", 3)));
        assert_eq!(iter.next(), Some(("fourth", 2)));
        assert_eq!(iter.next(), Some(("fifth", 1)));
        assert_eq!(iter.next(), None);
    }
}
