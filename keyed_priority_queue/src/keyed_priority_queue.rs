use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::fmt::{Debug, Display};
use std::hash::{BuildHasher, Hash};
use std::iter::FromIterator;

use crate::editable_binary_heap::{BinaryHeap, BinaryHeapIterator};
use crate::mediator::{
    Mediator, MediatorEntry, MediatorIndex, OccupiedEntry as MediatorOccupiedEntry,
    VacantEntry as MediatorVacantEntry,
};

/// A priority queue that support lookup by key.
///
/// Bigger `TPriority` values will have more priority.
///
/// It is logic error if priority values changes other way than by [`set_priority`] method.
/// It is logic error if key values changes somehow while in queue.
/// This changes normally possible only through `Cell`, `RefCell`, global state, IO, or unsafe code.
///
/// If you feel KeyedPriorityQueue slow, it can be because it uses RandomState (slightly slow but strong against HashDoS attack) hasher by default.
/// For example, you may try [fnv] or [rustc-hash] crates hashers.
///
/// [`set_priority`]: struct.KeyedPriorityQueue.html#method.set_priority
/// [fnv]: https://crates.io/crates/fnv
/// [rustc-hash]: https://crates.io/crates/rustc-hash
///
/// # Examples
///
/// ## Main example
/// ```
/// use keyed_priority_queue::KeyedPriorityQueue;
///
/// let mut queue = KeyedPriorityQueue::new();
///
/// // Currently queue is empty
/// assert_eq!(queue.peek(), None);
///
/// queue.push("Second", 4);
/// queue.push("Third", 3);
/// queue.push("First", 5);
/// queue.push("Fourth", 2);
/// queue.push("Fifth", 1);
///
/// // Peek return references to most important pair.
/// assert_eq!(queue.peek(), Some((&"First", &5)));
///
/// assert_eq!(queue.len(), 5);
///
/// // We can clone queue if both key and priority is clonable
/// let mut queue_clone = queue.clone();
///
/// // We can run consuming iterator on queue,
/// // and it will return items in decreasing order
/// for (key, priority) in queue_clone {
///     println!("Priority of key {} is {}", key, priority);
/// }
///
/// // Popping always will return the biggest element
/// assert_eq!(queue.pop(), Some(("First", 5)));
/// // We can change priority of item by key:
/// queue.set_priority(&"Fourth", 10);
/// // And get it
/// assert_eq!(queue.get_priority(&"Fourth"), Some(&10));
/// // Now biggest element is Fourth
/// assert_eq!(queue.pop(), Some(("Fourth", 10)));
/// // We can also decrease priority!
/// queue.set_priority(&"Second", -1);
/// assert_eq!(queue.pop(), Some(("Third", 3)));
/// assert_eq!(queue.pop(), Some(("Fifth", 1)));
/// assert_eq!(queue.pop(), Some(("Second", -1)));
/// // Now queue is empty
/// assert_eq!(queue.pop(), None);
///
/// // We can clear queue
/// queue.clear();
/// assert!(queue.is_empty());
/// ```
///
/// ## Partial ord queue
///
/// If you need to use float values (which don't implement Ord) as priority,
/// you can use some wrapper that implement it:
///
/// ```
/// use keyed_priority_queue::KeyedPriorityQueue;
/// use std::cmp::{Ord, Ordering, Eq, PartialEq, PartialOrd};
///
/// #[derive(Debug)]
/// struct OrdFloat(f32);
///
/// impl PartialOrd for OrdFloat {
///     fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(&other)) }
/// }
///
/// impl Eq for OrdFloat {}
///
/// impl PartialEq for OrdFloat {
///     fn eq(&self, other: &Self) -> bool { self.cmp(&other) == Ordering::Equal }
/// }
///
/// impl Ord for OrdFloat {
///     fn cmp(&self, other: &Self) -> Ordering {
///         self.0.partial_cmp(&other.0)
///             .unwrap_or(if self.0.is_nan() && other.0.is_nan() {
///                 Ordering::Equal
///             } else if self.0.is_nan() {
///                 Ordering::Less
///             } else { Ordering::Greater })
///     }
/// }
///
/// let mut queue = KeyedPriorityQueue::new();
/// queue.push(5, OrdFloat(5.0));
/// queue.push(4, OrdFloat(4.0));
/// assert_eq!(queue.pop(), Some((5, OrdFloat(5.0))));
/// assert_eq!(queue.pop(), Some((4, OrdFloat(4.0))));
/// assert_eq!(queue.pop(), None);
/// ```
#[derive(Clone)]
pub struct KeyedPriorityQueue<TKey, TPriority, S = RandomState>
where
    TKey: Hash + Eq,
    TPriority: Ord,
    S: BuildHasher,
{
    heap: BinaryHeap<TPriority>,
    key_to_pos: Mediator<TKey, S>,
}

impl<TKey: Hash + Eq, TPriority: Ord> KeyedPriorityQueue<TKey, TPriority, RandomState> {
    /// Creates an empty queue
    ///
    /// ### Examples
    ///
    ///
    /// ```
    /// use keyed_priority_queue::KeyedPriorityQueue;
    /// let mut queue = KeyedPriorityQueue::new();
    /// queue.push("Key", 4);
    /// ```
    #[inline]
    pub fn new() -> Self {
        Self::with_capacity_and_hasher(0, RandomState::default())
    }

    /// Creates an empty queue with allocated memory enough
    /// to keep `capacity` elements without reallocation.
    ///
    /// ### Examples
    ///
    ///
    /// ```
    /// use keyed_priority_queue::KeyedPriorityQueue;
    /// let mut queue = KeyedPriorityQueue::with_capacity(10);
    /// queue.push("Key", 4);
    /// ```
    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_and_hasher(capacity, RandomState::default())
    }
}

impl<TKey: Hash + Eq, TPriority: Ord, S: BuildHasher> KeyedPriorityQueue<TKey, TPriority, S> {
    /// Creates an empty queue with specific Hasher
    ///
    /// ### Examples
    ///
    ///
    /// ```
    /// use keyed_priority_queue::KeyedPriorityQueue;
    /// use std::collections::hash_map::RandomState;
    /// let mut queue = KeyedPriorityQueue::with_hasher(RandomState::default());
    /// queue.push("Key", 4);
    /// ```
    #[inline]
    pub fn with_hasher(hasher: S) -> Self {
        Self::with_capacity_and_hasher(0, hasher)
    }

    /// Creates an empty queue with allocated memory enough
    /// to keep `capacity` elements without reallocation.
    /// Also useful when Hasher cannot be defaulted.
    ///
    /// ### Examples
    ///
    ///
    /// ```
    /// use keyed_priority_queue::KeyedPriorityQueue;
    /// use std::collections::hash_map::RandomState;
    /// let mut queue = KeyedPriorityQueue::with_capacity_and_hasher(10, RandomState::default());
    /// queue.push("Key", 4);
    /// ```
    #[inline]
    pub fn with_capacity_and_hasher(capacity: usize, hasher: S) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(capacity),
            key_to_pos: Mediator::with_capacity_and_hasher(capacity, hasher),
        }
    }

    /// Reserves space for at least `additional` new elements.
    ///
    /// ### Panics
    ///
    /// Panics if the new capacity overflows `usize`.
    ///
    /// ### Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use keyed_priority_queue::KeyedPriorityQueue;
    /// let mut queue = KeyedPriorityQueue::new();
    /// queue.reserve(100);
    /// queue.push(4, 4);
    /// ```
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.heap.reserve(additional);
        self.key_to_pos.reserve(additional);
    }

    /// Adds new element to queue if missing key or replace its priority if key exists.
    /// In second case doesn't replace key.
    ///
    /// ### Examples
    ///
    ///
    /// ```
    /// use keyed_priority_queue::KeyedPriorityQueue;
    /// let mut queue = KeyedPriorityQueue::new();
    /// queue.push("First", 5);
    /// assert_eq!(queue.peek(), Some((&"First", &5)));
    /// queue.push("First", 10);
    /// assert_eq!(queue.peek(), Some((&"First", &10)));
    /// ```
    ///
    /// ### Time complexity
    ///
    /// Average complexity is ***O(log n)***
    /// If elements pushed in descending order, amortized complexity is ***O(1)***.
    ///
    /// The worst case is when reallocation appears.
    /// In this case complexity of single call is ***O(n)***.
    pub fn push(&mut self, key: TKey, priority: TPriority) -> Option<TPriority> {
        match self.entry(key) {
            Entry::Vacant(entry) => {
                entry.set_priority(priority);
                None
            }
            Entry::Occupied(entry) => Some(entry.set_priority(priority)),
        }
    }

    /// Remove and return item with the maximal priority.
    ///
    /// ### Examples
    ///
    ///
    /// ```
    /// use keyed_priority_queue::KeyedPriorityQueue;
    /// let mut queue: KeyedPriorityQueue<i32, i32> = (0..5).map(|x|(x,x)).collect();
    /// assert_eq!(queue.pop(), Some((4,4)));
    /// assert_eq!(queue.pop(), Some((3,3)));
    /// assert_eq!(queue.pop(), Some((2,2)));
    /// assert_eq!(queue.pop(), Some((1,1)));
    /// assert_eq!(queue.pop(), Some((0,0)));
    /// ```
    ///
    /// ### Time complexity
    ///
    /// Cost of pop is always ***O(log n)***
    pub fn pop(&mut self) -> Option<(TKey, TPriority)> {
        let (to_remove, _) = self.heap.most_prioritized_idx()?;
        Some(self.remove_internal(to_remove))
    }

    /// Get reference to the pair with the maximal priority.
    ///
    /// ### Examples
    ///
    ///
    /// ```
    /// use keyed_priority_queue::KeyedPriorityQueue;
    /// let mut queue: KeyedPriorityQueue<i32, i32> = (0..5).map(|x|(x,x)).collect();
    /// assert_eq!(queue.peek(), Some((&4, &4)));
    /// ```
    ///
    /// ### Time complexity
    ///
    /// Always ***O(1)***
    pub fn peek(&self) -> Option<(&TKey, &TPriority)> {
        let (first_idx, heap_idx) = self.heap.most_prioritized_idx()?;
        let (key, _) = self.key_to_pos.get_index(first_idx);
        let (_, priority) = self
            .heap
            .look_into(heap_idx)
            .expect("Checked using key_to_pos");
        Some((key, priority))
    }

    /// Gets the given key's corresponding entry in the map for in-place manipulation.
    ///
    /// ## Time complexity
    /// Amortized ***O(1)***, uses only one hash lookup
    pub fn entry(&mut self, key: TKey) -> Entry<TKey, TPriority, S> {
        // Borrow checker treats borrowing a field as borrowing whole structure
        // so we need to get references to fields to borrow them individually.
        let key_to_pos = &mut self.key_to_pos;
        let heap = &mut self.heap;

        match key_to_pos.entry(key) {
            MediatorEntry::Vacant(internal_entry) => Entry::Vacant(VacantEntry {
                internal_entry,
                heap,
            }),
            MediatorEntry::Occupied(internal_entry) => Entry::Occupied(OccupiedEntry {
                internal_entry,
                heap,
            }),
        }
    }

    /// Get reference to the priority by key.
    ///
    /// ### Examples
    ///
    ///
    /// ```
    /// use keyed_priority_queue::KeyedPriorityQueue;
    /// let mut queue: KeyedPriorityQueue<&str, i32> = [("first", 0), ("second", 1), ("third", 2)]
    ///                             .iter().cloned().collect();
    /// assert_eq!(queue.get_priority(&"second"), Some(&1));
    /// ```
    ///
    /// ### Time complexity
    ///
    /// ***O(1)*** in average (limited by hash map key lookup).
    pub fn get_priority<Q>(&self, key: &Q) -> Option<&TPriority>
    where
        TKey: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let heap_idx = self.key_to_pos.get(key)?;
        Some(
            self.heap
                .look_into(heap_idx)
                .expect("Must contain if key_to_pos contain")
                .1,
        )
    }

    /// Set new priority for existing key and reorder the queue.
    /// Returns old priority if succeeds or [`SetPriorityNotFoundError`].
    ///
    /// ### Examples
    ///
    ///
    /// ```
    /// use keyed_priority_queue::{KeyedPriorityQueue, SetPriorityNotFoundError};
    /// let mut queue: KeyedPriorityQueue<&str, i32> = [("first", 0), ("second", 1), ("third", 2)]
    ///                             .iter().cloned().collect();
    /// assert_eq!(queue.set_priority(&"second", 5), Ok(1));
    /// assert_eq!(queue.get_priority(&"second"), Some(&5));
    /// assert_eq!(queue.pop(), Some(("second", 5)));
    /// assert_eq!(queue.set_priority(&"Missing", 5), Err(SetPriorityNotFoundError{}));
    /// ```
    ///
    /// ### Time complexity
    ///
    /// In best case ***O(1)***, in average costs ***O(log n)***.
    ///
    /// [`SetPriorityNotFoundError`]: struct.SetPriorityNotFoundError.html
    #[inline]
    pub fn set_priority<Q>(
        &mut self,
        key: &Q,
        priority: TPriority,
    ) -> Result<TPriority, SetPriorityNotFoundError>
    where
        TKey: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let map_pos = match self.key_to_pos.get_full(key) {
            None => return Err(SetPriorityNotFoundError {}),
            Some((idx, _, _)) => idx,
        };

        Ok(self.set_priority_internal(map_pos, priority))
    }

    /// Allow removing item by key.
    /// Returns priority if succeeds.
    ///
    /// ### Examples
    ///
    ///
    /// ```
    /// use keyed_priority_queue::KeyedPriorityQueue;
    /// let mut queue: KeyedPriorityQueue<i32, i32> = (0..5).map(|x|(x,x)).collect();
    /// assert_eq!(queue.remove(&2), Some(2));
    /// assert_eq!(queue.pop(), Some((4,4)));
    /// assert_eq!(queue.pop(), Some((3,3)));
    /// // There is no 2
    /// assert_eq!(queue.pop(), Some((1,1)));
    /// assert_eq!(queue.pop(), Some((0,0)));
    /// assert_eq!(queue.remove(&10), None);
    /// ```
    ///
    /// ### Time complexity
    ///
    /// On average the function will require ***O(log n)*** operations.
    #[inline]
    pub fn remove<Q>(&mut self, key: &Q) -> Option<TPriority>
    where
        TKey: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let (_, priority) = self.remove_entry(key)?;
        Some(priority)
    }

    /// Allow removing item by key.
    /// Returns key and priority if succeeds.
    ///
    /// ### Examples
    ///
    ///
    /// ```
    /// use keyed_priority_queue::KeyedPriorityQueue;
    /// let mut queue: KeyedPriorityQueue<i32, i32> = (0..5).map(|x|(x,x)).collect();
    /// assert_eq!(queue.remove_entry(&2), Some((2, 2)));
    /// assert_eq!(queue.pop(), Some((4,4)));
    /// assert_eq!(queue.pop(), Some((3,3)));
    /// // There is no 2
    /// assert_eq!(queue.pop(), Some((1,1)));
    /// assert_eq!(queue.pop(), Some((0,0)));
    /// assert_eq!(queue.remove_entry(&10), None);
    /// ```
    ///
    /// ### Time complexity
    ///
    /// On average the function will require ***O(log n)*** operations.
    #[inline]
    pub fn remove_entry<Q>(&mut self, key: &Q) -> Option<(TKey, TPriority)>
    where
        TKey: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let (index, _, _) = self.key_to_pos.get_full(key)?;
        Some(self.remove_internal(index))
    }

    /// Get the number of elements in queue.
    ///
    /// ### Examples
    ///
    ///
    /// ```
    /// use keyed_priority_queue::KeyedPriorityQueue;
    /// let queue: KeyedPriorityQueue<i32, i32> = (0..5).map(|x|(x,x)).collect();
    /// assert_eq!(queue.len(), 5);
    /// ```
    ///
    /// ### Time complexity
    ///
    /// Always ***O(1)***
    #[inline]
    pub fn len(&self) -> usize {
        debug_assert_eq!(self.key_to_pos.len(), self.heap.usize_len());
        self.key_to_pos.len()
    }

    /// Returns true if queue is empty.
    ///
    /// ```
    /// let mut queue = keyed_priority_queue::KeyedPriorityQueue::new();
    /// assert!(queue.is_empty());
    /// queue.push(0,5);
    /// assert!(!queue.is_empty());
    /// ```
    ///
    /// ### Time complexity
    ///
    /// Always ***O(1)***
    #[inline]
    pub fn is_empty(&self) -> bool {
        debug_assert_eq!(self.heap.is_empty(), self.key_to_pos.is_empty());
        self.key_to_pos.is_empty()
    }

    /// Make the queue empty.
    ///
    /// ```
    /// use keyed_priority_queue::KeyedPriorityQueue;
    /// let mut queue: KeyedPriorityQueue<i32, i32> = (0..5).map(|x|(x,x)).collect();
    /// assert!(!queue.is_empty());
    /// queue.clear();
    /// assert!(queue.is_empty());
    /// ```
    ///
    /// ### Time complexity
    ///
    /// Always ***O(n)***
    #[inline]
    pub fn clear(&mut self) {
        self.heap.clear();
        self.key_to_pos.clear();
    }

    /// Create readonly borrowing iterator over heap
    ///
    /// ```
    /// use keyed_priority_queue::KeyedPriorityQueue;
    /// use std::collections::HashMap;
    /// let queue: KeyedPriorityQueue<i32, i32> = (0..5).map(|x|(x,x)).collect();
    /// let mut entries = HashMap::new();
    /// for (&key, &priority) in queue.iter(){
    ///     entries.insert(key, priority);
    /// }
    /// let second_map: HashMap<i32, i32> = (0..5).map(|x|(x,x)).collect();
    /// assert_eq!(entries, second_map);
    /// ```
    ///
    /// ### Time complexity
    ///
    /// Iterating over whole queue is ***O(n)***
    pub fn iter(&self) -> KeyedPriorityQueueBorrowIter<TKey, TPriority, S> {
        KeyedPriorityQueueBorrowIter {
            key_to_pos: &self.key_to_pos,
            heap_iterator: self.heap.iter(),
        }
    }

    // Removes entry from by index of map
    fn remove_internal(&mut self, position: MediatorIndex) -> (TKey, TPriority) {
        // Borrow checker treats borrowing a field as borrowing whole structure
        // so we need to get references to fields to borrow them individually.
        let key_to_pos = &mut self.key_to_pos;
        let heap = &mut self.heap;

        let (_, heap_to_rem) = key_to_pos.get_index(position);

        let (removed_idx, priority) = heap
            .remove(heap_to_rem, |index, heap_idx| {
                *key_to_pos.get_index_mut(index) = heap_idx
            })
            .expect("Checked by key_to_pos");
        debug_assert_eq!(position, removed_idx);

        let (removed_key, _) = key_to_pos.swap_remove_index(position);
        if MediatorIndex(key_to_pos.len()) != removed_idx {
            let (_, heap_idx_of_moved) = key_to_pos.get_index(removed_idx);
            heap.change_outer_pos(removed_idx, heap_idx_of_moved);
        }

        (removed_key, priority)
    }

    // Do O(log n) heap updates and by-index map changes
    fn set_priority_internal(&mut self, position: MediatorIndex, priority: TPriority) -> TPriority {
        // Borrow checker treats borrowing a field as borrowing whole structure
        // so we need to get references to fields to borrow them individually.
        let heap = &mut self.heap;
        let key_to_pos = &mut self.key_to_pos;

        let (_, heap_idx) = key_to_pos.get_index(position);

        heap.change_priority(heap_idx, priority, |index, heap_idx| {
            *key_to_pos.get_index_mut(index) = heap_idx
        })
    }
}

/// A view into a single entry in a queue, which may either be vacant or occupied.
///
/// This `enum` is constructed from the [`entry`] method on [`KeyedPriorityQueue`].
///
/// [`KeyedPriorityQueue`]: struct.KeyedPriorityQueue.html
/// [`entry`]: struct.KeyedPriorityQueue.html#method.entry
pub enum Entry<'a, TKey: Eq + Hash, TPriority: Ord, S: BuildHasher> {
    /// An occupied entry.
    Occupied(OccupiedEntry<'a, TKey, TPriority, S>),

    /// A vacant entry.
    Vacant(VacantEntry<'a, TKey, TPriority, S>),
}

/// A view into an occupied entry in a [`KeyedPriorityQueue`].
/// It is part of the [`Entry`] enum.
///
/// [`Entry`]: enum.Entry.html
/// [`KeyedPriorityQueue`]: struct.KeyedPriorityQueue.html
pub struct OccupiedEntry<'a, TKey, TPriority, S = RandomState>
where
    TKey: 'a + Eq + Hash,
    TPriority: 'a + Ord,
    S: BuildHasher,
{
    internal_entry: MediatorOccupiedEntry<'a, TKey, S>,
    heap: &'a mut BinaryHeap<TPriority>,
}

impl<'a, TKey, TPriority, S> OccupiedEntry<'a, TKey, TPriority, S>
where
    TKey: 'a + Eq + Hash,
    TPriority: 'a + Ord,
    S: BuildHasher,
{
    /// Returns reference to the priority associated to entry
    ///
    /// ## Time complexity
    /// ***O(1)*** instant access
    #[inline]
    pub fn get_priority(&self) -> &TPriority {
        let heap_idx = self.internal_entry.get_heap_idx();
        self.heap.look_into(heap_idx).expect("Must be in queue").1
    }

    /// Changes priority of key and returns old priority
    ///
    /// ## Time complexity
    /// Up to ***O(log n)*** operations in worst case
    /// ***O(1)*** in best case
    #[inline]
    pub fn set_priority(mut self, priority: TPriority) -> TPriority {
        let heap_idx = self.internal_entry.get_heap_idx();

        let heap = &mut self.heap;
        let key_to_pos = unsafe {
            // Safety: reference used only inside the method and never leaked away
            // This method can be called only when Mediator field alive along with queue itself.
            self.internal_entry.transform_to_map()
        };

        heap.change_priority(heap_idx, priority, |index, heap_idx| {
            *key_to_pos.get_index_mut(index) = heap_idx;
        })
    }

    /// Get the reference to actual key
    ///
    /// ## Time complexity
    /// ***O(1)*** instant access
    #[inline]
    pub fn get_key(&self) -> &TKey {
        self.internal_entry.get_key()
    }

    /// Remove entry from queue
    ///
    /// ## Time complexity
    /// Up to ***O(log n)*** operations
    pub fn remove(mut self) -> (TKey, TPriority) {
        let heap_idx = self.internal_entry.get_heap_idx();
        // Look `Mediator` `entry` method
        let key_to_pos = unsafe {
            // Safety: reference used only inside the method and never leaked away
            // This method can be called only when Mediator field alive along with queue itself.
            self.internal_entry.transform_to_map()
        };
        let heap = &mut self.heap;

        let (removed_idx, priority) = heap
            .remove(heap_idx, |index, heap_idx| {
                *key_to_pos.get_index_mut(index) = heap_idx
            })
            .expect("Checked by key_to_pos");

        let (removed_key, _) = key_to_pos.swap_remove_index(removed_idx);
        if MediatorIndex(key_to_pos.len()) != removed_idx {
            let (_, heap_idx_of_moved) = key_to_pos.get_index(removed_idx);
            heap.change_outer_pos(removed_idx, heap_idx_of_moved);
        }

        (removed_key, priority)
    }
}

/// A view into a vacant entry in a [`KeyedPriorityQueue`].
/// It is part of the [`Entry`] enum.
///
/// [`Entry`]: enum.Entry.html
/// [`KeyedPriorityQueue`]: struct.KeyedPriorityQueue.html
pub struct VacantEntry<'a, TKey, TPriority, S = RandomState>
where
    TKey: 'a + Eq + Hash,
    TPriority: 'a + Ord,
    S: BuildHasher,
{
    internal_entry: MediatorVacantEntry<'a, TKey, S>,
    heap: &'a mut BinaryHeap<TPriority>,
}

impl<'a, TKey, TPriority, S> VacantEntry<'a, TKey, TPriority, S>
where
    TKey: 'a + Eq + Hash,
    TPriority: 'a + Ord,
    S: BuildHasher,
{
    /// Insert priority of key to queue
    ///
    /// ## Time complexity
    /// Up to ***O(log n)*** operations
    #[inline]
    pub fn set_priority(self, priority: TPriority) {
        let heap = self.heap;
        let internal_entry = self.internal_entry;
        let (key_to_pos, mediator_index) = unsafe {
            // Safety: reference used only inside the method and never leaked away
            // This method can be called only when Mediator field alive along with queue itself.
            internal_entry.insert(heap.len())
        };
        heap.push(mediator_index, priority, |index, val| {
            *key_to_pos.get_index_mut(index) = val
        });
    }

    /// Get the reference to actual key
    ///
    /// ## Time complexity
    /// ***O(1)*** instant access
    #[inline]
    pub fn get_key(&self) -> &TKey {
        self.internal_entry.get_key()
    }
}

impl<TKey: Hash + Eq + Debug, TPriority: Ord + Debug, S: BuildHasher> Debug
    for KeyedPriorityQueue<TKey, TPriority, S>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "[")?;
        for entry in self.iter() {
            write!(f, "{:?}", entry)?;
        }
        write!(f, "]")
    }
}

impl<TKey: Hash + Eq, TPriority: Ord, S: BuildHasher + Default> Default
    for KeyedPriorityQueue<TKey, TPriority, S>
{
    #[inline]
    fn default() -> Self {
        Self::with_capacity_and_hasher(0, S::default())
    }
}

impl<TKey: Hash + Eq, TPriority: Ord, S: BuildHasher + Default> FromIterator<(TKey, TPriority)>
    for KeyedPriorityQueue<TKey, TPriority, S>
{
    /// Allows building queue from iterator using `collect()`.
    /// At result it will be valid queue with unique keys.
    ///
    /// ### Examples
    ///
    ///
    /// ```
    /// use keyed_priority_queue::KeyedPriorityQueue;
    /// let mut queue: KeyedPriorityQueue<&str, i32> =
    /// [("first", 0), ("second", 1), ("third", 2), ("first", -1)]
    ///                             .iter().cloned().collect();
    /// assert_eq!(queue.pop(), Some(("third", 2)));
    /// assert_eq!(queue.pop(), Some(("second", 1)));
    /// assert_eq!(queue.pop(), Some(("first", -1)));
    /// assert_eq!(queue.pop(), None);
    /// ```
    ///
    /// ### Time complexity
    ///
    /// ***O(n log n)*** in average.
    fn from_iter<T: IntoIterator<Item = (TKey, TPriority)>>(iter: T) -> Self {
        let (heap, key_to_pos) = BinaryHeap::produce_from_iter_hash(iter);
        Self { heap, key_to_pos }
    }
}

impl<TKey: Hash + Eq, TPriority: Ord> IntoIterator for KeyedPriorityQueue<TKey, TPriority> {
    type Item = (TKey, TPriority);
    type IntoIter = KeyedPriorityQueueIterator<TKey, TPriority>;

    /// Make iterator that return items in descending order.
    ///
    /// ### Examples
    ///
    ///
    /// ```
    /// use keyed_priority_queue::KeyedPriorityQueue;
    /// let mut queue: KeyedPriorityQueue<&str, i32> =
    ///     [("first", 0), ("second", 1), ("third", 2)]
    ///                             .iter().cloned().collect();
    /// let mut iterator = queue.into_iter();
    /// assert_eq!(iterator.next(), Some(("third", 2)));
    /// assert_eq!(iterator.next(), Some(("second", 1)));
    /// assert_eq!(iterator.next(), Some(("first", 0)));
    /// assert_eq!(iterator.next(), None);
    /// ```
    ///
    /// ### Time complexity
    ///
    /// ***O(n log n)*** for iteration.
    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter { queue: self }
    }
}

/// This is consuming iterator that returns elements in decreasing order
///
/// ### Time complexity
/// Overall complexity of iteration is ***O(n log n)***
pub struct KeyedPriorityQueueIterator<TKey, TPriority, S = RandomState>
where
    TKey: Hash + Eq,
    TPriority: Ord,
    S: BuildHasher,
{
    queue: KeyedPriorityQueue<TKey, TPriority, S>,
}

impl<TKey: Hash + Eq, TPriority: Ord, S: BuildHasher> Iterator
    for KeyedPriorityQueueIterator<TKey, TPriority, S>
{
    type Item = (TKey, TPriority);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.queue.pop()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.queue.len(), Some(self.queue.len()))
    }

    #[inline]
    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.queue.len()
    }
}

/// This is unordered borrowing iterator over queue.
///
/// ### Time complexity
/// Overall complexity of iteration is ***O(n)***
pub struct KeyedPriorityQueueBorrowIter<'a, TKey, TPriority, S = RandomState>
where
    TKey: 'a + Hash + Eq,
    TPriority: 'a,
    S: BuildHasher,
{
    heap_iterator: BinaryHeapIterator<'a, TPriority>,
    key_to_pos: &'a Mediator<TKey, S>,
}

impl<'a, TKey: 'a + Hash + Eq, TPriority: 'a, S: BuildHasher> Iterator
    for KeyedPriorityQueueBorrowIter<'a, TKey, TPriority, S>
{
    type Item = (&'a TKey, &'a TPriority);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let heap_iterator = &mut self.heap_iterator;
        let key_to_pos = &self.key_to_pos;
        heap_iterator.next().map(|(index, priority)| {
            let (key, _) = key_to_pos.get_index(index);
            (key, priority)
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.heap_iterator.size_hint()
    }

    #[inline]
    fn count(self) -> usize
    where
        Self: Sized,
    {
        self.heap_iterator.count()
    }
}

/// This is error type for [`set_priority`] method of [`KeyedPriorityQueue`].
/// It means that queue doesn't contain such key.
///
/// [`KeyedPriorityQueue`]: struct.KeyedPriorityQueue.html
/// [`set_priority`]: struct.KeyedPriorityQueue.html#method.set_priority
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Copy, Clone, Default)]
pub struct SetPriorityNotFoundError;

impl Display for SetPriorityNotFoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(f, "Key not found in KeyedPriorityQueue during set_priority")
    }
}

impl std::error::Error for SetPriorityNotFoundError {}

#[cfg(test)]
mod tests {
    use super::KeyedPriorityQueue;

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
        assert_eq!(
            queue.set_priority(&"HELLO", 64),
            Err(super::SetPriorityNotFoundError::default())
        );
        let old_priority = *queue.get_priority(&"fifth").unwrap();
        assert_eq!(queue.set_priority(&"fifth", old_priority + 10), Ok(1));
        assert_eq!(queue.get_priority(&"fifth"), Some(&11));
        assert_eq!(queue.pop(), Some(("fifth", 11)));

        let old_priority = *queue.get_priority(&"first").unwrap();
        assert_eq!(queue.set_priority(&"first", old_priority - 10), Ok(5));
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
        assert_eq!(queue.remove_entry(&3), Some((3, 3)));
        assert_eq!(queue.remove_entry(&20), None);
        assert_eq!(queue.len(), items.len() - 1);
        assert_eq!(queue.get_priority(&3), None);
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

    #[test]
    fn test_multiple_push() {
        let mut queue = KeyedPriorityQueue::new();
        queue.push(0, 1);
        assert_eq!(queue.peek(), Some((&0, &1)));
        queue.push(0, 5);
        assert_eq!(queue.peek(), Some((&0, &5)));
        queue.push(0, 7);
        assert_eq!(queue.peek(), Some((&0, &7)));
        queue.push(0, 9);
        assert_eq!(queue.peek(), Some((&0, &9)));
    }

    #[test]
    fn test_borrow_keys() {
        let mut queue: KeyedPriorityQueue<String, i32> = KeyedPriorityQueue::new();
        queue.push("Hello".to_string(), 5);
        let string = "Hello".to_string();
        let string_ref: &String = &string;
        let str_ref: &str = &string;
        assert_eq!(queue.get_priority(string_ref), Some(&5));
        assert_eq!(queue.get_priority(str_ref), Some(&5));
    }

    #[test]
    fn test_entry_vacant() {
        use super::Entry;

        let items = [
            ("first", 5i32),
            ("second", 4),
            ("third", 3),
            ("fourth", 2),
            ("fifth", 1),
        ];

        let mut queue = KeyedPriorityQueue::new();

        for &(k, v) in items.iter() {
            queue.push(k, v);
        }

        assert_eq!(queue.len(), 5);
        match queue.entry("Cotton") {
            Entry::Occupied(_) => unreachable!(),
            Entry::Vacant(entry) => {
                assert_eq!(entry.get_key(), &"Cotton");
                entry.set_priority(10);
            }
        };
        assert_eq!(queue.len(), 6);
        assert_eq!(queue.get_priority(&"Cotton"), Some(&10));
        match queue.entry("Cotton") {
            Entry::Occupied(entry) => {
                assert_eq!(entry.get_key(), &"Cotton");
                assert_eq!(entry.get_priority(), &10);
            }
            Entry::Vacant(_) => unreachable!(),
        };
    }

    #[test]
    fn test_entry_occupied() {
        use super::Entry;

        let items = [
            ("first", 5i32),
            ("second", 4),
            ("third", 3),
            ("fourth", 2),
            ("fifth", 1),
        ];

        let mut queue = KeyedPriorityQueue::new();

        for &(k, v) in items.iter() {
            queue.push(k, v);
        }

        assert_eq!(queue.len(), 5);
        match queue.entry("third") {
            Entry::Occupied(entry) => {
                assert_eq!(entry.get_key(), &"third");
                assert_eq!(entry.get_priority(), &3);
                assert_eq!(entry.set_priority(5), 3);
            }
            Entry::Vacant(_) => unreachable!(),
        };
        assert_eq!(queue.len(), 5);
        assert_eq!(queue.get_priority(&"third"), Some(&5));
        match queue.entry("third") {
            Entry::Occupied(entry) => {
                assert_eq!(entry.remove(), ("third", 5));
            }
            Entry::Vacant(_) => unreachable!(),
        };

        assert_eq!(queue.len(), 4);
        assert_eq!(queue.get_priority(&"third"), None);
    }

    #[test]
    fn test_borrow_iter() {
        use std::collections::HashMap;
        let items = [
            ("first", 5i32),
            ("third", 3),
            ("second", 4),
            ("fifth", 1),
            ("fourth", 2),
        ];

        let queue: KeyedPriorityQueue<String, i32> =
            items.iter().map(|&(k, p)| (k.to_owned(), p)).collect();

        let mut map: HashMap<&str, i32> = HashMap::new();

        let mut total_items = 0;
        for (key, &value) in queue.iter() {
            map.insert(key, value);
            total_items += 1;
        }
        assert_eq!(items.len(), total_items);
        assert_eq!(queue.len(), items.len());
        let other_map: HashMap<_, _> = items.iter().cloned().collect();
        assert_eq!(map, other_map);
    }

    #[test]
    fn test_sync() {
        fn assert_sync<T: Sync>() {}
        assert_sync::<KeyedPriorityQueue<i32, i32>>();
    }

    #[test]
    fn test_send() {
        fn assert_sync<T: Send>() {}
        assert_sync::<KeyedPriorityQueue<i32, i32>>();
    }

    #[test]
    fn test_fmt() {
        let items = [
            ("first", 5i32),
            ("second", 4),
            ("third", 3),
            ("fourth", 2),
            ("fifth", 1),
        ];

        let queue: KeyedPriorityQueue<&str, i32> = items.iter().cloned().collect();

        assert_eq!(
            format!("{:?}", queue),
            "[(\"first\", 5)(\"second\", 4)(\"third\", 3)(\"fourth\", 2)(\"fifth\", 1)]"
        );
    }

    #[test]
    fn test_not_clone_works() {
        use core::hash::Hash;
        #[derive(Hash, PartialEq, Eq)]
        struct Key(u32);

        let vals = [0u32, 1, 1, 2, 4, 5];
        let mut queue: KeyedPriorityQueue<Key, u32> =
            vals.iter().copied().map(|v| (Key(v), v)).collect();
        queue.set_priority(&Key(1), 10).unwrap();
        let mut res = Vec::with_capacity(5);
        while let Some((Key(k), p)) = queue.pop() {
            res.push((k, p));
        }
        assert_eq!(&res, &[(1, 10), (5, 5), (4, 4), (2, 2), (0, 0)]);
    }

    #[test]
    fn test_remove_change_tree() {
        use std::cmp::Reverse;
        let mut queue = KeyedPriorityQueue::new();

        queue.push(0, Reverse(300));
        queue.push(1, Reverse(500));
        queue.push(2, Reverse(400));
        queue.push(3, Reverse(400));
        queue.push(4, Reverse(600));
        queue.push(5, Reverse(100));
        queue.push(6, Reverse(200));
        queue.remove(&1);

        let mut list = Vec::new();
        while let Some((_, timestamp)) = queue.pop() {
            list.push(timestamp.0);
        }

        assert_eq!(list, [100, 200, 300, 400, 400, 600])
    }
}
