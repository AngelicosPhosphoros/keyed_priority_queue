# Change Log

## 0.1.3
## Changed
- Removed unsafe implementations of Sync + Send because they are deduced by compiler
- Make some optimizations which reduce timings by 50% but increase memory usage in worst case on 30%
- Added benchmarks

## 2019-11-24: 0.1.2
### Added
- Now items in queue can be looked up borrow using result, e.g. if `String` struct used as key, `&str` can be passed as lookup key.

## 2019-10-27: 0.1.1
### Added
- Now `KeyedPriorityQueue` implements `Default` trait

### Changed
- Some clippy fixes