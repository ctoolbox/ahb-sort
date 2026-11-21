# AHB Sort: An Optimized Bucket Sort Implementation 🚀

A pet project focused on optimizing the classic bucket sort algorithm through various techniques and improvements. The result is AHB (Adaptive Hybrid Bucket) Sort, a sort for `int` arrays that is much faster than the C library's `qsort` and about twice as fast as the bucket sort it started from.

## Performance Highlights 📊

Geometric mean over 6 input patterns and 5 array sizes (1,000 to 5,000,000 elements):

- 12.3x faster than `qsort`
- 2.4x faster than the original bucket sort
- 1.6x faster than a byte-wise LSD radix sort on average, but **slower than it on wide, evenly spread data** (see below)

Measured on an Apple M3 Max with Apple clang 21, `-O3 -march=native`. Numbers will differ on other machines.

## Detailed Results (5M Elements) 📊

Best of 7 runs, in milliseconds. Lower is better.

| Input | AHB Sort | Original bucket | `qsort` | Radix (base 256) |
|---|---|---|---|---|
| Random, range 1000 | 1.71 | 16.77 | 174.07 | 15.97 |
| Random, range 1e9 | 42.61 | 51.38 | 428.06 | 14.93 |
| Nearly sorted | 22.96 | 39.58 | 171.35 | 38.83 |
| 100 distinct values | 7.11 | 16.55 | 106.83 | 34.98 |
| Exponential | 47.97 | 95.35 | 398.67 | 17.10 |
| Cluster + 1% outliers | 21.85 | 45.24 | 176.76 | 34.18 |

How many times faster AHB Sort is (below 1x means AHB Sort is slower):

| Input | vs original bucket | vs `qsort` | vs radix |
|---|---|---|---|
| Random, range 1000 | 9.8x | 101.9x | 9.3x |
| Random, range 1e9 | 1.2x | 10.0x | 0.35x |
| Nearly sorted | 1.7x | 7.5x | 1.7x |
| 100 distinct values | 2.3x | 15.0x | 4.9x |
| Exponential | 2.0x | 8.3x | 0.36x |
| Cluster + 1% outliers | 2.1x | 8.1x | 1.6x |

## Results by Array Size 📈

Geometric mean speedup of AHB Sort over the 6 input patterns:

| Elements | vs original bucket | vs `qsort` | vs radix |
|---|---|---|---|
| 1,000 | 2.2x | 6.8x | 1.7x |
| 10,000 | 2.5x | 14.0x | 1.8x |
| 100,000 | 2.5x | 14.6x | 1.6x |
| 1,000,000 | 2.4x | 14.1x | 1.5x |
| 5,000,000 | 2.4x | 14.1x | 1.6x |

## Performance Summary 🎯

1. **Where it shines**
   - Narrow value ranges: when the range is at most twice the array size, AHB Sort switches to a counting sort (about 100x faster than `qsort` at 5M elements)
   - Few distinct values, clustered data and nearly sorted data: 1.4x to 5.3x faster than radix sort at 100,000 elements and up

2. **Where it loses**
   - Wide, evenly spread values (random over 1e9, exponential): from 100,000 elements up, radix sort is about 2-3x faster. Spreading elements over hundreds of thousands of buckets is hard on the cache.

3. **Caveats**
   - `qsort` calls its comparator through a function pointer, so it is a slow baseline; an inlined comparison sort would be closer
   - The inputs are all non-negative and at most 1e9, because the original bucket sort baseline cannot handle wider ranges. AHB Sort itself handles the full `int` range, and the tests cover it.

## How It Works 🔍

AHB Sort is built on the foundation of bucket sort but incorporates several optimizations and adaptive strategies:

1. **Initial Checks**
   - Handles edge cases (empty or single-element arrays)
   - Detects already sorted sequences
   - Uses insertion sort for small arrays (n ≤ 16)

2. **Counting Path for Narrow Ranges**
   - Finds the minimum and maximum in one pass
   - If the range is at most 2n, counts each value and writes the values straight back
   - No scratch buffer or bucket pass needed

3. **Bucket Path**
   - Aims for n / 2 buckets
   - Rounds the bucket width up to a power of two, so the bucket index is a single shift
   - Counts, computes offsets, then distributes into one scratch buffer

4. **Hybrid Sorting Approach**
   - Small buckets are finished with insertion sort
   - Large buckets are sorted recursively; each level works on a strictly narrower range

## Optimizations in Detail ⚡

### 1. Early Exit Strategies
- Immediate return for single-element arrays
- Detection of pre-sorted sequences
- Small array optimization with insertion sort

### 2. Range Handling
- Integer-only min/max pass
- 64-bit range and unsigned offsets, so any spread of `int` values is safe
- Counting sort when the range is narrow

### 3. Memory Management
- Single contiguous allocation for the scratch buffer and bucket bookkeeping
- Falls back to an in-place sort if that allocation fails

### 4. Distribution Techniques
- Shift-based bucket indexing
- Unrolled counting loop
- Bucket count scaled to the array size

### 5. Sorting Strategy
- Adaptive threshold between recursion and insertion sort
- Guaranteed progress at every recursion level

## Implementation Details 🛠️

- Single header, C99, no dependencies beyond the standard library
- Sorts `int` arrays in ascending order
- Uses O(n) extra memory
- Not a stable sort (irrelevant for plain integers)

## Usage
```c
#include "ahb_sort.h"

// Sort an integer array
void AHB_Sort(int *arr, int size);
```

## Building and Testing

```sh
make test       # correctness tests against qsort, with AddressSanitizer and UBSan
make benchmark  # the benchmark behind the numbers above
make variants   # timing comparison of experimental variants (see experiments/)
```

## Future Improvements 🔮

Potential areas for further optimization:
- Fewer buckets per level, to close the gap to radix sort on wide ranges
- One scratch buffer shared across the recursion
- SIMD operations integration
- Parallel processing support

## Contributing

Feel free to experiment with the code and submit improvements. This is a learning project aimed at understanding sorting algorithm optimization techniques.

## License

Open source - feel free to use and modify.

---

*Exploring the limits of sorting algorithm optimization*
