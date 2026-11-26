# AHB Sort: An Optimized Bucket Sort Implementation 🚀

A pet project focused on optimizing the classic bucket sort algorithm through various techniques and improvements. The result is AHB (Adaptive Hybrid Bucket) Sort, a sort for integer arrays that is much faster than the C library's `qsort` and about twice as fast as the bucket sort it started from.

## Performance Highlights 📊

Geometric mean over 6 input patterns and 5 array sizes (1,000 to 5,000,000 elements):

- 12.8x faster than `qsort`
- 2.5x faster than the original bucket sort
- 1.7x faster than a byte-wise LSD radix sort on average, but **slower than it on wide, evenly spread data** (see below)

Measured on an Apple M3 Max with Apple clang 21, `-O3 -march=native`. Numbers will differ on other machines.

## Detailed Results (5M Elements) 📊

Best of 7 runs, in milliseconds. Lower is better.

| Input | AHB Sort | Original bucket | `qsort` | Radix (base 256) |
|---|---|---|---|---|
| Random, range 1000 | 1.92 | 17.28 | 178.39 | 14.88 |
| Random, range 1e9 | 41.20 | 54.89 | 436.85 | 15.49 |
| Nearly sorted | 22.32 | 40.36 | 173.95 | 38.89 |
| 100 distinct values | 7.48 | 16.60 | 108.75 | 35.35 |
| Exponential | 48.36 | 96.05 | 399.86 | 16.80 |
| Cluster + 1% outliers | 21.74 | 44.83 | 179.55 | 34.35 |

How many times faster AHB Sort is (below 1x means AHB Sort is slower):

| Input | vs original bucket | vs `qsort` | vs radix |
|---|---|---|---|
| Random, range 1000 | 9.0x | 92.7x | 7.7x |
| Random, range 1e9 | 1.3x | 10.6x | 0.38x |
| Nearly sorted | 1.8x | 7.8x | 1.7x |
| 100 distinct values | 2.2x | 14.5x | 4.7x |
| Exponential | 2.0x | 8.3x | 0.35x |
| Cluster + 1% outliers | 2.1x | 8.3x | 1.6x |

## Results by Array Size 📈

Geometric mean speedup of AHB Sort over the 6 input patterns:

| Elements | vs original bucket | vs `qsort` | vs radix |
|---|---|---|---|
| 1,000 | 2.2x | 6.9x | 1.7x |
| 10,000 | 2.7x | 15.0x | 1.9x |
| 100,000 | 2.7x | 15.2x | 1.6x |
| 1,000,000 | 2.7x | 15.6x | 1.7x |
| 5,000,000 | 2.4x | 14.0x | 1.5x |

## Wide, Evenly Spread Data 🎲

This is the input bucket sort is worst at against radix sort, and the one the latest change targets. AHB Sort now samples the array first and, when the values look evenly spread, switches to a strategy built for that case (see [How It Works](#how-it-works-)).

"Random, range 1e9" with 32-bit elements, in milliseconds:

| Elements | AHB Sort | Radix (base 256) | vs radix |
|---|---|---|---|
| 100,000 | 0.46 | 0.28 | 0.61x |
| 1,000,000 | 4.73 | 2.91 | 0.62x |
| 5,000,000 | 41.20 | 15.49 | 0.38x |

Compared with the previous version of AHB Sort, this input is about 20-35% faster at 100,000 and 1,000,000 elements and about 10-15% faster at 5,000,000, where the bucket bookkeeping no longer fits in the cache. Radix sort is still ahead, by about 1.6x up to 1,000,000 elements instead of about 2.5x.

Data that is wide but *not* evenly spread (the exponential input) does not take the new route and is unchanged: radix sort is about 3x faster there.

## 64-bit Elements 📏

The first three tables are for 32-bit elements (`int`). The same benchmark with `int64_t` elements, over 10 input patterns (the six above plus four that need more than 32 bits), gives a geometric mean of:

- 11.3x faster than `qsort`
- 2.0x faster than an eight-pass byte-wise radix sort

At 5M elements (best of 7 runs, in milliseconds):

| Input | AHB Sort | `qsort` | Radix (base 256) | vs `qsort` | vs radix |
|---|---|---|---|---|---|
| Random, range 1000 | 2.94 | 177.70 | 22.14 | 60.5x | 7.5x |
| Random, range 1e9 | 44.17 | 431.60 | 22.15 | 9.8x | 0.50x |
| Nearly sorted | 22.53 | 171.88 | 45.27 | 7.6x | 2.0x |
| 100 distinct values | 8.89 | 110.32 | 41.85 | 12.4x | 4.7x |
| Exponential | 55.22 | 407.00 | 22.01 | 7.4x | 0.40x |
| Cluster + 1% outliers | 24.79 | 181.17 | 40.29 | 7.3x | 1.6x |
| Random, range 1e18 | 47.92 | 432.32 | 31.58 | 9.0x | 0.66x |
| Random, full 64-bit | 45.35 | 430.47 | 30.83 | 9.5x | 0.68x |
| 100 distinct over 1e18 | 8.32 | 123.95 | 77.91 | 14.9x | 9.4x |
| Exponential, scale 1e15 | 55.53 | 430.36 | 30.82 | 7.8x | 0.55x |

At 5M elements AHB Sort is 1% to 19% slower than on the same values stored as 32-bit data, except on the narrow-range input, where it is about 50% slower (2.94 ms against 1.92 ms). The pattern is the same: it wins on narrow, clustered and repetitive data and loses to radix sort on wide, evenly spread data, though by less than at 32 bits because radix needs more passes. At 100,000 and 1,000,000 elements the random 64-bit inputs come out between 0.83x and 1.28x of radix sort.

## Performance Summary 🎯

1. **Where it shines**
   - Narrow value ranges: when the range is at most twice the array size, AHB Sort switches to a counting sort (about 90x to 120x faster than `qsort` from 100,000 elements up)
   - Few distinct values, clustered data and nearly sorted data: 1.4x to 5.3x faster than radix sort at 100,000 elements and up

2. **Where it loses**
   - Wide, evenly spread values (random over 1e9): radix sort is about 1.6x faster at 100,000 to 1,000,000 elements and 2.6x faster at 5,000,000
   - Wide, skewed values (exponential): radix sort is about 3x faster from 100,000 elements up
   - In both cases the cause is the same: spreading elements over hundreds of thousands of buckets is hard on the cache

3. **Caveats**
   - `qsort` calls its comparator through a function pointer, so it is a slow baseline; an inlined comparison sort would be closer
   - The benchmark sorts `int` arrays whose values are all non-negative and at most 1e9, because the original bucket sort baseline cannot handle wider ranges. AHB Sort itself handles the full range of every supported type, and the tests cover it.
   - The bucket counts, the counting threshold and the sample test were tuned on 32-bit data on one machine (an Apple M3 Max); the 64-bit variants use the same settings. The best bucket count depends on cache size, so the gains on evenly spread data may differ elsewhere.
   - The sample test is a heuristic on 64 elements. About 1 in 12 evenly spread inputs fails it by chance and is sorted the ordinary way: still correct, just without the speedup. "Random, full 64-bit" at 1,000,000 elements looks like one such case in the benchmark (7.2 ms, against 5.2 ms for "Random, range 1e18").

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

3. **Sampling**
   - For arrays of 1,024 elements or more, looks at 64 evenly spaced elements
   - The data counts as evenly spread if no two samples are equal and every eighth of the value range holds at least three of them

4. **Bucket Path**
   - Aims for n / 2 buckets, or n buckets if the sample says the data is evenly spread
   - Rounds the bucket width up to a power of two, so the bucket index is a single shift
   - Counts, computes offsets, then distributes into one scratch buffer

5. **Finishing the Buckets**
   - Ordinary data: small buckets are finished with insertion sort, and large buckets are sorted recursively; each level works on a strictly narrower range
   - Evenly spread data where every bucket came out small (64 elements or fewer): one insertion sort over the whole array. The buckets are already in order relative to each other, so no element moves further than its own bucket, and the sort skips the visit to each of the hundreds of thousands of tiny buckets
   - If the sample was wrong and a large bucket turns up, the sort goes back to the ordinary route

## Optimizations in Detail ⚡

### 1. Early Exit Strategies
- Immediate return for single-element arrays
- Detection of pre-sorted sequences
- Small array optimization with insertion sort

### 2. Range Handling
- Integer-only min/max pass
- Unsigned offsets from the minimum, so any spread of values is safe, up to the full range of the type
- Counting sort when the range is narrow

### 3. Memory Management
- Single contiguous allocation for the scratch buffer and bucket bookkeeping
- One 32-bit counter per bucket: it holds the bucket's size, then its write position, and ends up marking where the next bucket starts
- Falls back to an in-place sort if that allocation fails

### 4. Distribution Techniques
- Shift-based bucket indexing
- Unrolled counting loop
- Bucket count scaled to the array size and to how the sample is spread

### 5. Sorting Strategy
- Adaptive threshold between recursion and insertion sort
- Single final insertion pass when every bucket is small
- Guaranteed progress at every recursion level

## Implementation Details 🛠️

- Single header, C99, no dependencies beyond the standard library
- Sorts 32-bit and 64-bit integers, signed and unsigned, in ascending order
- Uses O(n) extra memory; if it cannot be allocated, falls back to `qsort`
- Arrays of more than `UINT32_MAX` elements are handed to `qsort`
- Sorts plain integers only, not records with a key
- All functions are `static`, and every name is prefixed with `ahb_` or `AHB_`

## Usage
```c
#include "ahb_sort.h"

void ahb_sort_i32(int32_t  *arr, size_t count);
void ahb_sort_u32(uint32_t *arr, size_t count);
void ahb_sort_i64(int64_t  *arr, size_t count);
void ahb_sort_u64(uint64_t *arr, size_t count);
```

For example:

```c
int32_t values[] = {5, -3, 2147483647, 0, -2147483648};
ahb_sort_i32(values, 5);
```

The original `AHB_Sort(int *arr, int size)` still works and calls `ahb_sort_i32`.

## Building and Testing

```sh
make test       # correctness tests against qsort, with AddressSanitizer and UBSan
make benchmark  # the benchmark behind the numbers above
make variants   # timing comparison of experimental variants (see experiments/)
```

## Future Improvements 🔮

Potential areas for further optimization:
- Closing the rest of the gap to radix sort on wide ranges, especially skewed data such as the exponential input, which the sample test does not help. Capping the buckets per level and sharing one scratch buffer across the recursion were both tried in `experiments/` and did not help.
- Checking the evenly-spread settings on x86 and other cache sizes
- Sorting records by an integer key
- SIMD operations integration
- Parallel processing support

## Contributing

Feel free to experiment with the code and submit improvements. This is a learning project aimed at understanding sorting algorithm optimization techniques.

## License

MIT - see [LICENSE](LICENSE).

---

*Exploring the limits of sorting algorithm optimization*
