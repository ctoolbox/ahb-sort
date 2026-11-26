# AHB Sort: An Optimized Bucket Sort Implementation 🚀

A pet project focused on optimizing the classic bucket sort algorithm through various techniques and improvements. The result is AHB (Adaptive Hybrid Bucket) Sort, a sort for integer arrays that is much faster than the C library's `qsort` and about twice as fast as the bucket sort it started from.

## Performance Highlights 📊

Geometric mean over 6 input patterns and 5 array sizes (1,000 to 5,000,000 elements):

- 15.2x faster than `qsort`
- 3.0x faster than the original bucket sort
- 2.0x faster than a byte-wise LSD radix sort on average, but **slower than it on wide 32-bit data** (see below)

Measured on an Apple M3 Max with Apple clang 21, `-O3 -march=native`. Numbers will differ on other machines.

## Detailed Results (5M Elements) 📊

Best of 7 runs, in milliseconds. Lower is better.

| Input | AHB Sort | Original bucket | `qsort` | Radix (base 256) |
|---|---|---|---|---|
| Random, range 1000 | 1.76 | 17.27 | 177.43 | 14.97 |
| Random, range 1e9 | 23.82 | 54.54 | 438.33 | 15.42 |
| Nearly sorted | 15.19 | 40.04 | 171.89 | 39.73 |
| 100 distinct values | 3.92 | 16.99 | 108.90 | 35.31 |
| Exponential | 21.88 | 97.56 | 408.89 | 16.12 |
| Cluster + 1% outliers | 22.81 | 45.90 | 180.44 | 35.06 |

How many times faster AHB Sort is (below 1x means AHB Sort is slower):

| Input | vs original bucket | vs `qsort` | vs radix |
|---|---|---|---|
| Random, range 1000 | 9.8x | 100.9x | 8.5x |
| Random, range 1e9 | 2.3x | 18.4x | 0.65x |
| Nearly sorted | 2.6x | 11.3x | 2.6x |
| 100 distinct values | 4.3x | 27.8x | 9.0x |
| Exponential | 4.5x | 18.7x | 0.74x |
| Cluster + 1% outliers | 2.0x | 7.9x | 1.5x |

## Results by Array Size 📈

Geometric mean speedup of AHB Sort over the 6 input patterns:

| Elements | vs original bucket | vs `qsort` | vs radix |
|---|---|---|---|
| 1,000 | 2.2x | 7.0x | 1.8x |
| 10,000 | 2.8x | 15.7x | 2.1x |
| 100,000 | 3.1x | 18.0x | 1.9x |
| 1,000,000 | 3.3x | 19.9x | 2.0x |
| 5,000,000 | 3.6x | 21.0x | 2.3x |

## Wide Data 🎲

Wide value ranges are where a bucket sort struggles against radix sort: the elements have to be spread over a very large number of buckets, and the per-bucket bookkeeping stops fitting in the CPU cache. AHB Sort handles this in two ways (see [How It Works](#how-it-works-)): a faster route for evenly spread data, and far fewer, larger buckets once the array is big enough for the cache to matter.

With 32-bit elements, in milliseconds:

| Input | Elements | AHB Sort | Radix (base 256) | vs radix |
|---|---|---|---|---|
| Random, range 1e9 | 100,000 | 0.44 | 0.28 | 0.62x |
| Random, range 1e9 | 1,000,000 | 4.57 | 2.91 | 0.64x |
| Random, range 1e9 | 5,000,000 | 23.82 | 15.42 | 0.65x |
| Exponential | 100,000 | 0.69 | 0.24 | 0.35x |
| Exponential | 1,000,000 | 6.33 | 2.53 | 0.40x |
| Exponential | 5,000,000 | 21.88 | 16.12 | 0.74x |

Radix sort is still ahead on both, by about 1.6x on evenly spread data and by 1.4x to 2.9x on the skewed exponential input. The gap closes as the array grows. With 64-bit elements, where radix sort needs twice the passes, the two are roughly level (see below).

## 64-bit Elements 📏

The first three tables are for 32-bit elements (`int`). The same benchmark with `int64_t` elements, over 10 input patterns (the six above plus four that need more than 32 bits), gives a geometric mean of:

- 13.8x faster than `qsort`
- 2.4x faster than an eight-pass byte-wise radix sort

At 5M elements (best of 7 runs, in milliseconds):

| Input | AHB Sort | `qsort` | Radix (base 256) | vs `qsort` | vs radix |
|---|---|---|---|---|---|
| Random, range 1000 | 2.58 | 177.72 | 20.70 | 69.0x | 8.0x |
| Random, range 1e9 | 26.60 | 436.15 | 22.55 | 16.4x | 0.85x |
| Nearly sorted | 16.47 | 180.07 | 45.09 | 10.9x | 2.7x |
| 100 distinct values | 4.47 | 109.38 | 42.92 | 24.5x | 9.6x |
| Exponential | 23.03 | 407.00 | 21.68 | 17.7x | 0.94x |
| Cluster + 1% outliers | 23.73 | 181.83 | 39.49 | 7.7x | 1.7x |
| Random, range 1e18 | 26.79 | 436.97 | 31.81 | 16.3x | 1.2x |
| Random, full 64-bit | 25.26 | 434.99 | 30.84 | 17.2x | 1.2x |
| 100 distinct over 1e18 | 5.32 | 121.61 | 78.92 | 22.8x | 14.8x |
| Exponential, scale 1e15 | 28.45 | 435.19 | 30.74 | 15.3x | 1.1x |

At 5M elements AHB Sort is 4% to 14% slower than on the same values stored as 32-bit data, except on the narrow-range input, where it is about 45% slower (2.58 ms against 1.76 ms). At 100,000 and 1,000,000 elements the wide 64-bit inputs come out between 0.55x and 1.33x of radix sort.

## Performance Summary 🎯

1. **Where it shines**
   - Narrow value ranges: AHB Sort switches to a counting sort, about 100x faster than `qsort` from 100,000 elements up
   - Few distinct values, however far apart: a small hash table counts them, about 9x faster than radix sort from 100,000 elements up
   - Clustered and nearly sorted data: 1.4x to 2.6x faster than radix sort from 100,000 elements up

2. **Where it loses**
   - Wide, evenly spread 32-bit values (random over 1e9): radix sort is about 1.6x faster from 100,000 elements up
   - Wide, skewed 32-bit values (exponential): radix sort is 2.5x to 2.9x faster up to 1,000,000 elements and 1.4x faster at 5,000,000

3. **Caveats**
   - `qsort` calls its comparator through a function pointer, so it is a slow baseline; an inlined comparison sort would be closer. The radix sort is a plain byte-wise one; tuned radix sorts are faster.
   - The benchmark sorts `int` arrays whose values are all non-negative and at most 1e9, because the original bucket sort baseline cannot handle wider ranges. AHB Sort itself handles the full range of every supported type, and the tests cover it.
   - Every threshold was tuned on 32-bit data on one machine, an Apple M3 Max; the 64-bit variants use the same settings. The array sizes at which AHB Sort switches to fewer buckets (2^18 and 2^21 elements) reflect that machine's caches (128 KB L1, 16 MB L2) and are probably not the best choice elsewhere.
   - The strategy is chosen from a sample of 64 elements, so it can be wrong. The result is always correctly sorted; a wrong guess only costs speed. About 1 in 12 evenly spread inputs fails the evenly-spread test by chance. "Random, full 64-bit" at 1,000,000 elements looks like one such case in the benchmark (7.8 ms, against 5.0 ms for "Random, range 1e18").

## How It Works 🔍

AHB Sort is built on the foundation of bucket sort but incorporates several optimizations and adaptive strategies:

1. **Initial Checks**
   - Handles edge cases (empty or single-element arrays)
   - Detects already sorted sequences
   - Uses insertion sort for small arrays (n ≤ 16)
   - Finds the minimum and maximum in one pass

2. **Sampling**
   - For arrays of 1,024 elements or more, looks at 64 evenly spaced elements
   - *Evenly spread*: no two samples are equal and every eighth of the value range holds at least three of them
   - *Repeats*: two samples are equal, so there may be few distinct values
   - *Nearly sorted*: at least 56 of the 63 neighbouring samples are already in order
   - Also notes how wide the middle half of the sample is

3. **Counting Path for Narrow Ranges**
   - If the range is at most 2n (at most n for evenly spread data), counts each value and writes the values straight back
   - No scratch buffer or bucket pass needed

4. **Sparse Path for Few Distinct Values**
   - If the sample has repeats, counts the distinct values in a 1,024-slot hash table, sorts them, and writes each back as many times as it occurred
   - Gives up as soon as it has seen more than 256 distinct values

5. **Bucket Path**
   - Aims for n / 2 buckets, or n buckets for evenly spread data
   - Large arrays get far fewer buckets, so the bookkeeping stays in the cache: n / 4096 for evenly spread data above 2^21 elements, and 8,192 for other data from 2^18 elements
   - No such limit for nearly sorted data, which fills the buckets in sequence anyway, or for tightly clustered data, where narrow buckets isolate the cluster at once
   - Rounds the bucket width up to a power of two, so the bucket index is a single shift
   - Counts, computes offsets, then distributes into one scratch buffer

6. **Finishing the Buckets**
   - Ordinary case: small buckets are finished with insertion sort, and large buckets are sorted recursively; each level works on a strictly narrower range
   - Evenly spread data where every bucket came out small (64 elements or fewer): one insertion sort over the whole array. The buckets are already in order relative to each other, so no element moves further than its own bucket, and the sort skips the visit to each of the many tiny buckets
   - If the sample was wrong and a large bucket turns up, the sort goes back to the ordinary case

## Optimizations in Detail ⚡

### 1. Early Exit Strategies
- Immediate return for single-element arrays
- Detection of pre-sorted sequences
- Small array optimization with insertion sort

### 2. Range Handling
- Integer-only min/max pass
- Unsigned offsets from the minimum, so any spread of values is safe, up to the full range of the type
- Counting sort when the range is narrow
- Hash-table counting when the values are few but far apart

### 3. Memory Management
- Single contiguous allocation for the scratch buffer and bucket bookkeeping
- One 32-bit counter per bucket: it holds the bucket's size, then its write position, and ends up marking where the next bucket starts
- Falls back to an in-place sort if that allocation fails

### 4. Distribution Techniques
- Shift-based bucket indexing
- Unrolled counting loop
- Bucket count chosen from the array size, the sample, and what fits in the cache

### 5. Sorting Strategy
- Adaptive threshold between recursion and insertion sort
- Single final insertion pass when every bucket is small
- Guaranteed progress at every recursion level

### Tried and Not Adopted
These were measured in `experiments/` and did not help on the test machine:
- Software write combining (buffering each bucket's elements and copying them in blocks): 10-16% slower on wide uniform data at 3M to 5M elements
- One scratch buffer shared across the recursion: no measurable difference
- Trying an insertion sort with a limited number of moves before anything else: slower on every input tested, including nearly sorted ones
- A variance-based bucket count (the original design): slower than a fixed n / 2

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
- Closing the rest of the gap to radix sort on wide 32-bit ranges
- Choosing the bucket limits from the machine's cache sizes instead of fixed array sizes, and checking them on x86
- Splitting skewed data by sampled quantiles instead of equal-width buckets
- Stronger baselines in the benchmark: a tuned radix sort and an inlined comparison sort
- Sorting records by an integer key
- SIMD operations integration
- Parallel processing support

## Contributing

Feel free to experiment with the code and submit improvements. This is a learning project aimed at understanding sorting algorithm optimization techniques.

## License

MIT - see [LICENSE](LICENSE).

---

*Exploring the limits of sorting algorithm optimization*
