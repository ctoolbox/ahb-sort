# AHB Sort: An Optimized Bucket Sort Implementation 🚀

A pet project focused on optimizing the classic bucket sort algorithm through various techniques and improvements. The result is AHB (Adaptive Hybrid Bucket) Sort, a sort for integer arrays that is much faster than the C library's `qsort`, several times faster than the bucket sort it started from, and level with or ahead of a tuned radix sort. It keeps the bucket sort for data with some order or repetition in it and hands wide, unstructured data to a radix sort of its own.

## Performance Highlights 📊

Geometric mean over 6 input patterns and 5 array sizes (1,000 to 5,000,000 elements):

- 18.4x faster than `qsort`
- 3.6x faster than the original bucket sort
- 2.5x faster than a byte-wise LSD radix sort

Measured on an Apple M3 Max with Apple clang 21, `-O3 -march=native`. Numbers will differ on other machines.

**Read these as best-case figures.** The sort's thresholds were tuned on these same six input shapes, and `qsort` and a byte-wise radix sort are easy targets. Against stronger baselines on 12 shapes that played no part in the tuning, AHB Sort at 5M elements is 4.7x faster than pdqsort and level with a three-pass radix sort (1.04x); see [Inputs Outside the Tuning Set](#inputs-outside-the-tuning-set-) and [Stronger Baselines](#stronger-baselines-).

## Detailed Results (5M Elements) 📊

Best of 7 runs, in milliseconds. Lower is better.

| Input | AHB Sort | Original bucket | `qsort` | Radix (base 256) |
|---|---|---|---|---|
| Random, range 1000 | 1.77 | 17.29 | 175.14 | 15.40 |
| Random, range 1e9 | 12.51 | 52.57 | 429.58 | 15.18 |
| Nearly sorted | 14.93 | 39.63 | 169.94 | 39.04 |
| 100 distinct values | 3.39 | 16.76 | 108.65 | 34.64 |
| Exponential | 11.47 | 96.37 | 401.55 | 16.66 |
| Cluster + 1% outliers | 21.08 | 44.79 | 177.56 | 33.84 |

How many times faster AHB Sort is:

| Input | vs original bucket | vs `qsort` | vs radix |
|---|---|---|---|
| Random, range 1000 | 9.8x | 98.8x | 8.7x |
| Random, range 1e9 | 4.2x | 34.3x | 1.2x |
| Nearly sorted | 2.7x | 11.4x | 2.6x |
| 100 distinct values | 4.9x | 32.0x | 10.2x |
| Exponential | 8.4x | 35.0x | 1.5x |
| Cluster + 1% outliers | 2.1x | 8.4x | 1.6x |

## Results by Array Size 📈

Geometric mean speedup of AHB Sort over the 6 input patterns:

| Elements | vs original bucket | vs `qsort` | vs radix |
|---|---|---|---|
| 1,000 | 2.2x | 6.9x | 1.7x |
| 10,000 | 3.1x | 17.2x | 2.2x |
| 100,000 | 4.3x | 24.7x | 2.8x |
| 1,000,000 | 4.6x | 26.8x | 2.9x |
| 5,000,000 | 4.6x | 26.7x | 3.0x |

## Wide Data 🎲

Wide value ranges are where a bucket sort struggles: the elements have to be spread over a very large number of buckets, the bookkeeping stops fitting in the CPU cache, and the small buckets are finished with an insertion sort whose branches the CPU cannot predict. Profiling 5M uniform values showed 19 of the 23 ms going into sorting the cache-sized buckets, and an in-cache radix pass doing the same work in about 7 ms.

So wide data with no sign of order or repetition is now sorted by an LSD radix sort on `value - minimum`, in at most three passes of at most 11 bits. Subtracting the minimum means only the bits the data actually uses are sorted. With 32-bit elements, in milliseconds:

| Input | Elements | Before | AHB Sort | Radix (base 256) | vs radix |
|---|---|---|---|---|---|
| Random, range 1e9 | 100,000 | 0.46 | 0.21 | 0.28 | 1.3x |
| Random, range 1e9 | 1,000,000 | 4.69 | 2.10 | 2.89 | 1.4x |
| Random, range 1e9 | 5,000,000 | 23.45 | 12.51 | 15.18 | 1.2x |
| Exponential | 100,000 | 0.67 | 0.21 | 0.29 | 1.3x |
| Exponential | 1,000,000 | 8.14 | 2.21 | 3.17 | 1.4x |
| Exponential | 5,000,000 | 21.18 | 11.47 | 16.66 | 1.5x |

The bucket route is kept wherever the elements arrive in some order, because there it is two to three times faster than a radix sort. Choosing between the two from a sample is the delicate part; see [How It Works](#how-it-works-).

## Inputs Outside the Tuning Set 🧪

The thresholds in the sort were chosen by timing the six inputs above, so those numbers flatter it. Four further sets of shapes have been written as checks. The first three (44 shapes) each exposed slowdowns that were then fixed, which makes them tuning inputs as well. The fourth was timed once, after the code was final, and nothing was changed in response.

### Shapes never used for tuning

32-bit elements, 5M elements, AHB Sort time in milliseconds. "Previous" is AHB Sort before the radix route was added; "radix-2048" is an LSD radix sort with 11-bit digits (three passes).

| Input | AHB Sort | vs previous version | vs pdqsort | vs radix-2048 |
|---|---|---|---|---|
| Uniform, 24 bits | 11.2 | 2.0x | 8.9x | 1.4x |
| Squared uniform | 12.8 | 2.5x | 7.7x | 1.05x |
| Bell (sum of three uniforms) | 12.6 | 2.4x | 7.9x | 1.06x |
| Rotated sorted | 10.1 | 1.0x | 4.5x | 2.5x |
| Block shuffled | 17.6 | 1.0x | 4.6x | 1.08x |
| Four rising sequences, interleaved | 8.5 | 1.0x | 10.1x | 2.5x |
| Descending runs of 1,000 | 59.9 | 1.0x | 1.3x | 0.32x |
| Half duplicates of 1,000 values | 29.2 | 1.0x | 2.4x | 0.67x |
| Signed, range ±2e9 | 12.8 | 1.8x | 7.7x | 1.05x |
| Noisy staircase | 19.2 | 1.0x | 4.5x | 1.05x |
| Bit-reversed counter | 31.9 | 0.38x | 3.0x | 1.0x |
| Cauchy tails | 33.3 | 1.0x | 3.0x | 0.64x |

- **Against the previous version** the geometric mean is 1.19x at 5M, 1.40x at 1M and 1.32x at 100,000 elements. Four shapes are about twice as fast; seven are unchanged.
- **One shape is much slower than before: a bit-reversed counter at 5M (0.38x).** Nothing in the sample marks it as ordered, so it is radix sorted, and radix passes run badly on data this regular: the two plain radix sorts take just as long. The bucket route happened to handle it well. At 1M it is not slower. This is an open problem.
- **Against pdqsort** the geometric mean is 4.7x and AHB Sort wins every shape.
- **Against the three-pass radix sort** it is level overall (1.04x): clearly ahead on three shapes, clearly behind on three, and within 8% on the other six.
- **Descending runs** is the worst case found so far, three times slower than radix sort. The runs are recognised as ordered, so the bucket route is kept, but runs that start at random values scatter over millions of buckets. It was just as slow before this change.

### Shapes used for tuning

The three earlier sets (16, 16 and 12 shapes) found, in turn: bucket limits that made clustered and part-sorted data up to 2.5x slower; organ-pipe data that was not recognised as ordered (2.9x slower); and, with the radix route, ordered shapes that were sent to the radix sort (sine waves and interleaved sequences, up to 3.3x slower). Each was fixed, and the conditions in [How It Works](#how-it-works-) are the result.

With the current code the geometric mean against the previous version is 1.2x on each of the three sets at 5M elements, with no shape more than 7% slower, and 1.1x to 1.3x at 100,000 elements, where two shapes are slower: "three clusters" (1.25x) and "sorted, 25% random" (1.14x). The shapes and the harnesses that time them are not in the repository yet.

## Stronger Baselines 🥊

`qsort` is an easy target, so `make baselines` times AHB Sort against `std::sort`, [pdqsort](https://github.com/orlp/pdqsort), [ska_sort](https://github.com/skarupke/ska_sort) (in place, and its out-of-place `ska_sort_copy`) and two LSD radix sorts: the byte-wise one from the main benchmark and one with 11-bit digits, which needs three passes instead of four. The two third-party headers are downloaded on first use and are not part of this repository.

Geometric mean of each sort's time divided by AHB Sort's, 32-bit elements (above 1 means AHB Sort is faster). "Tuning" is the six inputs of the main benchmark, "other" a set of 12 further shapes:

| Elements | Inputs | `std::sort` | pdqsort | ska_sort | ska_sort_copy | Radix-256 | Radix-2048 |
|---|---|---|---|---|---|---|---|
| 100,000 | tuning | 5.9x | 5.4x | 6.2x | 3.1x | 2.6x | 2.6x |
| 100,000 | other | 5.6x | 5.6x | 4.9x | 1.8x | 1.6x | 1.6x |
| 1,000,000 | tuning | 6.5x | 6.2x | 6.5x | 3.5x | 2.8x | 2.6x |
| 1,000,000 | other | 5.8x | 5.9x | 4.8x | 1.5x | 1.5x | 1.5x |
| 5,000,000 | tuning | 6.5x | 6.1x | 5.8x | 3.5x | 2.9x | 2.6x |
| 5,000,000 | other | 5.7x | 5.5x | 4.9x | 1.5x | 1.5x | 1.3x |

- **Comparison sorts and in-place radix:** AHB Sort is faster on every one of the 18 inputs at every size; the closest is 1.6x (pdqsort on the clustered input).
- **Out-of-place radix sorts:** before the radix route was added, the 11-bit radix sort was level with AHB Sort on the 12 other shapes at 5M elements (1.1x) and twice as fast on wide unstructured data. Now AHB Sort matches it there (1.02x to 1.07x) and keeps its lead elsewhere.
- **Where AHB Sort still loses at 5M:** "one third one value" (0.6x) and "sorted, 25% random" (0.7x to 0.9x across runs); "descending, 5% random" is level or slightly behind (0.9x to 1.0x). In all three the sample sees repeats or order, so the bucket route is kept where the radix sort would have been faster.
- **Where it wins:** narrow ranges (6x to 9x), few distinct values (6x to 8x), and ordered or run-structured data (1.6x to 3.2x).
- The 12 "other" shapes are the third check set described above. They were untouched when this comparison was first run, but have since been used to adjust the choice between buckets and radix sort; the fourth set is the independent one.
- The radix sorts here are handed a scratch buffer allocated in advance, while AHB Sort allocates its own, which favours them slightly. One run on a busy machine; treat gaps under about 10% as noise.

## 64-bit Elements 📏

The first three tables are for 32-bit elements (`int`). The same benchmark with `int64_t` elements, over 10 input patterns (the six above plus four that need more than 32 bits), gives a geometric mean of:

- 15.0x faster than `qsort`
- 2.6x faster than an eight-pass byte-wise radix sort

At 5M elements (best of 7 runs, in milliseconds):

| Input | AHB Sort | `qsort` | Radix (base 256) | vs `qsort` | vs radix |
|---|---|---|---|---|---|
| Random, range 1000 | 3.00 | 177.54 | 21.68 | 59.3x | 7.2x |
| Random, range 1e9 | 14.77 | 433.38 | 22.34 | 29.3x | 1.5x |
| Nearly sorted | 15.97 | 171.83 | 44.43 | 10.8x | 2.8x |
| 100 distinct values | 3.85 | 108.10 | 42.25 | 28.1x | 11.0x |
| Exponential | 13.09 | 400.42 | 21.34 | 30.6x | 1.6x |
| Cluster + 1% outliers | 23.07 | 178.94 | 39.86 | 7.8x | 1.7x |
| Random, range 1e18 | 27.17 | 427.63 | 30.96 | 15.7x | 1.1x |
| Random, full 64-bit | 26.09 | 428.39 | 30.47 | 16.4x | 1.2x |
| 100 distinct over 1e18 | 4.57 | 126.74 | 77.95 | 27.7x | 17.1x |
| Exponential, scale 1e15 | 28.98 | 429.93 | 30.17 | 14.8x | 1.0x |

The radix route only takes ranges of up to 33 bits, so it helps 64-bit elements whose values fit in that (the first six rows) and not the four inputs that really use 64 bits. Those still go through the buckets, and their recursive calls reach the radix route once the range has narrowed.

## Performance Summary 🎯

1. **Where it shines**
   - Narrow value ranges: AHB Sort switches to a counting sort, about 100x faster than `qsort` from 100,000 elements up
   - Few distinct values, however far apart: a small hash table counts them, about 10x faster than radix sort at 5,000,000 elements
   - Clustered and nearly sorted data: 1.6x to 2.6x faster than radix sort at 5,000,000 elements
   - Against `std::sort` and pdqsort: 5x to 6x faster on average, and faster on every input tested

2. **Where it is level or loses**
   - Wide data with no order or repetition (random, exponential, Gaussian): radix sorted, 1.2x to 1.5x faster than a byte-wise radix sort and level with a three-pass one
   - Short sorted runs that start at random values: up to 3x slower than radix sort
   - Data where the sample sees repeats or partial order but a radix sort would still have been faster (a third of the elements equal, a quarter of a sorted array replaced by random values): 0.6x to 0.9x
   - Very regular data that looks random to the sample (a bit-reversed counter): 2.6x slower than the previous version at 5M elements

3. **Caveats**
   - `qsort` calls its comparator through a function pointer, so it is a slow baseline; an inlined comparison sort would be closer. The radix sort is a plain byte-wise one. [Stronger Baselines](#stronger-baselines-) has both.
   - The main benchmark uses the inputs the sort was tuned on; see [Inputs Outside the Tuning Set](#inputs-outside-the-tuning-set-).
   - The benchmark sorts `int` arrays whose values are all non-negative and at most 1e9, because the original bucket sort baseline cannot handle wider ranges. AHB Sort itself handles the full range of every supported type, and the tests cover it.
   - Every threshold was tuned on 32-bit data on one machine, an Apple M3 Max; the 64-bit variants use the same settings. The array size at which AHB Sort switches to fewer buckets (2^21 elements) reflects that machine's caches (128 KB L1, 16 MB L2) and is probably not the best choice elsewhere.
   - The strategy is chosen from a sample of 64 elements, so it can be wrong. The result is always correctly sorted; a wrong guess only costs speed. About 1 in 12 evenly spread inputs fails the evenly-spread test by chance. "Random, full 64-bit" at 1,000,000 elements looks like one such case in the benchmark (7.2 ms, against 5.0 ms for "Random, range 1e18").

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
   - *Nearly sorted*: at least 48 of the 63 neighbouring samples are already in order, or at least 48 of the 64 samples are in order with the element right after them (which catches arrays made of sorted runs), or the samples change direction at most 7 times (which catches long rising and falling stretches). Random data gives about 32, 32 and 41

3. **Counting Path for Narrow Ranges**
   - If the range is at most 2n (at most n for evenly spread data), counts each value and writes the values straight back
   - No scratch buffer or bucket pass needed

4. **Sparse Path for Few Distinct Values**
   - If the sample has repeats, counts the distinct values in a 4,096-slot hash table, sorts them, and writes each back as many times as it occurred
   - Gives up as soon as it has seen more than 256 distinct values

5. **Radix Path for Wide, Unstructured Data**
   - Taken when the sample shows no repeats and no order, the range fits in 33 bits, and the array has at least 1,024 elements (65,536 if evenly spread; below that the bucket path is faster)
   - Before committing, looks harder for reasons to stay with the buckets, because data that arrives in order makes them two to three times faster than a radix sort:
     - local order: 64 windows of four consecutive elements are mostly monotone
     - two interleaved sequences, each rising or falling
     - a few interleaved sequences: samples that are neighbours in value are also in order of position
     - clusters: the sample falls into fewer than 32 of 8,192 equal slices of the range
     - dense data: the middle half of the sample spans fewer values than an eighth of the element count
     - a few thousand distinct values: on arrays of 65,536 or more, repeats among 256 further samples
   - Sorts on `value - minimum` in at most three passes of at most 11 bits, counting all digits in one pass and skipping a pass in which every element has the same digit

6. **Bucket Path**
   - Aims for n / 2 buckets, or n buckets for evenly spread data
   - Arrays of more than 2^21 elements get far fewer buckets, so the bookkeeping stays in the cache: n / 4096 for evenly spread data and 8,192 for other data
   - That limit is skipped where it would not help: for nearly sorted data, which fills the buckets in sequence anyway; when the sample falls into fewer than 32 of the 8,192 buckets, because narrow buckets isolate clusters at once; and when the sparse path got through more than half the array before giving up
   - Rounds the bucket width up to a power of two, so the bucket index is a single shift
   - Counts, computes offsets, then distributes into one scratch buffer

7. **Finishing the Buckets**
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
- Bucket boundaries taken from the sample's distribution, so that skewed data fills the buckets evenly: 12-20% faster on skewed data, 15% slower on uniform data, and far behind the radix route on both
- Leaving the radix route when its digit counts come out almost exactly equal (the bit-reversed counter case): faster at 5M elements, slower at 1M

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
- Recognising short sorted runs that start at random values, and very regular data such as a bit-reversed counter
- A radix route for ranges wider than 33 bits
- Choosing the thresholds from the machine's cache sizes instead of fixed array sizes, and checking them on x86
- Putting the check sets of input shapes in the repository
- Sorting records by an integer key
- SIMD operations integration
- Parallel processing support

## Contributing

Feel free to experiment with the code and submit improvements. This is a learning project aimed at understanding sorting algorithm optimization techniques.

## License

MIT - see [LICENSE](LICENSE).

---

*Exploring the limits of sorting algorithm optimization*
