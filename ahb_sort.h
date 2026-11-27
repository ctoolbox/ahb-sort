/*
 * Adaptive Hybrid Bucket Sort (AHB Sort)
 * =====================================
 * An optimized bucket sort for integer arrays. Single header, C99.
 *
 *     #include "ahb_sort.h"
 *
 *     void ahb_sort_i32(int32_t  *arr, size_t count);
 *     void ahb_sort_u32(uint32_t *arr, size_t count);
 *     void ahb_sort_i64(int64_t  *arr, size_t count);
 *     void ahb_sort_u64(uint64_t *arr, size_t count);
 *
 * Each sorts the array in ascending order, for any values of that type.
 * The sort needs O(count) temporary memory; if that cannot be allocated it
 * falls back to the C library's qsort, so the array always ends up sorted.
 * Arrays of more than UINT32_MAX elements are also handed to qsort.
 *
 * AHB_Sort(int *arr, int size) is kept for existing code.
 *
 * Every function is static, and everything this header defines is prefixed
 * with ahb_ or AHB_.
 *
 * How this file is organised: the algorithm is written once, as a template
 * in the second half of the file. The first half includes the file itself
 * once per element type to generate the four functions, so the file must
 * keep the name ahb_sort.h.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef AHB_SORT_TEMPLATE_PASS

#ifndef AHB_SORT_H
#define AHB_SORT_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define AHB_SORT_TEMPLATE_PASS
#define AHB_CONCAT_(a, b) a##b
#define AHB_CONCAT(a, b) AHB_CONCAT_(a, b)

#define AHB_SUFFIX i32
#define AHB_T int32_t
#define AHB_U uint32_t
#include "ahb_sort.h"

#define AHB_SUFFIX u32
#define AHB_T uint32_t
#define AHB_U uint32_t
#include "ahb_sort.h"

#define AHB_SUFFIX i64
#define AHB_T int64_t
#define AHB_U uint64_t
#include "ahb_sort.h"

#define AHB_SUFFIX u64
#define AHB_T uint64_t
#define AHB_U uint64_t
#include "ahb_sort.h"

#undef AHB_CONCAT
#undef AHB_CONCAT_
#undef AHB_SORT_TEMPLATE_PASS

/* AHB_Sort treats an int array as int32_t, so int must be 32 bits wide. */
typedef char ahb_sort_requires_32_bit_int[sizeof(int) == sizeof(int32_t) ? 1 : -1];

/* The original entry point, kept for existing code. */
static inline void AHB_Sort(int *arr, int size) {
    if (size > 1) ahb_sort_i32((int32_t *)arr, (size_t)size);
}

#endif /* AHB_SORT_H */

#else /* AHB_SORT_TEMPLATE_PASS */

/*
 * Template: generates ahb_sort_<suffix> and its helpers for one element type.
 *
 *   AHB_SUFFIX  name suffix, e.g. i32
 *   AHB_T       element type
 *   AHB_U       unsigned type of the same width as AHB_T
 */

#define AHB_SORT_FN AHB_CONCAT(ahb_sort_, AHB_SUFFIX)
#define AHB_INSERTION_FN AHB_CONCAT(ahb_insertion_sort_, AHB_SUFFIX)
#define AHB_COMPARE_FN AHB_CONCAT(ahb_compare_, AHB_SUFFIX)
#define AHB_SPARSE_FN AHB_CONCAT(ahb_sparse_sort_, AHB_SUFFIX)

/* Slot of a value in the 4096-slot table of the sparse sort */
#define AHB_HASH(x) ((size_t)(((uint64_t)(AHB_U)(x) * 0x9E3779B97F4A7C15ULL) >> 52))

/* Bucket of an element: its offset from the minimum, divided by the bucket
 * width (a power of two). Unsigned subtraction gives the exact offset for any
 * pair of values, including ones whose difference overflows the signed type. */
#define AHB_BUCKET_OF(x) ((size_t)(((AHB_U)(x) - min_u) >> shift))

/* Helper for small arrays */
static inline void AHB_INSERTION_FN(AHB_T *arr, size_t count) {
    for (size_t i = 1; i < count; i++) {
        AHB_T key = arr[i];
        size_t j = i;
        while (j > 0 && arr[j - 1] > key) {
            arr[j] = arr[j - 1];
            j--;
        }
        arr[j] = key;
    }
}

/* Comparator for the qsort fallback */
static int AHB_COMPARE_FN(const void *a, const void *b) {
    AHB_T x = *(const AHB_T *)a, y = *(const AHB_T *)b;
    return (x > y) - (x < y);
}

/* Sort for arrays with few distinct values: count each distinct value in a
 * small hash table, sort the distinct values, and write each one back as
 * many times as it occurred. Returns 1 if it sorted the array. Returns 0,
 * leaving the array untouched, if it met more than 256 distinct values or
 * could not allocate the table; *scanned is then the number of elements it
 * got through first. */
static int AHB_SPARSE_FN(AHB_T *arr, size_t count, size_t *scanned) {
    enum { AHB_SLOTS = 4096, AHB_MAX_DISTINCT = 256 };
    size_t keys_bytes = AHB_SLOTS * sizeof(AHB_T);
    size_t distinct_bytes = AHB_MAX_DISTINCT * sizeof(AHB_T);
    char *memory = (char *)malloc(keys_bytes + distinct_bytes + AHB_SLOTS * sizeof(uint32_t));
    *scanned = 0;
    if (!memory) return 0;

    AHB_T *keys = (AHB_T *)memory;
    AHB_T *distinct = (AHB_T *)(memory + keys_bytes);
    uint32_t *counts = (uint32_t *)(memory + keys_bytes + distinct_bytes);
    size_t distinct_count = 0;
    memset(counts, 0, AHB_SLOTS * sizeof(uint32_t));

    /* A slot is in use when its count is non-zero. Collisions move on to the
     * next slot; the table is at most one sixteenth full, so runs stay short. */
    size_t i;
    for (i = 0; i < count; i++) {
        AHB_T value = arr[i];
        size_t slot = AHB_HASH(value);
        while (counts[slot] && keys[slot] != value) slot = (slot + 1) & (AHB_SLOTS - 1);
        if (!counts[slot]) {
            if (distinct_count == AHB_MAX_DISTINCT) break;
            keys[slot] = value;
            distinct[distinct_count++] = value;
        }
        counts[slot]++;
    }

    int sorted = i == count;
    if (sorted) {
        AHB_INSERTION_FN(distinct, distinct_count);
        AHB_T *out = arr;
        for (size_t d = 0; d < distinct_count; d++) {
            AHB_T value = distinct[d];
            size_t slot = AHB_HASH(value);
            while (keys[slot] != value) slot = (slot + 1) & (AHB_SLOTS - 1);
            for (uint32_t c = counts[slot]; c > 0; c--) *out++ = value;
        }
    }
    *scanned = i;
    free(memory);
    return sorted;
}

/* Main sorting function */
static inline void AHB_SORT_FN(AHB_T *arr, size_t count) {
    if (count <= 1) return;
    if (count <= 16) {
        AHB_INSERTION_FN(arr, count);
        return;
    }

    /* Early exit for sorted sequences */
    int is_sorted = 1;
    for (size_t i = 1; i < count; i++) {
        if (arr[i] < arr[i - 1]) {
            is_sorted = 0;
            break;
        }
    }
    if (is_sorted) return;

    /* Bucket counters are 32-bit to keep the bookkeeping small */
    if (count > UINT32_MAX) {
        qsort(arr, count, sizeof(AHB_T), AHB_COMPARE_FN);
        return;
    }

    /* Find the value range */
    AHB_T max_value = arr[0], min_value = arr[0];
    size_t i;
    for (i = 1; i < count; i++) {
        AHB_T val = arr[i];
        max_value = val > max_value ? val : max_value;
        min_value = val < min_value ? val : min_value;
    }

    /* span is (number of possible values in the range) - 1. Keeping it one
     * short of the range means it always fits in the unsigned type.
     * span >= 1 here: the array is unsorted, so it holds two distinct values. */
    const AHB_U min_u = (AHB_U)min_value;
    const AHB_U span = (AHB_U)max_value - min_u;

    /* Look at 64 evenly spaced elements to choose a strategy:
     *
     *   repeats  two samples are equal, so there may be few distinct values
     *   spread   the values look evenly spread over the range: no two samples
     *            are equal and every eighth of the range holds at least three
     *   ordered  the array is close to sorted, or made of sorted runs, so its
     *            elements reach the buckets more or less in sequence. One of:
     *            at least 48 of the 63 neighbouring samples are in ascending
     *            order (or 48 in descending order); at least 48 of the 64
     *            samples are in order with the element right after them; or
     *            the samples change direction at most 7 times (long rising
     *            and falling stretches). Random data gives about 32, 32
     *            and 41. */
    AHB_T sample[64];
    int spread = 0, repeats = 0, ordered = 0;
    if (count >= 1024) {
        unsigned bins[8] = {0};
        unsigned rising = 0, falling = 0, steps_up = 0, steps_down = 0, turns = 0;
        int direction = 0;
        const size_t step = count / 64;
        const AHB_U bin_width = span / 8 + 1;
        for (i = 0; i < 64; i++) {
            AHB_T value = arr[i * step], next = arr[i * step + 1];
            sample[i] = value;
            bins[((AHB_U)value - min_u) / bin_width]++;
            if (next >= value) steps_up++;
            if (next <= value) steps_down++;
        }
        for (i = 1; i < 64; i++) {
            const int d = (sample[i] > sample[i - 1]) - (sample[i] < sample[i - 1]);
            if (d >= 0) rising++;
            if (d <= 0) falling++;
            if (d && direction && d != direction) turns++;
            if (d) direction = d;
        }
        ordered = rising >= 48 || falling >= 48 || steps_up >= 48 || steps_down >= 48 || turns <= 7;
        AHB_INSERTION_FN(sample, 64);
        for (i = 1; i < 64; i++) {
            if (sample[i] == sample[i - 1]) repeats = 1;
        }
        spread = !repeats;
        for (i = 0; i < 8; i++) {
            if (bins[i] < 3) spread = 0;
        }
    }

    /* Narrow range: count each value and write the values straight back. No
     * scratch buffer or bucket pass. Used when the range holds at most
     * 2 * count possible values, or at most count if the data is evenly
     * spread, where the bucket route below is faster for the wider ranges. */
    if ((uint64_t)span < (spread ? 1 : 2) * (uint64_t)count) {
        size_t range = (size_t)span + 1;
        uint32_t *counts = (uint32_t *)calloc(range, sizeof(uint32_t));
        if (counts) {
            for (i = 0; i < count; i++) counts[(AHB_U)arr[i] - min_u]++;
            AHB_T *out = arr;
            for (size_t v = 0; v < range; v++) {
                AHB_T value = (AHB_T)(min_u + (AHB_U)v);
                for (uint32_t c = counts[v]; c > 0; c--) *out++ = value;
            }
            free(counts);
            return;
        }
        /* Allocation failed: carry on with the bucket pass below */
    }

    /* Wide range but repeated values in the sample: try the sparse sort */
    size_t sparse_scanned = 0;
    if (repeats && AHB_SPARSE_FN(arr, count, &sparse_scanned)) return;

    /* Choose how many buckets to aim for.
     *
     *   - count / 2 by default.
     *   - count for evenly spread data: that leaves so few elements per
     *     bucket that one insertion pass over the whole array finishes the
     *     job (see below).
     *   - Arrays of more than 2^21 elements get far fewer buckets, because
     *     the per-bucket counters stop fitting in the cache and every count
     *     and every move becomes a cache miss. The buckets are then large and
     *     are sorted recursively, each one small enough to stay in the cache.
     *     Evenly spread data gets count / 4096 buckets and other data 8192.
     *
     * Fewer buckets only help if the data really spreads over them, so the
     * limit is not applied when:
     *   - the array is close to sorted or made of sorted runs: it fills the
     *     buckets in sequence, which is already kind to the cache;
     *   - the sample falls into fewer than 32 of the 8192 buckets: the data
     *     sits in a few clusters, and narrow buckets isolate those at once;
     *   - the sparse sort got through more than half the array before giving
     *     up: a few values dominate, and narrow buckets isolate those too. */
    size_t target = count / 2;
    if (spread) {
        target = count;
        if (count > ((size_t)1 << 21) && !ordered) target = count / 4096;
    } else if (count > ((size_t)1 << 21) && !ordered && sparse_scanned <= count / 2) {
        const AHB_U wide_width = span / 8192 + 1;
        unsigned occupied = 1;
        for (i = 1; i < 64; i++) {
            if (((AHB_U)sample[i] - min_u) / wide_width != ((AHB_U)sample[i - 1] - min_u) / wide_width) {
                occupied++;
            }
        }
        if (occupied >= 32) target = 8192;
    }

    /* Round the bucket width up to a power of two so the bucket index is a
     * single shift.
     *
     * The width starts as ceil(range / target) with target >= 2, and rounding
     * it up to a power of two still leaves it <= span. So the minimum and
     * maximum always land in different buckets, and every recursive call
     * works on a strictly narrower range. */
    AHB_U bucket_width = span / (AHB_U)target + 1;
    unsigned shift = 0;
    while (((AHB_U)1 << shift) < bucket_width) shift++;
    size_t bucket_count = (size_t)(span >> shift) + 1;

    /* One allocation: the scratch copy of the data, then one counter per
     * bucket */
    size_t data_bytes = count * sizeof(AHB_T);
    size_t bookkeeping_bytes = bucket_count * sizeof(uint32_t);
    char *memory = NULL;
    if (bookkeeping_bytes <= SIZE_MAX - data_bytes) {
        memory = (char *)malloc(data_bytes + bookkeeping_bytes);
    }
    if (!memory) {
        /* No scratch space: still sort, in place */
        qsort(arr, count, sizeof(AHB_T), AHB_COMPARE_FN);
        return;
    }

    /* positions holds the size of each bucket first, then the place where
     * the bucket's next element goes. Once the elements are distributed,
     * positions[b] is the end of bucket b, which is the start of bucket
     * b + 1. */
    AHB_T *buckets = (AHB_T *)memory;
    uint32_t *positions = (uint32_t *)(memory + data_bytes);

    memset(positions, 0, bookkeeping_bytes);

    /* Count elements per bucket */
    i = 0;
    for (; i + 4 <= count; i += 4) {
        size_t b1 = AHB_BUCKET_OF(arr[i]);
        size_t b2 = AHB_BUCKET_OF(arr[i + 1]);
        size_t b3 = AHB_BUCKET_OF(arr[i + 2]);
        size_t b4 = AHB_BUCKET_OF(arr[i + 3]);

        positions[b1]++;
        positions[b2]++;
        positions[b3]++;
        positions[b4]++;
    }
    for (; i < count; i++) {
        positions[AHB_BUCKET_OF(arr[i])]++;
    }

    /* Turn sizes into start offsets */
    uint32_t running_total = 0, largest_bucket = 0;
    for (i = 0; i < bucket_count; i++) {
        uint32_t bucket_size = positions[i];
        largest_bucket = bucket_size > largest_bucket ? bucket_size : largest_bucket;
        positions[i] = running_total;
        running_total += bucket_size;
    }

    /* Distribute elements */
    for (i = 0; i < count; i++) {
        buckets[positions[AHB_BUCKET_OF(arr[i])]++] = arr[i];
    }

    if (spread && largest_bucket <= 64) {
        /* Evenly spread data, and every bucket turned out small. The buckets
         * are already in order relative to each other, so one insertion sort
         * over everything only ever moves an element within its own bucket.
         * That is cheaper than visiting the buckets one at a time. It reads
         * from the scratch buffer and writes into arr, which also replaces
         * the copy back. */
        arr[0] = buckets[0];
        for (i = 1; i < count; i++) {
            AHB_T key = buckets[i];
            size_t j = i;
            while (j > 0 && arr[j - 1] > key) {
                arr[j] = arr[j - 1];
                j--;
            }
            arr[j] = key;
        }
    } else {
        /* Sort individual buckets */
        size_t threshold = count < 1024 ? count / 16 : 64;
        size_t bucket_start = 0;
        for (i = 0; i < bucket_count; i++) {
            size_t bucket_end = positions[i];
            size_t bucket_size = bucket_end - bucket_start;
            AHB_T *bucket = &buckets[bucket_start];
            bucket_start = bucket_end;
            if (bucket_size <= 1) continue;

            if (bucket_size > threshold) {
                AHB_SORT_FN(bucket, bucket_size);
            } else {
                AHB_INSERTION_FN(bucket, bucket_size);
            }
        }
        memcpy(arr, buckets, data_bytes);
    }

    free(memory);
}

#undef AHB_BUCKET_OF
#undef AHB_HASH
#undef AHB_SPARSE_FN
#undef AHB_COMPARE_FN
#undef AHB_INSERTION_FN
#undef AHB_SORT_FN
#undef AHB_SUFFIX
#undef AHB_T
#undef AHB_U

#endif /* AHB_SORT_TEMPLATE_PASS */
