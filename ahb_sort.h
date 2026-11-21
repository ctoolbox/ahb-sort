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

    /* Narrow range (at most 2 * count possible values): count each value and
     * write the values straight back. No scratch buffer or bucket pass. */
    if ((uint64_t)span < 2 * (uint64_t)count) {
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

    /* Aim for count / 2 buckets, then round the bucket width up to a power of
     * two so the bucket index is a single shift.
     *
     * The width starts as ceil(range / target) with target >= 2, and rounding
     * it up to a power of two still leaves it <= span. So the minimum and
     * maximum always land in different buckets, and every recursive call
     * works on a strictly narrower range. */
    AHB_U bucket_width = span / (AHB_U)(count / 2) + 1;
    unsigned shift = 0;
    while (((AHB_U)1 << shift) < bucket_width) shift++;
    size_t bucket_count = (size_t)(span >> shift) + 1;

    /* One allocation: the scratch copy of the data, then three arrays of
     * per-bucket bookkeeping */
    size_t data_bytes = count * sizeof(AHB_T);
    size_t bookkeeping_bytes = bucket_count * sizeof(uint32_t) * 3;
    char *memory = NULL;
    if (bookkeeping_bytes <= SIZE_MAX - data_bytes) {
        memory = (char *)malloc(data_bytes + bookkeeping_bytes);
    }
    if (!memory) {
        /* No scratch space: still sort, in place */
        qsort(arr, count, sizeof(AHB_T), AHB_COMPARE_FN);
        return;
    }

    AHB_T *buckets = (AHB_T *)memory;
    uint32_t *bucket_offsets = (uint32_t *)(memory + data_bytes);
    uint32_t *bucket_sizes = bucket_offsets + bucket_count;
    uint32_t *positions = bucket_offsets + 2 * bucket_count;

    memset(bucket_offsets, 0, bookkeeping_bytes);

    /* Count elements per bucket */
    i = 0;
    for (; i + 4 <= count; i += 4) {
        size_t b1 = AHB_BUCKET_OF(arr[i]);
        size_t b2 = AHB_BUCKET_OF(arr[i + 1]);
        size_t b3 = AHB_BUCKET_OF(arr[i + 2]);
        size_t b4 = AHB_BUCKET_OF(arr[i + 3]);

        bucket_sizes[b1]++;
        bucket_sizes[b2]++;
        bucket_sizes[b3]++;
        bucket_sizes[b4]++;
    }
    for (; i < count; i++) {
        bucket_sizes[AHB_BUCKET_OF(arr[i])]++;
    }

    /* Calculate offsets */
    uint32_t running_total = 0;
    for (i = 0; i < bucket_count; i++) {
        bucket_offsets[i] = running_total;
        positions[i] = running_total;
        running_total += bucket_sizes[i];
    }

    /* Distribute elements */
    for (i = 0; i < count; i++) {
        buckets[positions[AHB_BUCKET_OF(arr[i])]++] = arr[i];
    }

    /* Sort individual buckets */
    size_t threshold = count < 1024 ? count / 16 : 64;
    for (i = 0; i < bucket_count; i++) {
        size_t bucket_size = bucket_sizes[i];
        if (bucket_size <= 1) continue;

        AHB_T *bucket = &buckets[bucket_offsets[i]];
        if (bucket_size > threshold) {
            AHB_SORT_FN(bucket, bucket_size);
        } else {
            AHB_INSERTION_FN(bucket, bucket_size);
        }
    }

    memcpy(arr, buckets, data_bytes);
    free(memory);
}

#undef AHB_BUCKET_OF
#undef AHB_COMPARE_FN
#undef AHB_INSERTION_FN
#undef AHB_SORT_FN
#undef AHB_SUFFIX
#undef AHB_T
#undef AHB_U

#endif /* AHB_SORT_TEMPLATE_PASS */
