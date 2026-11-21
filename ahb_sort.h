#ifndef AHB_SORT_H
#define AHB_SORT_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/*
 * Adaptive Hybrid Bucket Sort (AHB Sort)
 * =====================================
 * An optimized version of bucket sort that combines multiple optimization techniques
 * to achieve significant performance improvements.
 *
 */

// Helper function for small arrays
static inline void insertion_sort(int *arr, int size) {
    for (int i = 1; i < size; i++) {
        int key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

// Comparator for the qsort fallback used when the scratch buffer can't be allocated
static int ahb_compare_ints(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

// Main sorting function
static inline void AHB_Sort(int *arr, int size) {
    if (size <= 1) return;
    if (size <= 16) {
        insertion_sort(arr, size);
        return;
    }

    // Early exit for sorted sequences
    int is_sorted = 1;
    for (int i = 1; i < size; i++) {
        if (arr[i] < arr[i-1]) {
            is_sorted = 0;
            break;
        }
    }
    if (is_sorted) return;

    // Find the value range
    int max_value = arr[0], min_value = arr[0];
    int i;
    for (i = 1; i < size; i++) {
        int val = arr[i];
        max_value = val > max_value ? val : max_value;
        min_value = val < min_value ? val : min_value;
    }

    // The spread of int values can reach 2^32, so it needs 64 bits.
    // range >= 2 here: the array is unsorted, so it holds two distinct values.
    int64_t range = (int64_t)max_value - (int64_t)min_value + 1;

    // Offsets from the minimum are computed in unsigned arithmetic, which is
    // exact for any pair of ints (the true offset is always below 2^32).
    const uint32_t min_u = (uint32_t)min_value;

    // Narrow range: count each value and write the values straight back.
    // No scratch buffer or bucket pass is needed.
    if (range <= 2 * (int64_t)size) {
        int *counts = calloc((size_t)range, sizeof(int));
        if (counts) {
            for (i = 0; i < size; i++) counts[(uint32_t)arr[i] - min_u]++;
            int *out = arr;
            for (int v = 0; v < (int)range; v++) {
                int value = min_value + v;
                for (int c = counts[v]; c > 0; c--) *out++ = value;
            }
            free(counts);
            return;
        }
        // Allocation failed: carry on with the bucket pass below
    }

    // Aim for size / 2 buckets, then round the bucket width up to a power of
    // two so the bucket index is a single shift.
    //
    // The width starts as ceil(range / target) with target >= 2, and rounding
    // it up to a power of two still leaves it <= range - 1. So the minimum and
    // maximum always land in different buckets, and every recursive call works
    // on a strictly narrower range.
    int64_t bucket_size = (range - 1) / (size / 2) + 1;
    int shift = 0;
    while (((int64_t)1 << shift) < bucket_size) shift++;
    int bucket_count = (int)((range - 1) >> shift) + 1;

    // Allocate memory
    size_t total_size = size * sizeof(int) +
                        bucket_count * sizeof(int) * 3;
    char *memory = malloc(total_size);
    if (!memory) {
        // No scratch space: still sort, in place
        qsort(arr, size, sizeof(int), ahb_compare_ints);
        return;
    }

    int *buckets = (int*)memory;
    int *bucket_data = (int*)(buckets + size);
    int *bucket_offsets = bucket_data;
    int *bucket_sizes = bucket_data + bucket_count;
    int *positions = bucket_data + 2 * bucket_count;

    memset(bucket_data, 0, bucket_count * sizeof(int) * 3);

    // Count elements per bucket
    i = 0;
    for (; i + 4 <= size; i += 4) {
        int b1 = (int)(((uint32_t)arr[i] - min_u) >> shift);
        int b2 = (int)(((uint32_t)arr[i+1] - min_u) >> shift);
        int b3 = (int)(((uint32_t)arr[i+2] - min_u) >> shift);
        int b4 = (int)(((uint32_t)arr[i+3] - min_u) >> shift);

        bucket_sizes[b1]++;
        bucket_sizes[b2]++;
        bucket_sizes[b3]++;
        bucket_sizes[b4]++;
    }
    for (; i < size; i++) {
        bucket_sizes[((uint32_t)arr[i] - min_u) >> shift]++;
    }

    // Calculate offsets
    int running_total = 0;
    for (i = 0; i < bucket_count; i++) {
        bucket_offsets[i] = running_total;
        positions[i] = running_total;
        running_total += bucket_sizes[i];
    }

    // Distribute elements
    for (i = 0; i < size; i++) {
        buckets[positions[((uint32_t)arr[i] - min_u) >> shift]++] = arr[i];
    }

    // Sort individual buckets
    int threshold = size < 1024 ? size / 16 : 64;
    for (i = 0; i < bucket_count; i++) {
        int bsize = bucket_sizes[i];
        if (bsize <= 1) continue;
        
        int *bucket = &buckets[bucket_offsets[i]];
        if (bsize > threshold) {
            AHB_Sort(bucket, bsize);
        } else {
            insertion_sort(bucket, bsize);
        }
    }

    memcpy(arr, buckets, size * sizeof(int));
    free(memory);
}

#endif // AHB_SORT_H 