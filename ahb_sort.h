#ifndef AHB_SORT_H
#define AHB_SORT_H

#include <stdlib.h>
#include <string.h>
#include <math.h>

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

    #define THRESHOLD_DIVISOR 16

    // Statistics gathering with loop unrolling
    int max_value = arr[0], min_value = arr[0];
    double sum = 0.0, sum_sq = 0.0;
    int i = 0;
    for (; i + 4 <= size; i += 4) {
        int v1 = arr[i], v2 = arr[i+1], v3 = arr[i+2], v4 = arr[i+3];
        
        max_value = v1 > max_value ? v1 : max_value;
        max_value = v2 > max_value ? v2 : max_value;
        max_value = v3 > max_value ? v3 : max_value;
        max_value = v4 > max_value ? v4 : max_value;
        
        min_value = v1 < min_value ? v1 : min_value;
        min_value = v2 < min_value ? v2 : min_value;
        min_value = v3 < min_value ? v3 : min_value;
        min_value = v4 < min_value ? v4 : min_value;
        
        sum += v1 + v2 + v3 + v4;
        sum_sq += (double)v1 * v1 + (double)v2 * v2 + 
                  (double)v3 * v3 + (double)v4 * v4;
    }
    for (; i < size; i++) {
        int val = arr[i];
        if (val > max_value) max_value = val;
        if (val < min_value) min_value = val;
        sum += val;
        sum_sq += (double)val * val;
    }
    
    // Calculate variance and bucket count
    double mean = sum / size;
    double variance = (sum_sq / size) - (mean * mean);
    int range = max_value - min_value + 1;
    int bucket_count = (int)(sqrt(variance) / range * size) + 1;
    bucket_count = bucket_count < 2 ? 2 : bucket_count;
    int bucket_size = range / bucket_count + 1;

    // Allocate memory
    size_t total_size = size * sizeof(int) +                
                        bucket_count * sizeof(int) * 3;     
    char *memory = malloc(total_size);
    if (!memory) return;

    int *buckets = (int*)memory;
    int *bucket_data = (int*)(buckets + size);
    int *bucket_offsets = bucket_data;
    int *bucket_sizes = bucket_data + bucket_count;
    int *positions = bucket_data + 2 * bucket_count;

    memset(bucket_data, 0, bucket_count * sizeof(int) * 3);

    // Distribution phase
    const double bucket_size_inv = 1.0 / bucket_size;
    i = 0;
    for (; i + 4 <= size; i += 4) {
        int v1 = arr[i] - min_value;
        int v2 = arr[i+1] - min_value;
        int v3 = arr[i+2] - min_value;
        int v4 = arr[i+3] - min_value;

        int b1 = (int)(v1 * bucket_size_inv);
        int b2 = (int)(v2 * bucket_size_inv);
        int b3 = (int)(v3 * bucket_size_inv);
        int b4 = (int)(v4 * bucket_size_inv);

        b1 = b1 >= bucket_count ? bucket_count - 1 : b1;
        b2 = b2 >= bucket_count ? bucket_count - 1 : b2;
        b3 = b3 >= bucket_count ? bucket_count - 1 : b3;
        b4 = b4 >= bucket_count ? bucket_count - 1 : b4;

        bucket_sizes[b1]++;
        bucket_sizes[b2]++;
        bucket_sizes[b3]++;
        bucket_sizes[b4]++;
    }
    for (; i < size; i++) {
        int val = arr[i] - min_value;
        int bucket_index = (int)(val * bucket_size_inv);
        bucket_index = bucket_index >= bucket_count ? bucket_count - 1 : bucket_index;
        bucket_sizes[bucket_index]++;
    }

    // Calculate offsets
    int running_total = 0;
    for (i = 0; i < bucket_count; i++) {
        bucket_offsets[i] = running_total;
        positions[i] = running_total;
        running_total += bucket_sizes[i];
    }

    // Distribute elements
    i = 0;
    for (; i + 4 <= size; i += 4) {
        for (int j = 0; j < 4; j++) {
            int val = arr[i+j] - min_value;
            int bucket_index = (int)(val * bucket_size_inv);
            bucket_index = bucket_index >= bucket_count ? bucket_count - 1 : bucket_index;
            buckets[positions[bucket_index]++] = arr[i+j];
        }
    }
    for (; i < size; i++) {
        int val = arr[i] - min_value;
        int bucket_index = (int)(val * bucket_size_inv);
        bucket_index = bucket_index >= bucket_count ? bucket_count - 1 : bucket_index;
        buckets[positions[bucket_index]++] = arr[i];
    }

    // Sort individual buckets
    int threshold = size < 1024 ? size / THRESHOLD_DIVISOR : 64;
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