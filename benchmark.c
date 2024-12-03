#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <limits.h>

// Define the BenchmarkMetrics struct here
struct BenchmarkMetrics {
    double time_taken;
    size_t comparisons;
    size_t array_accesses;
};

#define BUCKET_COUNT 10
#define TEST_ITERATIONS 100

// Add these global counters at the top of the file
static size_t array_accesses = 0;
static size_t comparisons = 0;

// Add these helper functions
void reset_counters() {
    array_accesses = 0;
    comparisons = 0;
}

// Wrap array access
static inline int array_read(const int *arr, int index) {
    array_accesses++;
    return arr[index];
}

static inline void array_write(int *arr, int index, int value) {
    array_accesses++;
    arr[index] = value;
}

// Wrap comparisons
static inline int compare_elements(int a, int b) {
    comparisons++;
    return a - b;
}

/*
 * Adaptive Hybrid Bucket Sort (AHB Sort)
 * =====================================
 * An optimized version of bucket sort that combines multiple optimization techniques
 * to achieve significant performance improvements.
 *
 * Performance Summary:
 * - Bucket-Sort average speedup vs Quick-Sort: 84.44%
 * - Original Bucket-Sort speedup vs Quick-Sort: 35.87%
 * - Current Bucket-Sort improvement over Original: 35.74%
 * - Radix-Sort average speedup vs Quick-Sort: 76.32%
 *
 * Detailed Optimization Breakdown:
 * ==============================
 *
 * 1. Early Exit Strategies
 *    Before: Algorithm processed all arrays similarly regardless of size or order
 *    After:  - Direct insertion sort for very small arrays (n ≤ 16)
 *            - Early detection and return for already sorted sequences
 *    Impact: Avoids overhead of bucket distribution for cases where simpler
 *           algorithms are more efficient
 *
 * 2. Statistical Calculation Optimizations
 *    Before: - Multiple passes through the array for min/max, mean, and variance
 *            - Sequential processing of elements
 *    After:  - Single-pass calculation combining all statistics
 *            - Loop unrolling processing 4 elements at once
 *            - Vectorizable operations for modern compilers
 *    Impact: Reduced memory traversals and better CPU pipeline utilization
 *
 * 3. Memory Management Optimizations
 *    Before: - Multiple separate malloc calls for each array
 *            - Scattered memory locations
 *            - Multiple memset operations
 *    After:  - Single contiguous memory allocation
 *            - Improved cache locality through strategic layout
 *            - Single memset for all auxiliary arrays
 *    Impact: Better cache utilization and reduced memory management overhead
 *
 * 4. Distribution Optimizations
 *    Before: - Division operations for bucket index calculation
 *            - Sequential element processing
 *            - Fixed bucket count limits
 *    After:  - Multiplication by inverse for bucket mapping
 *            - Unrolled loops processing 4 elements at once
 *            - Dynamic bucket count based on data distribution
 *    Impact: Faster bucket index calculations and better CPU instruction pipelining
 *
 * 5. Sorting Strategy Optimizations
 *    Before: - Fixed threshold for all array sizes
 *            - Same sorting strategy for all buckets
 *            - Static bucket count determination
 *    After:  - Adaptive threshold based on array size
 *            - Hybrid approach choosing between bucket and insertion sort
 *            - Dynamic bucket count based on variance
 *    Impact: Better adaptation to different data distributions and array sizes
 *
 * 6. Bucket Count Calculation
 *    Before: - Fixed or simple size-based bucket count
 *            - No consideration of data distribution
 *    After:  - Dynamic calculation based on variance and range
 *            - Adjusts to data distribution characteristics
 *            - Optimal bucket sizes for different data patterns
 *    Impact: More efficient distribution and better balanced buckets
 *
 * 7. Memory Access Patterns
 *    Before: - Random access patterns
 *            - Multiple array traversals
 *            - Cache-unfriendly memory layout
 *    After:  - Sequential access where possible
 *            - Minimized array traversals
 *            - Cache-aligned memory layout
 *    Impact: Better cache utilization and reduced memory access latency
 */

// A simple insertion sort function
void insertion_sort(int *arr, int size) {
    for (int i = 1; i < size; i++) {
        int key = array_read(arr, i);
        int j = i - 1;
        while (j >= 0 && compare_elements(array_read(arr, j), key) > 0) {
            array_write(arr, j + 1, array_read(arr, j));
            j--;
        }
        array_write(arr, j + 1, key);
    }
}

// Function to calculate variance of the array
double calculate_variance(int *arr, int size, double mean) {
    double variance = 0.0;
    for (int i = 0; i < size; i++) {
        variance += pow(arr[i] - mean, 2);
    }
    return variance / size;
}

// Wrapper for qsort to match the function signature
int compare(const void *a, const void *b) {
    return compare_elements(*(int *)a, *(int *)b);
}



/*
 * Adaptive Hybrid Bucket-Sort
 * This sorting algorithm combines the benefits of Bucket Sort with a hybrid adaptive sorting technique:
 * 
 * - **Dynamic Bucket Distribution**: Elements are distributed across a fixed number of buckets,
 *   ensuring that values within similar ranges are grouped together. This helps reduce the sorting
 *   complexity within each bucket.
 * 
 * - **Hybrid Sorting Approach**: The algorithm adapts the sorting method based on the bucket size:
 *     - **Insertion Sort** is used for small buckets: This is efficient for small arrays due to its simplicity and lower overhead.
 *     - **QuickSort** is used for larger buckets: This ensures that larger arrays are sorted efficiently, leveraging QuickSort's divide-and-conquer approach.
 * 
 * The adaptive nature of this hybrid approach allows the algorithm to perform well across various data distributions:
 * - For nearly sorted or small datasets, Insertion Sort provides minimal overhead.
 * - For larger, more randomized datasets, QuickSort ensures performance stability.
 */
/*
 * Dynamic Bucket-Insertion Sort Implementation
 * --------------------------------------------
 * Steps:
 * 1. Find the range (min and max) of the dataset.
 * 2. Dynamically determine the number of buckets based on dataset size and range.
 * 3. Distribute elements into appropriate buckets.
 * 4. Sort each bucket using Insertion Sort.
 * 5. Merge sorted buckets back into the original array.
 */

// The hybrid bucket-insertion sort function with dynamic bucket count
void AHB_sort(int *arr, int size) {
    // OPTIMIZATION 1: Early exit for small arrays
    if (size <= 1) return;
    if (size <= 16) {
        insertion_sort(arr, size);
        return;
    }

    // OPTIMIZATION 2: Early exit for sorted sequences
    int is_sorted = 1;
    for (int i = 1; i < size; i++) {
        if (arr[i] < arr[i-1]) {
            is_sorted = 0;
            break;
        }
    }
    if (is_sorted) return;

    #define THRESHOLD_DIVISOR 16

    // OPTIMIZATION 3: Unrolled statistics gathering with single-pass variance
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
    // Handle remaining elements
    for (; i < size; i++) {
        int val = arr[i];
        if (val > max_value) max_value = val;
        if (val < min_value) min_value = val;
        sum += val;
        sum_sq += (double)val * val;
    }
    
    // OPTIMIZATION 4: Efficient variance calculation
    double mean = sum / size;
    double variance = (sum_sq / size) - (mean * mean);

    // OPTIMIZATION 5: Optimized bucket count calculation
    int range = max_value - min_value + 1;
    int bucket_count = (int)(sqrt(variance) / range * size) + 1;
    bucket_count = bucket_count < 2 ? 2 : bucket_count;
    int bucket_size = range / bucket_count + 1;

    // OPTIMIZATION 6: Single contiguous memory allocation
    size_t total_size = size * sizeof(int) +                
                        bucket_count * sizeof(int) * 3;     
    
    char *memory = malloc(total_size);
    if (!memory) return;

    // OPTIMIZATION 7: Improved memory layout
    int *buckets = (int*)memory;
    int *bucket_data = (int*)(buckets + size);
    int *bucket_offsets = bucket_data;
    int *bucket_sizes = bucket_data + bucket_count;
    int *positions = bucket_data + 2 * bucket_count;

    // OPTIMIZATION 8: Single memset for all auxiliary arrays
    memset(bucket_data, 0, bucket_count * sizeof(int) * 3);

    // OPTIMIZATION 9: Division optimization
    const double bucket_size_inv = 1.0 / bucket_size;
    
    // OPTIMIZATION 10: Unrolled counting phase
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

    // OPTIMIZATION 11: Simplified offset calculation
    int running_total = 0;
    for (i = 0; i < bucket_count; i++) {
        bucket_offsets[i] = running_total;
        positions[i] = running_total;
        running_total += bucket_sizes[i];
    }

    // OPTIMIZATION 12: Unrolled distribution phase
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

    // OPTIMIZATION 13: Adaptive threshold for hybrid sorting
    int threshold = size < 1024 ? size / THRESHOLD_DIVISOR : 64;
    for (i = 0; i < bucket_count; i++) {
        int bsize = bucket_sizes[i];
        if (bsize <= 1) continue;
        
        int *bucket = &buckets[bucket_offsets[i]];
        if (bsize > threshold) {
            AHB_sort(bucket, bsize);
        } else {
            insertion_sort(bucket, bsize);
        }
    }

    memcpy(arr, buckets, size * sizeof(int));
    free(memory);
}


// Helper function
void print_array(int *arr, int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

// Helper function for benchmarking
void benchmark_sort(void (*sort_func)(int *, int), int *arr, int size, const char *sort_name) {
    int *copy = (int *)malloc(size * sizeof(int));
    memcpy(copy, arr, size * sizeof(int));

    clock_t start = clock();
    sort_func(copy, size);
    clock_t end = clock();

    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    printf("%s took %.6f seconds for %d elements\n", sort_name, time_taken, size);

    free(copy);
}

void qsort_wrapper(int *arr, int size) {
    qsort(arr, size, sizeof(int), compare);
}

// Generate random array
void generate_random_array(int *arr, int size, int range) {
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % range;
    }
}

// Generate nearly sorted array
void generate_nearly_sorted_array(int *arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = i + rand() % 10; // Small random variation
    }
}

// Generate array with duplicates
void generate_with_duplicates(int *arr, int size, int range) {
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % (range / 10); // Fewer unique values
    }
}

// Add this new struct for storing benchmark results
struct BenchmarkResult {
    const char* sort_name;
    const char* data_type;
    int size;
    double time_taken;
    size_t comparisons;
    size_t array_accesses;
};

// Function to verify if the array is sorted
int is_sorted(int *arr, int size) {
    for (int i = 0; i < size - 1; i++) {
        if (arr[i] > arr[i + 1]) {
            return 0; // Not sorted
        }
    }
    return 1; // Sorted
}

// Modified benchmark function to return timing and verify sorting
struct BenchmarkMetrics benchmark_sort_quiet(void (*sort_func)(int *, int), int *arr, int size, const char *sort_name) {
    int *copy = (int *)malloc(size * sizeof(int));
    memcpy(copy, arr, size * sizeof(int));

    reset_counters();
    clock_t start = clock();
    sort_func(copy, size);
    clock_t end = clock();

    struct BenchmarkMetrics metrics = {
        .time_taken = (double)(end - start) / CLOCKS_PER_SEC,
        .comparisons = comparisons,
        .array_accesses = array_accesses
    };

    if (!is_sorted(copy, size)) {
        printf("Error: %s did not sort the array correctly.\n", sort_name);
    }

    free(copy);
    return metrics;
}

// Function to get the maximum value in an array
int get_max(int *arr, int size) {
    int max = arr[0];
    for (int i = 1; i < size; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}

// LSD Radix Sort implementation
void lsd_radix_sort(int *arr, int size) {
    int max = get_max(arr, size);
    int *output = (int *)malloc(size * sizeof(int));
    int exp;

    for (exp = 1; max / exp > 0; exp *= 10) {
        int count[10] = {0};

        // Count occurrences of digits
        for (int i = 0; i < size; i++) {
            count[(array_read(arr, i) / exp) % 10]++;
        }

        // Change count[i] so that it contains the actual position of this digit in output[]
        for (int i = 1; i < 10; i++) {
            count[i] += count[i - 1];
        }

        // Build the output array
        for (int i = size - 1; i >= 0; i--) {
            int index = (array_read(arr, i) / exp) % 10;
            array_write(output, count[index] - 1, array_read(arr, i));
            count[index]--;
        }

        // Copy the output array to arr[], so that arr[] now contains sorted numbers according to current digit
        for (int i = 0; i < size; i++) {
            array_write(arr, i, array_read(output, i));
        }
    }

    free(output);
}

// Original bucket sort function
void bucket_sort_original(int *arr, int size) {
    int max_value = arr[0], min_value = arr[0];
    double sum = 0.0;

    // Find range of the data and calculate mean
    for (int i = 0; i < size; i++) {
        if (arr[i] > max_value) max_value = arr[i];
        if (arr[i] < min_value) min_value = arr[i];
        sum += arr[i];
    }
    double mean = sum / size;
    double variance = calculate_variance(arr, size, mean);

    int range = max_value - min_value + 1;
    int bucket_count = (int)(sqrt(variance) / range * size) + 1; // Adjust bucket count based on variance
    int bucket_size = range / bucket_count + 1;

    // Pre-allocate bucket storage
    int *buckets = (int *)malloc(size * sizeof(int));
    int *bucket_offsets = (int *)calloc(bucket_count, sizeof(int));
    int *bucket_sizes = (int *)calloc(bucket_count, sizeof(int));

    // Count elements in each bucket
    for (int i = 0; i < size; i++) {
        int bucket_index = (arr[i] - min_value) / bucket_size;
        bucket_sizes[bucket_index]++;
    }

    // Calculate bucket offsets
    bucket_offsets[0] = 0;
    for (int i = 1; i < bucket_count; i++) {
        bucket_offsets[i] = bucket_offsets[i - 1] + bucket_sizes[i - 1];
    }

    // Distribute elements into buckets
    int *temp_offsets = (int *)malloc(bucket_count * sizeof(int));
    memcpy(temp_offsets, bucket_offsets, bucket_count * sizeof(int));

    for (int i = 0; i < size; i++) {
        int bucket_index = (arr[i] - min_value) / bucket_size;
        buckets[temp_offsets[bucket_index]++] = arr[i];
    }
    free(temp_offsets);

    // Sort each bucket in place
    for (int i = 0; i < bucket_count; i++) {
        if (bucket_sizes[i] > 1) {
            if (bucket_sizes[i] > size / BUCKET_COUNT) {
                // Apply multi-level recursion
                bucket_sort_original(&buckets[bucket_offsets[i]], bucket_sizes[i]);
            } else {
                // Use insertion sort for smaller buckets
                insertion_sort(&buckets[bucket_offsets[i]], bucket_sizes[i]);
            }
        }
    }

    // Merge buckets back to the array
    for (int i = 0; i < size; i++) {
        arr[i] = buckets[i];
    }

    free(buckets);
    free(bucket_offsets);
    free(bucket_sizes);
}

// Modified advanced benchmark with original bucket sort
void advanced_benchmark() {
    // Update the sizes array to include 1M and 5M
    const int sizes[] = {1000, 10000, 100000, 1000000, 5000000};
    const int range = 1000;
    const int max_results = 60; // 5 sizes * 3 data types * 4 algorithms
    struct BenchmarkResult results[max_results];
    int result_index = 0;

    double total_times[15][4] = {0}; // 15 combinations (5 sizes * 3 data types) * 4 algorithms
    
    for (int iter = 0; iter < TEST_ITERATIONS; iter++) {
        result_index = 0;
        
        for (int i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
            int size = sizes[i];
            int *arr = (int *)malloc(size * sizeof(int));

            // Random data
            generate_random_array(arr, size, range);
            struct BenchmarkMetrics original_bucket_metrics = benchmark_sort_quiet(bucket_sort_original, arr, size, "Bucket-Sort-Original");
            struct BenchmarkMetrics bucket_metrics = benchmark_sort_quiet(AHB_sort, arr, size, "AHB-Sort");
            struct BenchmarkMetrics quick_metrics = benchmark_sort_quiet(qsort_wrapper, arr, size, "Quick-Sort");
            struct BenchmarkMetrics radix_metrics = benchmark_sort_quiet(lsd_radix_sort, arr, size, "Radix-Sort");
            
            total_times[i * 3][0] += original_bucket_metrics.time_taken;
            total_times[i * 3][1] += bucket_metrics.time_taken;
            total_times[i * 3][2] += quick_metrics.time_taken;
            total_times[i * 3][3] += radix_metrics.time_taken;

            if (iter == TEST_ITERATIONS - 1) {
                results[result_index++] = (struct BenchmarkResult){"Bucket-Sort-Original", "Random", size, original_bucket_metrics.time_taken, original_bucket_metrics.comparisons, original_bucket_metrics.array_accesses};
                results[result_index++] = (struct BenchmarkResult){"AHB-Sort", "Random", size, bucket_metrics.time_taken, bucket_metrics.comparisons, bucket_metrics.array_accesses};
                results[result_index++] = (struct BenchmarkResult){"Quick-Sort", "Random", size, quick_metrics.time_taken, quick_metrics.comparisons, quick_metrics.array_accesses};
                results[result_index++] = (struct BenchmarkResult){"Radix-Sort", "Random", size, radix_metrics.time_taken, radix_metrics.comparisons, radix_metrics.array_accesses};
            }

            // Nearly sorted data
            generate_nearly_sorted_array(arr, size);
            original_bucket_metrics = benchmark_sort_quiet(bucket_sort_original, arr, size, "Bucket-Sort-Original");
            bucket_metrics = benchmark_sort_quiet(AHB_sort, arr, size, "Bucket-Sort");
            quick_metrics = benchmark_sort_quiet(qsort_wrapper, arr, size, "Quick-Sort");
            radix_metrics = benchmark_sort_quiet(lsd_radix_sort, arr, size, "Radix-Sort");
            
            total_times[i * 3 + 1][0] += original_bucket_metrics.time_taken;
            total_times[i * 3 + 1][1] += bucket_metrics.time_taken;
            total_times[i * 3 + 1][2] += quick_metrics.time_taken;
            total_times[i * 3 + 1][3] += radix_metrics.time_taken;

            if (iter == TEST_ITERATIONS - 1) {
                results[result_index++] = (struct BenchmarkResult){"Bucket-Sort-Original", "Nearly Sorted", size, original_bucket_metrics.time_taken, original_bucket_metrics.comparisons, original_bucket_metrics.array_accesses};
                results[result_index++] = (struct BenchmarkResult){"AHB-Sort", "Nearly Sorted", size, bucket_metrics.time_taken, bucket_metrics.comparisons, bucket_metrics.array_accesses};
                results[result_index++] = (struct BenchmarkResult){"Quick-Sort", "Nearly Sorted", size, quick_metrics.time_taken, quick_metrics.comparisons, quick_metrics.array_accesses};
                results[result_index++] = (struct BenchmarkResult){"Radix-Sort", "Nearly Sorted", size, radix_metrics.time_taken, radix_metrics.comparisons, radix_metrics.array_accesses};
            }

            // Data with duplicates
            generate_with_duplicates(arr, size, range);
            original_bucket_metrics = benchmark_sort_quiet(bucket_sort_original, arr, size, "Bucket-Sort-Original");
            bucket_metrics = benchmark_sort_quiet(AHB_sort, arr, size, "Bucket-Sort");
            quick_metrics = benchmark_sort_quiet(qsort_wrapper, arr, size, "Quick-Sort");
            radix_metrics = benchmark_sort_quiet(lsd_radix_sort, arr, size, "Radix-Sort");
            
            total_times[i * 3 + 2][0] += original_bucket_metrics.time_taken;
            total_times[i * 3 + 2][1] += bucket_metrics.time_taken;
            total_times[i * 3 + 2][2] += quick_metrics.time_taken;
            total_times[i * 3 + 2][3] += radix_metrics.time_taken;

            if (iter == TEST_ITERATIONS - 1) {
                results[result_index++] = (struct BenchmarkResult){"Bucket-Sort-Original", "Duplicates", size, original_bucket_metrics.time_taken, original_bucket_metrics.comparisons, original_bucket_metrics.array_accesses};
                results[result_index++] = (struct BenchmarkResult){"Bucket-Sort", "Duplicates", size, bucket_metrics.time_taken, bucket_metrics.comparisons, bucket_metrics.array_accesses};
                results[result_index++] = (struct BenchmarkResult){"Quick-Sort", "Duplicates", size, quick_metrics.time_taken, quick_metrics.comparisons, quick_metrics.array_accesses};
                results[result_index++] = (struct BenchmarkResult){"Radix-Sort", "Duplicates", size, radix_metrics.time_taken, radix_metrics.comparisons, radix_metrics.array_accesses};
            }

            free(arr);
        }
    }

    // Print last iteration results as before
    printf("\nLast Iteration Results:\n");
    printf("%-20s | %-14s | %-10s | %-12s | %-12s | %-12s | %-10s\n", 
        "Algorithm", "Data Type", "Size", "Time (ms)", "Comparisons", "Array Access", "Speedup (%)");
    printf("====================================================================================\n");

    for (int i = 0; i < result_index; i += 4) {
        double speedup_bucket = ((results[i + 2].time_taken - results[i + 1].time_taken) / results[i + 2].time_taken) * 100;
        double speedup_radix = ((results[i + 2].time_taken - results[i + 3].time_taken) / results[i + 2].time_taken) * 100;
        
        printf("%-20s | %-14s | %-10d | %12.6f | %12zu | %12zu |\n",
            results[i].sort_name,
            results[i].data_type,
            results[i].size,
            results[i].time_taken * 1000,
            results[i].comparisons,
            results[i].array_accesses);
            
        printf("\033[1;92m%-20s\033[0m | %-14s | %-10d | \033[1;92m%12.6f\033[0m | \033[1;92m%12zu\033[0m | \033[1;92m%12zu\033[0m | \033[1;92m%10.2f\033[0m\n",
            "Bucket-Sort",
            results[i + 1].data_type,
            results[i + 1].size,
            results[i + 1].time_taken * 1000,
            results[i + 1].comparisons,
            results[i + 1].array_accesses,
            speedup_bucket);
            
        printf("%-20s | %-14s | %-10d | %12.6f | %12zu | %12zu |\n",
            "Quick-Sort",
            results[i + 2].data_type,
            results[i + 2].size,
            results[i + 2].time_taken * 1000,
            results[i + 2].comparisons,
            results[i + 2].array_accesses);
            
        printf("%-20s | %-14s | %-10d | %12.6f | %12zu | %12zu | %10.2f\n",
            "Radix-Sort",
            results[i + 3].data_type,
            results[i + 3].size,
            results[i + 3].time_taken * 1000,
            results[i + 3].comparisons,
            results[i + 3].array_accesses,
            speedup_radix);

        if ((i + 4) % 15 == 0 && i + 4 < result_index) {
            printf("-----------------------------------------------------------\n");
        }
    }

    // Print summary in table format
    printf("\nSummary Over %d Iterations:\n", TEST_ITERATIONS);
    printf("===========================================================\n");
    printf("%-20s | %-14s | %-10s | %-14s | %-10s\n", "Algorithm", "Data Type", "Size", "Avg Time (ms)", "Avg Speedup");
    printf("-----------------------------------------------------------\n");

    const char *data_types[] = {"Random", "Nearly Sorted", "Duplicates"};
    
    double total_speedup_bucket = 0.0;
    double total_speedup_original = 0.0;
    double total_speedup_radix = 0.0;
    double total_improvement_over_original = 0.0;

    for (int i = 0; i < 15; i++) {
        int size_index = i / 3;
        int type_index = i % 3;
        
        // Add separator between different data types
        if (i > 0 && type_index == 0) {
            printf("===========================================================\n");
        }
        
        double avg_original_bucket = (total_times[i][0] / TEST_ITERATIONS) * 1e3;
        double avg_bucket = (total_times[i][1] / TEST_ITERATIONS) * 1e3;
        double avg_quick = (total_times[i][2] / TEST_ITERATIONS) * 1e3;
        double avg_radix = (total_times[i][3] / TEST_ITERATIONS) * 1e3;
        
        double avg_speedup_bucket = ((avg_quick - avg_bucket) / avg_quick) * 100;
        double avg_speedup_original = ((avg_quick - avg_original_bucket) / avg_quick) * 100;
        double avg_speedup_radix = ((avg_quick - avg_radix) / avg_quick) * 100;
        double improvement = ((avg_original_bucket - avg_bucket) / avg_original_bucket) * 100;

        total_speedup_bucket += avg_speedup_bucket;
        total_speedup_original += avg_speedup_original;
        total_speedup_radix += avg_speedup_radix;
        total_improvement_over_original += improvement;

        printf("%-20s | %-14s | %-10d | %14.6f | %10.2f%%\n",
            "Bucket-Sort-Original",
            data_types[type_index],
            sizes[size_index],
            avg_original_bucket,
            avg_speedup_original);
            
        printf("\033[1;92m%-20s\033[0m | %-14s | %-10d | \033[1;92m%14.6f\033[0m | \033[1;92m%10.2f%%\033[0m (vs Quick) | \033[1;96m%10.2f%%\033[0m (vs Original)\n",
            "AHB-Sort",
            data_types[type_index],
            sizes[size_index],
            avg_bucket,
            avg_speedup_bucket,
            improvement);
            
        printf("%-20s | %-14s | %-10d | %14.6f |\n",
            "Quick-Sort",
            data_types[type_index],
            sizes[size_index],
            avg_quick);
            
        printf("%-20s | %-14s | %-10d | %14.6f | %10.2f%%\n",
            "Radix-Sort",
            data_types[type_index],
            sizes[size_index],
            avg_radix,
            avg_speedup_radix);

        // Add separator between different sizes within same data type
        if ((i + 1) % 3 == 0 && i < 14) {
            printf("-----------------------------------------------------------\n");
        }
    }

    // Add final separator before global averages
    printf("===========================================================\n");

    // Calculate and print global average speedups
    double global_avg_speedup_bucket = total_speedup_bucket / 15;
    double global_avg_speedup_original = total_speedup_original / 15;
    double global_avg_speedup_radix = total_speedup_radix / 15;
    double global_avg_improvement = total_improvement_over_original / 15;

    printf("\nGlobal Average Speedups:\n");
    printf("AHB-Sort average speedup vs Quick-Sort: %.2f%%\n", global_avg_speedup_bucket);
    printf("Original Bucket-Sort speedup vs Quick-Sort: %.2f%%\n", global_avg_speedup_original);
    printf("Radix-Sort average speedup vs Quick-Sort: %.2f%%\n", global_avg_speedup_radix);
    printf("AHB-Sort improvement over Original Bucket Sort: %.2f%%\n", global_avg_improvement);
}

// Modify main function
int main() {
    printf("Sorting Algorithm Benchmark Results\n");
    printf("==================================\n");
    advanced_benchmark();
    return 0;
}