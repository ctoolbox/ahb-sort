// Benchmark: AHB_Sort against the original bucket sort it started from,
// the C library's qsort, and a byte-wise LSD radix sort.
//
//   make benchmark
//
// Every algorithm sorts a fresh copy of the same input. The best time of
// several runs is reported, and every output is checked against qsort.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "ahb_sort.h"

// ---- Baselines -------------------------------------------------------------

#define BUCKET_COUNT 10

// A simple insertion sort function for bucket_sort_original
void insertion_sort_helper(int *arr, int size) {
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

// Function to calculate variance of the array
double calculate_variance(int *arr, int size, double mean) {
    double variance = 0.0;
    for (int i = 0; i < size; i++) {
        variance += pow(arr[i] - mean, 2);
    }
    return variance / size;
}

// Original bucket sort function: the starting point AHB Sort was developed
// from, kept unchanged as a baseline. It only supports value ranges that fit
// in an int, so the benchmark inputs stay within [0, 1e9].
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
                insertion_sort_helper(&buckets[bucket_offsets[i]], bucket_sizes[i]);
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

int compare(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

void qsort_wrapper(int *arr, int size) {
    qsort(arr, size, sizeof(int), compare);
}

// LSD radix sort, one byte per pass. Flipping the sign bit makes the unsigned
// byte order match signed int order. Passes where every element has the same
// byte are skipped.
void lsd_radix_sort(int *arr, int size) {
    if (size <= 1) return;

    uint32_t *data = (uint32_t *)arr;
    uint32_t *scratch = (uint32_t *)malloc(size * sizeof(uint32_t));
    if (!scratch) {
        qsort_wrapper(arr, size);
        return;
    }

    size_t count[4][256] = {{0}};
    for (int i = 0; i < size; i++) {
        uint32_t value = data[i] ^ 0x80000000u;
        data[i] = value;
        count[0][value & 255]++;
        count[1][(value >> 8) & 255]++;
        count[2][(value >> 16) & 255]++;
        count[3][value >> 24]++;
    }

    uint32_t *source = data, *target = scratch;
    for (int pass = 0; pass < 4; pass++) {
        int shift = pass * 8;
        if (count[pass][(source[0] >> shift) & 255] == (size_t)size) continue;

        size_t offsets[256];
        size_t running_total = 0;
        for (int b = 0; b < 256; b++) {
            offsets[b] = running_total;
            running_total += count[pass][b];
        }
        for (int i = 0; i < size; i++) target[offsets[(source[i] >> shift) & 255]++] = source[i];

        uint32_t *swap = source;
        source = target;
        target = swap;
    }

    if (source != data) memcpy(data, source, size * sizeof(uint32_t));
    for (int i = 0; i < size; i++) data[i] ^= 0x80000000u;
    free(scratch);
}

typedef void (*sort_fn)(int *, int);

// AHB-Sort must stay first: the speedup columns are relative to it.
static const struct {
    const char *name;
    sort_fn sort;
} algorithms[] = {
    {"AHB-Sort", AHB_Sort},
    {"Bucket-Orig", bucket_sort_original},
    {"qsort", qsort_wrapper},
    {"Radix-256", lsd_radix_sort},
};
#define ALGORITHM_COUNT ((int)(sizeof(algorithms) / sizeof(algorithms[0])))

// ---- Inputs ----------------------------------------------------------------

// Deterministic xorshift generator, so runs are comparable across platforms
static uint64_t rng_state;

static uint32_t next_u32(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 16);
}

static void generate_random_narrow(int *arr, int size) {
    for (int i = 0; i < size; i++) arr[i] = (int)(next_u32() % 1000);
}

static void generate_random_wide(int *arr, int size) {
    for (int i = 0; i < size; i++) arr[i] = (int)(next_u32() % 1000000000);
}

static void generate_nearly_sorted(int *arr, int size) {
    for (int i = 0; i < size; i++) arr[i] = i + (int)(next_u32() % 10);
}

// 100 distinct values scattered over [0, 1e9)
static void generate_few_distinct(int *arr, int size) {
    int values[100];
    for (int i = 0; i < 100; i++) values[i] = (int)(next_u32() % 1000000000);
    for (int i = 0; i < size; i++) arr[i] = values[next_u32() % 100];
}

static void generate_exponential(int *arr, int size) {
    for (int i = 0; i < size; i++) {
        double unit = (next_u32() + 1.0) / 4294967297.0;
        arr[i] = (int)(-log(unit) * 1e6);
    }
}

// 99% of values in [0, 1000), 1% spread over [0, 1e9)
static void generate_cluster_with_outliers(int *arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = next_u32() % 100 == 0 ? (int)(next_u32() % 1000000000) : (int)(next_u32() % 1000);
    }
}

static const struct {
    const char *name;
    void (*generate)(int *, int);
} inputs[] = {
    {"Random, range 1000", generate_random_narrow},
    {"Random, range 1e9", generate_random_wide},
    {"Nearly sorted", generate_nearly_sorted},
    {"100 distinct values", generate_few_distinct},
    {"Exponential", generate_exponential},
    {"Cluster + 1% outliers", generate_cluster_with_outliers},
};
#define INPUT_COUNT ((int)(sizeof(inputs) / sizeof(inputs[0])))

static const int sizes[] = {1000, 10000, 100000, 1000000, 5000000};
#define SIZE_COUNT ((int)(sizeof(sizes) / sizeof(sizes[0])))

// ---- Measurement -----------------------------------------------------------

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

// More runs for small arrays, where each run is short and noisier
static int runs_for(int size) {
    if (size >= 5000000) return 7;
    if (size >= 1000000) return 15;
    return 41;
}

// Small arrays sort faster than the clock can resolve, so several copies are
// sorted back to back inside one timed region.
static int copies_for(int size) {
    int copies = 200000 / size;
    return copies < 1 ? 1 : copies;
}

static int failures = 0;

// Best time in ms for each algorithm to sort one copy of one input
static void measure(int input, int size, double best[ALGORITHM_COUNT]) {
    int copies = copies_for(size);
    size_t bytes = (size_t)size * sizeof(int);
    int *source = malloc(bytes), *expected = malloc(bytes), *work = malloc(bytes * copies);
    if (!source || !expected || !work) {
        fprintf(stderr, "out of memory\n");
        exit(2);
    }

    rng_state = 88172645463325252ULL + (uint64_t)input;
    inputs[input].generate(source, size);
    memcpy(expected, source, bytes);
    qsort(expected, size, sizeof(int), compare);

    for (int a = 0; a < ALGORITHM_COUNT; a++) best[a] = 1e300;

    int runs = runs_for(size);
    for (int run = 0; run < runs; run++) {
        for (int a = 0; a < ALGORITHM_COUNT; a++) {
            for (int c = 0; c < copies; c++) memcpy(work + (size_t)c * size, source, bytes);

            double start = now_ms();
            for (int c = 0; c < copies; c++) algorithms[a].sort(work + (size_t)c * size, size);
            double elapsed = (now_ms() - start) / copies;
            if (elapsed < best[a]) best[a] = elapsed;

            if (run == 0 && memcmp(work + (size_t)(copies - 1) * size, expected, bytes) != 0) {
                failures++;
                printf("Error: %s did not sort \"%s\" (%d elements) correctly.\n",
                       algorithms[a].name, inputs[input].name, size);
            }
        }
    }

    free(source);
    free(expected);
    free(work);
}

int main(void) {
    // Product of (other time / AHB time) per algorithm, for geometric means
    double overall[ALGORITHM_COUNT];
    for (int a = 0; a < ALGORITHM_COUNT; a++) overall[a] = 1.0;

    printf("Sorting Algorithm Benchmark Results\n");
    printf("===================================\n");
    printf("Times are the best run in ms. \"x\" columns: how many times faster\n");
    printf("AHB-Sort is than that algorithm (below 1.00x means AHB-Sort is slower).\n");

    for (int s = 0; s < SIZE_COUNT; s++) {
        int size = sizes[s];
        double per_size[ALGORITHM_COUNT];
        for (int a = 0; a < ALGORITHM_COUNT; a++) per_size[a] = 1.0;

        printf("\n%d elements (best of %d runs)\n", size, runs_for(size));
        printf("%-22s", "Input");
        for (int a = 0; a < ALGORITHM_COUNT; a++) printf(" %11s", algorithms[a].name);
        for (int a = 1; a < ALGORITHM_COUNT; a++) printf(" %13s", algorithms[a].name);
        printf("\n");

        for (int in = 0; in < INPUT_COUNT; in++) {
            double best[ALGORITHM_COUNT];
            measure(in, size, best);

            printf("%-22s", inputs[in].name);
            for (int a = 0; a < ALGORITHM_COUNT; a++) printf(" %11.4f", best[a]);
            for (int a = 1; a < ALGORITHM_COUNT; a++) {
                double ratio = best[a] / best[0];
                printf(" %12.2fx", ratio);
                per_size[a] *= ratio;
                overall[a] *= ratio;
            }
            printf("\n");
            fflush(stdout);
        }

        printf("%-22s", "Geometric mean");
        for (int a = 0; a < ALGORITHM_COUNT; a++) printf(" %11s", "");
        for (int a = 1; a < ALGORITHM_COUNT; a++) {
            printf(" %12.2fx", pow(per_size[a], 1.0 / INPUT_COUNT));
        }
        printf("\n");
    }

    printf("\nOverall (geometric mean over all sizes and inputs)\n");
    for (int a = 1; a < ALGORITHM_COUNT; a++) {
        printf("AHB-Sort vs %-12s %.2fx\n", algorithms[a].name,
               pow(overall[a], 1.0 / (SIZE_COUNT * INPUT_COUNT)));
    }

    if (failures) printf("\n%d outputs were WRONG\n", failures);
    return failures ? 1 : 0;
}
