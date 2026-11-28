// Timing harness for experimental AHB_Sort variants.
//
//   make variants                      (default sizes)
//   ./release/variant_bench 50000 2000000
//
// Each variant is generated from ahb_variant.inc. For every input the variants
// run interleaved, the best time of several rounds is reported, and every
// output is checked against qsort.

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "../ahb_sort.h"

// Only used by variants without power-of-two bucket widths
static inline int ahbv_clamp_index(int index, int bucket_count) {
    return index >= bucket_count ? bucket_count - 1 : index;
}

static inline void ahbv_insertion_sort(int *arr, int size) {
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

static int ahbv_compare_ints(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

// ---- Variants --------------------------------------------------------------
//
// "current" is the bucket route of ahb_sort.h expressed through the template.
// It does not include the header's radix route for wide, unstructured data,
// so "header" is much faster than "current" on the wide inputs below.
// The others:
//
//   sampled   the header before sparse counting, the spread-aware counting
//             limit and the bucket caps were added
//   no_caps   current without the bucket caps for large arrays
//   early     the first version of the caps: from 2^18 elements for data that
//             is not evenly spread, with weaker guards and a 1024-slot sparse
//             table. Fast on the inputs below, but up to 3x slower than
//             "sampled" on other shapes (two clusters, sorted runs).
//   wc16      current with software write combining in the capped scatter

#define AHB_FN v_current
#define AHB_OPT_COUNTING 8
#define AHB_OPT_POW2 1
#define AHB_OPT_DIVISOR 2
#define AHB_OPT_COMPACT 1
#define AHB_OPT_SAMPLE 1
#define AHB_OPT_FINAL_PASS 64
#define AHB_OPT_SPARSE 4096
#define AHB_OPT_COUNT_SPREAD 1
#define AHB_OPT_CAP_SKEW 8192
#define AHB_OPT_CAP_SKEW_AT 2097153
#define AHB_OPT_CAP_GUARDS 2
#define AHB_OPT_CAP_SPREAD_AT 2097152
#define AHB_OPT_CAP_SPREAD_DIV 4096
#define AHB_OPT_ORDERED_MIN 48
#define AHB_OPT_ORDERED_TURNS 7
#include "ahb_variant.inc"

#define AHB_FN v_sampled
#define AHB_OPT_COUNTING 8
#define AHB_OPT_POW2 1
#define AHB_OPT_DIVISOR 2
#define AHB_OPT_COMPACT 1
#define AHB_OPT_SAMPLE 1
#define AHB_OPT_FINAL_PASS 64
#include "ahb_variant.inc"

#define AHB_FN v_no_caps
#define AHB_OPT_COUNTING 8
#define AHB_OPT_POW2 1
#define AHB_OPT_DIVISOR 2
#define AHB_OPT_COMPACT 1
#define AHB_OPT_SAMPLE 1
#define AHB_OPT_FINAL_PASS 64
#define AHB_OPT_SPARSE 4096
#define AHB_OPT_COUNT_SPREAD 1
#include "ahb_variant.inc"

#define AHB_FN v_early
#define AHB_OPT_COUNTING 8
#define AHB_OPT_POW2 1
#define AHB_OPT_DIVISOR 2
#define AHB_OPT_COMPACT 1
#define AHB_OPT_SAMPLE 1
#define AHB_OPT_FINAL_PASS 64
#define AHB_OPT_SPARSE 1024
#define AHB_OPT_COUNT_SPREAD 1
#define AHB_OPT_CAP_SKEW 8192
#define AHB_OPT_CAP_SKEW_AT 262144
#define AHB_OPT_CAP_GUARDS 1
#define AHB_OPT_CAP_SPREAD_AT 2097152
#define AHB_OPT_CAP_SPREAD_DIV 4096
#include "ahb_variant.inc"

#define AHB_FN v_wc16
#define AHB_OPT_COUNTING 8
#define AHB_OPT_POW2 1
#define AHB_OPT_DIVISOR 2
#define AHB_OPT_COMPACT 1
#define AHB_OPT_SAMPLE 1
#define AHB_OPT_FINAL_PASS 64
#define AHB_OPT_SPARSE 4096
#define AHB_OPT_COUNT_SPREAD 1
#define AHB_OPT_CAP_SKEW 8192
#define AHB_OPT_CAP_SKEW_AT 2097153
#define AHB_OPT_CAP_GUARDS 2
#define AHB_OPT_CAP_SPREAD_AT 2097152
#define AHB_OPT_CAP_SPREAD_DIV 4096
#define AHB_OPT_ORDERED_MIN 48
#define AHB_OPT_ORDERED_TURNS 7
#define AHB_OPT_WC 16
#include "ahb_variant.inc"

static void header(int *arr, int size) { AHB_Sort(arr, size); }

typedef void (*sort_fn)(int *, int);

// The first entry is the reference that the others are compared against.
static const struct {
    const char *name;
    sort_fn sort;
} variants[] = {
    {"header", header},
    {"current", v_current},
    {"sampled", v_sampled},
    {"no_caps", v_no_caps},
    {"early", v_early},
    {"wc16", v_wc16},
};
#define VARIANT_COUNT ((int)(sizeof(variants) / sizeof(variants[0])))

// ---- Inputs ----------------------------------------------------------------

static uint64_t rng_state;

static uint32_t next_u32(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 16);
}

static double next_unit(void) { return (next_u32() + 1.0) / 4294967297.0; }

static void fill_uniform(int *arr, int size, uint32_t range) {
    for (int i = 0; i < size; i++) arr[i] = (int)(next_u32() % range);
}

static void in_uniform_1000(int *arr, int size) { fill_uniform(arr, size, 1000); }
static void in_uniform_half_n(int *arr, int size) { fill_uniform(arr, size, (uint32_t)size / 2 + 1); }
static void in_uniform_2n(int *arr, int size) { fill_uniform(arr, size, (uint32_t)size * 2); }
static void in_uniform_1_2n(int *arr, int size) { fill_uniform(arr, size, (uint32_t)size / 5 * 6); }
static void in_uniform_1e9(int *arr, int size) { fill_uniform(arr, size, 1000000000); }

static void in_uniform_full(int *arr, int size) {
    for (int i = 0; i < size; i++) arr[i] = (int)next_u32();
}

static void in_exponential(int *arr, int size) {
    for (int i = 0; i < size; i++) arr[i] = (int)(-log(next_unit()) * 1e6);
}

// 99% of values in [0, 1000), 1% spread up to 1e9
static void in_cluster_outliers(int *arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = next_u32() % 100 == 0 ? (int)(next_u32() % 1000000000) : (int)(next_u32() % 1000);
    }
}

static void in_nearly_sorted(int *arr, int size) {
    for (int i = 0; i < size; i++) arr[i] = i + (int)(next_u32() % 10);
}

// Ascending values with 1% of the elements swapped to random places
static void in_sorted_swapped(int *arr, int size) {
    for (int i = 0; i < size; i++) arr[i] = i * 3;
    for (int k = 0; k < size / 100; k++) {
        int a = (int)(next_u32() % (uint32_t)size), b = (int)(next_u32() % (uint32_t)size);
        int t = arr[a]; arr[a] = arr[b]; arr[b] = t;
    }
}

// 1000 distinct values scattered over the whole int range
static void in_thousand_distinct_wide(int *arr, int size) {
    static int values[1000];
    for (int i = 0; i < 1000; i++) values[i] = (int)next_u32();
    for (int i = 0; i < size; i++) arr[i] = values[next_u32() % 1000];
}

// 100 distinct values scattered over the whole int range
static void in_few_distinct_wide(int *arr, int size) {
    int values[100];
    for (int i = 0; i < 100; i++) values[i] = (int)next_u32();
    for (int i = 0; i < size; i++) arr[i] = values[next_u32() % 100];
}

static const struct {
    const char *name;
    void (*generate)(int *, int);
} inputs[] = {
    {"uniform, range 1000", in_uniform_1000},
    {"uniform, range n/2", in_uniform_half_n},
    {"uniform, range 2n", in_uniform_2n},
    {"uniform, range 1.2n", in_uniform_1_2n},
    {"uniform, range 1e9", in_uniform_1e9},
    {"uniform, full int", in_uniform_full},
    {"exponential", in_exponential},
    {"cluster + 1% outliers", in_cluster_outliers},
    {"nearly sorted", in_nearly_sorted},
    {"sorted, 1% swapped", in_sorted_swapped},
    {"100 distinct, wide", in_few_distinct_wide},
    {"1000 distinct, wide", in_thousand_distinct_wide},
};
#define INPUT_COUNT ((int)(sizeof(inputs) / sizeof(inputs[0])))

// ---- Measurement -----------------------------------------------------------

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static int compare_ints(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

static int failures = 0;

static void measure(int input, int size, int rounds, double best[VARIANT_COUNT]) {
    // Small arrays sort faster than the clock can resolve, so several copies
    // are sorted back to back inside one timed region.
    int copies = 200000 / size < 1 ? 1 : 200000 / size;
    size_t bytes = (size_t)size * sizeof(int);
    int *source = malloc(bytes), *expected = malloc(bytes), *work = malloc(bytes * copies);
    if (!source || !expected || !work) {
        fprintf(stderr, "out of memory\n");
        exit(2);
    }

    rng_state = 88172645463325252ULL + (uint64_t)input;
    inputs[input].generate(source, size);
    memcpy(expected, source, bytes);
    qsort(expected, size, sizeof(int), compare_ints);

    for (int v = 0; v < VARIANT_COUNT; v++) best[v] = 1e300;

    for (int round = 0; round < rounds; round++) {
        for (int v = 0; v < VARIANT_COUNT; v++) {
            for (int c = 0; c < copies; c++) memcpy(work + (size_t)c * size, source, bytes);
            double start = now_ms();
            for (int c = 0; c < copies; c++) variants[v].sort(work + (size_t)c * size, size);
            double elapsed = (now_ms() - start) / copies;
            if (elapsed < best[v]) best[v] = elapsed;

            if (round == 0 && memcmp(work, expected, bytes) != 0) {
                failures++;
                printf("WRONG OUTPUT: %s on %s, size %d\n", variants[v].name, inputs[input].name, size);
            }
        }
    }

    free(source);
    free(expected);
    free(work);
}

static void run_size(int size) {
    // More rounds for small arrays, where each run is short and noisier
    int rounds = size >= 1000000 ? 15 : 41;
    double product[VARIANT_COUNT];
    for (int v = 0; v < VARIANT_COUNT; v++) product[v] = 1.0;

    printf("\nn = %d   (best of %d rounds; ms for %s, then %% change vs %s, negative = faster)\n",
           size, rounds, variants[0].name, variants[0].name);
    printf("%-22s %9s", "input", variants[0].name);
    for (int v = 1; v < VARIANT_COUNT; v++) printf(" %8s", variants[v].name);
    printf("\n");

    for (int in = 0; in < INPUT_COUNT; in++) {
        double best[VARIANT_COUNT];
        measure(in, size, rounds, best);

        printf("%-22s %9.3f", inputs[in].name, best[0]);
        for (int v = 1; v < VARIANT_COUNT; v++) {
            printf(" %+7.0f%%", (best[v] / best[0] - 1.0) * 100.0);
            product[v] *= best[v] / best[0];
        }
        printf("\n");
        fflush(stdout);
    }

    printf("%-22s %9s", "geometric mean", "");
    for (int v = 1; v < VARIANT_COUNT; v++) {
        printf(" %+7.0f%%", (pow(product[v], 1.0 / INPUT_COUNT) - 1.0) * 100.0);
    }
    printf("\n");
}

int main(int argc, char **argv) {
    static const int default_sizes[] = {10000, 100000, 1000000};

    if (argc > 1) {
        for (int a = 1; a < argc; a++) {
            int size = atoi(argv[a]);
            if (size < 1) {
                fprintf(stderr, "usage: %s [size ...]\n", argv[0]);
                return 2;
            }
            run_size(size);
        }
    } else {
        for (size_t s = 0; s < sizeof(default_sizes) / sizeof(default_sizes[0]); s++) {
            run_size(default_sizes[s]);
        }
    }

    if (failures) printf("\n%d variant outputs were WRONG\n", failures);
    return failures ? 1 : 0;
}
