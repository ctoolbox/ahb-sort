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

static inline int ahbv_clamp_index(int index, int bucket_count) {
    return index >= bucket_count ? bucket_count - 1 : index;
}

// ---- Variants --------------------------------------------------------------

#define AHB_FN v_count4
#define AHB_OPT_COUNTING 4
#include "ahb_variant.inc"

#define AHB_FN v_count8
#define AHB_OPT_COUNTING 8
#include "ahb_variant.inc"

#define AHB_FN v_pow2up
#define AHB_OPT_POW2 1
#include "ahb_variant.inc"

#define AHB_FN v_div2
#define AHB_OPT_DIVISOR 2
#include "ahb_variant.inc"

#define AHB_FN v_c8_p2up
#define AHB_OPT_COUNTING 8
#define AHB_OPT_POW2 1
#include "ahb_variant.inc"

#define AHB_FN v_c8_p2dn
#define AHB_OPT_COUNTING 8
#define AHB_OPT_POW2 2
#include "ahb_variant.inc"

#define AHB_FN v_c8_div2
#define AHB_OPT_COUNTING 8
#define AHB_OPT_DIVISOR 2
#include "ahb_variant.inc"

#define AHB_FN v_c8_up_d2
#define AHB_OPT_COUNTING 8
#define AHB_OPT_POW2 1
#define AHB_OPT_DIVISOR 2
#include "ahb_variant.inc"

#define AHB_FN v_c8_dn_d2
#define AHB_OPT_COUNTING 8
#define AHB_OPT_POW2 2
#define AHB_OPT_DIVISOR 2
#include "ahb_variant.inc"

#define AHB_FN v_c8_dn_d4
#define AHB_OPT_COUNTING 8
#define AHB_OPT_POW2 2
#define AHB_OPT_DIVISOR 4
#include "ahb_variant.inc"

static void shipped(int *arr, int size) { AHB_Sort(arr, size); }

typedef void (*sort_fn)(int *, int);

// The first entry is the reference that the others are compared against.
static const struct {
    const char *name;
    sort_fn sort;
} variants[] = {
    {"shipped", shipped},
    {"count4", v_count4},
    {"count8", v_count8},
    {"pow2up", v_pow2up},
    {"div2", v_div2},
    {"c8_p2up", v_c8_p2up},
    {"c8_p2dn", v_c8_p2dn},
    {"c8_div2", v_c8_div2},
    {"c8_up_d2", v_c8_up_d2},
    {"c8_dn_d2", v_c8_dn_d2},
    {"c8_dn_d4", v_c8_dn_d4},
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
    {"uniform, range 1e9", in_uniform_1e9},
    {"uniform, full int", in_uniform_full},
    {"exponential", in_exponential},
    {"cluster + 1% outliers", in_cluster_outliers},
    {"nearly sorted", in_nearly_sorted},
    {"100 distinct, wide", in_few_distinct_wide},
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
    size_t bytes = (size_t)size * sizeof(int);
    int *source = malloc(bytes), *expected = malloc(bytes), *work = malloc(bytes);
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
            memcpy(work, source, bytes);
            double start = now_ms();
            variants[v].sort(work, size);
            double elapsed = now_ms() - start;
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
    int rounds = size >= 1000000 ? 15 : size >= 100000 ? 41 : 201;
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
