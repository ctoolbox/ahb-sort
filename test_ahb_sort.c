// Correctness tests for AHB_Sort: every case is checked against qsort.
// Build and run with `make test` (uses AddressSanitizer + UBSan).

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ahb_sort.h"

// Deterministic xorshift generator so failures reproduce on every platform
static uint64_t rng_state = 88172645463325252ULL;

static uint32_t next_u32(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 16);
}

static int compare_ints(const void *a, const void *b) {
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

typedef void (*generator)(int *arr, int size);

static void gen_small_range(int *arr, int size) {
    for (int i = 0; i < size; i++) arr[i] = (int)(next_u32() % 1000);
}

static void gen_small_negatives(int *arr, int size) {
    for (int i = 0; i < size; i++) arr[i] = (int)(next_u32() % 2001) - 1000;
}

static void gen_full_int_range(int *arr, int size) {
    for (int i = 0; i < size; i++) arr[i] = (int)next_u32();
}

static void gen_int_extremes(int *arr, int size) {
    static const int values[] = {INT_MIN, INT_MIN + 1, -1, 0, 1, INT_MAX - 1, INT_MAX};
    for (int i = 0; i < size; i++) arr[i] = values[next_u32() % 7];
}

static void gen_zero_to_int_max(int *arr, int size) {
    for (int i = 0; i < size; i++) arr[i] = (int)(next_u32() % INT_MAX);
    if (size > 1) {
        arr[0] = INT_MAX;
        arr[size - 1] = 0;
    }
}

static void gen_two_values_far_apart(int *arr, int size) {
    for (int i = 0; i < size; i++) arr[i] = (next_u32() & 1) ? 2000000000 : -2000000000;
}

// Large magnitude, tiny spread: used to recurse forever
static void gen_large_positive_pair(int *arr, int size) {
    for (int i = 0; i < size; i++) arr[i] = 2000000000 + (int)(next_u32() & 1);
}

static void gen_large_negative_pair(int *arr, int size) {
    for (int i = 0; i < size; i++) arr[i] = -2000000000 - (int)(next_u32() & 1);
}

static void gen_large_positive_triple(int *arr, int size) {
    for (int i = 0; i < size; i++) arr[i] = 1500000000 + (int)(next_u32() % 3);
}

static void gen_int_max_pair(int *arr, int size) {
    for (int i = 0; i < size; i++) arr[i] = INT_MAX - (int)(next_u32() & 1);
}

static void gen_single_low_outlier(int *arr, int size) {
    for (int i = 0; i < size; i++) arr[i] = 7;
    if (size > 0) arr[size - 1] = 6;
}

static void gen_cluster_with_outlier(int *arr, int size) {
    for (int i = 0; i < size; i++) arr[i] = (int)(next_u32() % 100);
    if (size > 0) arr[size / 2] = 1000000000;
}

static void gen_powers_of_two(int *arr, int size) {
    for (int i = 0; i < size; i++) arr[i] = 1 << (next_u32() % 31);
}

static void gen_all_equal(int *arr, int size) {
    for (int i = 0; i < size; i++) arr[i] = 42;
}

static void gen_sorted(int *arr, int size) {
    for (int i = 0; i < size; i++) arr[i] = i;
}

static void gen_reversed(int *arr, int size) {
    for (int i = 0; i < size; i++) arr[i] = size - i;
}

static void gen_nearly_sorted(int *arr, int size) {
    for (int i = 0; i < size; i++) arr[i] = i + (int)(next_u32() % 10);
}

// Value range of exactly 2 * size: the widest range that takes the counting path
static void gen_range_twice_size(int *arr, int size) {
    for (int i = 0; i < size; i++) arr[i] = (int)(next_u32() % (2 * (uint32_t)size)) - size;
    if (size > 1) {
        arr[0] = size - 1;
        arr[size - 1] = -size;
    }
}

// One wider than that: the narrowest range that takes the bucket path
static void gen_range_twice_size_plus_one(int *arr, int size) {
    gen_range_twice_size(arr, size);
    if (size > 1) arr[0] = size;
}

static const struct {
    const char *name;
    generator generate;
} cases[] = {
    {"small range", gen_small_range},
    {"small negatives", gen_small_negatives},
    {"full int range", gen_full_int_range},
    {"INT_MIN/INT_MAX extremes", gen_int_extremes},
    {"0..INT_MAX", gen_zero_to_int_max},
    {"two values 4e9 apart", gen_two_values_far_apart},
    {"2e9 + {0,1}", gen_large_positive_pair},
    {"-2e9 - {0,1}", gen_large_negative_pair},
    {"1.5e9 + {0,1,2}", gen_large_positive_triple},
    {"INT_MAX - {0,1}", gen_int_max_pair},
    {"single low outlier", gen_single_low_outlier},
    {"cluster with outlier", gen_cluster_with_outlier},
    {"powers of two", gen_powers_of_two},
    {"all equal", gen_all_equal},
    {"already sorted", gen_sorted},
    {"reversed", gen_reversed},
    {"nearly sorted", gen_nearly_sorted},
    {"range == 2 * size", gen_range_twice_size},
    {"range == 2 * size + 1", gen_range_twice_size_plus_one},
};

static const int sizes[] = {0, 1, 2, 3, 15, 16, 17, 18, 19, 20, 33, 100,
                            1000, 1023, 1024, 1025, 10000, 200000};

int main(void) {
    int failures = 0, checks = 0;

    for (size_t c = 0; c < sizeof(cases) / sizeof(cases[0]); c++) {
        for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); s++) {
            int size = sizes[s];
            size_t bytes = (size_t)size * sizeof(int);
            int *actual = malloc(bytes + 1);
            int *expected = malloc(bytes + 1);
            if (!actual || !expected) {
                fprintf(stderr, "out of memory\n");
                return 2;
            }

            cases[c].generate(actual, size);
            memcpy(expected, actual, bytes);
            qsort(expected, size, sizeof(int), compare_ints);

            AHB_Sort(actual, size);

            checks++;
            if (memcmp(actual, expected, bytes) != 0) {
                failures++;
                printf("FAIL: %s, size %d\n", cases[c].name, size);
            }

            free(actual);
            free(expected);
        }
    }

    printf("%d/%d checks passed\n", checks - failures, checks);
    return failures ? 1 : 0;
}
