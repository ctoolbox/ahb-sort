// Correctness tests for ahb_sort.h: every case is checked against qsort,
// for all four element types.
// Build and run with `make test` (uses AddressSanitizer + UBSan).
//
// Test inputs are generated as 64-bit offsets from the smallest value of the
// element type. Offset order is the same as element order for every type, so
// one set of patterns covers signed and unsigned, 32-bit and 64-bit.

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ahb_sort.h"

// Deterministic xorshift generator so failures reproduce on every platform
static uint64_t rng_state = 88172645463325252ULL;

static uint64_t next_u64(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

static int compare_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

// ---- Patterns --------------------------------------------------------------
//
// Each fills out[0..count) with offsets in [0, top], where top is the largest
// offset the element type can hold (UINT32_MAX or UINT64_MAX). For signed
// types, offset top / 2 + 1 is the value zero.

typedef void (*pattern)(uint64_t *out, size_t count, uint64_t top);

static void pat_small_range_at_bottom(uint64_t *out, size_t count, uint64_t top) {
    (void)top;
    for (size_t i = 0; i < count; i++) out[i] = next_u64() % 1000;
}

static void pat_small_range_around_middle(uint64_t *out, size_t count, uint64_t top) {
    uint64_t middle = top / 2 + 1;
    for (size_t i = 0; i < count; i++) out[i] = middle - 1000 + next_u64() % 2001;
}

static void pat_full_range(uint64_t *out, size_t count, uint64_t top) {
    for (size_t i = 0; i < count; i++) out[i] = next_u64() & top;
}

static void pat_upper_half(uint64_t *out, size_t count, uint64_t top) {
    uint64_t middle = top / 2 + 1;
    for (size_t i = 0; i < count; i++) out[i] = middle + (next_u64() & (top / 2));
    if (count > 1) {
        out[0] = top;
        out[count - 1] = middle;
    }
}

static void pat_extremes(uint64_t *out, size_t count, uint64_t top) {
    uint64_t middle = top / 2 + 1;
    uint64_t values[] = {0, 1, middle - 1, middle, middle + 1, top - 1, top};
    for (size_t i = 0; i < count; i++) out[i] = values[next_u64() % 7];
}

static void pat_two_values_far_apart(uint64_t *out, size_t count, uint64_t top) {
    for (size_t i = 0; i < count; i++) out[i] = (next_u64() & 1) ? top - 100 : 100;
}

// Large magnitude, tiny spread
static void pat_pair_at_top(uint64_t *out, size_t count, uint64_t top) {
    for (size_t i = 0; i < count; i++) out[i] = top - (next_u64() & 1);
}

static void pat_pair_at_bottom(uint64_t *out, size_t count, uint64_t top) {
    (void)top;
    for (size_t i = 0; i < count; i++) out[i] = next_u64() & 1;
}

static void pat_triple_near_top(uint64_t *out, size_t count, uint64_t top) {
    for (size_t i = 0; i < count; i++) out[i] = top - top / 4 + next_u64() % 3;
}

static void pat_single_low_outlier(uint64_t *out, size_t count, uint64_t top) {
    uint64_t middle = top / 2 + 1;
    for (size_t i = 0; i < count; i++) out[i] = middle + 7;
    if (count > 0) out[count - 1] = middle + 6;
}

static void pat_cluster_with_outlier(uint64_t *out, size_t count, uint64_t top) {
    for (size_t i = 0; i < count; i++) out[i] = next_u64() % 100;
    if (count > 0) out[count / 2] = top - top / 3;
}

static void pat_powers_of_two(uint64_t *out, size_t count, uint64_t top) {
    unsigned bits = top == UINT32_MAX ? 32 : 64;
    for (size_t i = 0; i < count; i++) out[i] = (uint64_t)1 << (next_u64() % bits);
}

static void pat_all_equal(uint64_t *out, size_t count, uint64_t top) {
    for (size_t i = 0; i < count; i++) out[i] = top / 2 + 43;
}

static void pat_sorted(uint64_t *out, size_t count, uint64_t top) {
    uint64_t middle = top / 2 + 1;
    for (size_t i = 0; i < count; i++) out[i] = middle + i;
}

static void pat_reversed(uint64_t *out, size_t count, uint64_t top) {
    uint64_t middle = top / 2 + 1;
    for (size_t i = 0; i < count; i++) out[i] = middle + (count - i);
}

static void pat_nearly_sorted(uint64_t *out, size_t count, uint64_t top) {
    uint64_t middle = top / 2 + 1;
    for (size_t i = 0; i < count; i++) out[i] = middle + i + next_u64() % 10;
}

// Exactly 2 * count possible values, straddling zero for signed types:
// the widest range that takes the counting path
static void pat_range_twice_count(uint64_t *out, size_t count, uint64_t top) {
    uint64_t low = top / 2 + 1 - count;
    for (size_t i = 0; i < count; i++) out[i] = low + next_u64() % (2 * (uint64_t)count);
    if (count > 1) {
        out[0] = low + 2 * (uint64_t)count - 1;
        out[count - 1] = low;
    }
}

// One wider than that: the narrowest range that takes the bucket path
static void pat_range_twice_count_plus_one(uint64_t *out, size_t count, uint64_t top) {
    pat_range_twice_count(out, count, top);
    if (count > 1) out[0] += 1;
}

static const struct {
    const char *name;
    pattern fill;
} patterns[] = {
    {"small range at bottom", pat_small_range_at_bottom},
    {"small range around middle", pat_small_range_around_middle},
    {"full range", pat_full_range},
    {"upper half", pat_upper_half},
    {"extremes", pat_extremes},
    {"two values far apart", pat_two_values_far_apart},
    {"pair at top", pat_pair_at_top},
    {"pair at bottom", pat_pair_at_bottom},
    {"triple near top", pat_triple_near_top},
    {"single low outlier", pat_single_low_outlier},
    {"cluster with outlier", pat_cluster_with_outlier},
    {"powers of two", pat_powers_of_two},
    {"all equal", pat_all_equal},
    {"already sorted", pat_sorted},
    {"reversed", pat_reversed},
    {"nearly sorted", pat_nearly_sorted},
    {"range == 2 * count", pat_range_twice_count},
    {"range == 2 * count + 1", pat_range_twice_count_plus_one},
};

static const size_t counts[] = {0, 1, 2, 3, 15, 16, 17, 18, 19, 20, 33, 100,
                                1000, 1023, 1024, 1025, 10000, 200000};

// ---- Per-type checks -------------------------------------------------------

// Defines check_<suffix>: builds an array of TYPE from the offsets, sorts it,
// and compares it with the array built from the already sorted offsets.
// LOWEST is the smallest value of TYPE; UTYPE is its unsigned counterpart.
#define DEFINE_CHECK(SUFFIX, TYPE, UTYPE, LOWEST)                                  \
    static int check_##SUFFIX(const uint64_t *offsets, const uint64_t *sorted,     \
                              size_t count) {                                      \
        size_t bytes = count * sizeof(TYPE);                                       \
        TYPE *actual = malloc(bytes + 1), *expected = malloc(bytes + 1);           \
        if (!actual || !expected) {                                                \
            fprintf(stderr, "out of memory\n");                                    \
            exit(2);                                                               \
        }                                                                          \
        for (size_t i = 0; i < count; i++) {                                       \
            actual[i] = (TYPE)((UTYPE)(LOWEST) + (UTYPE)offsets[i]);               \
            expected[i] = (TYPE)((UTYPE)(LOWEST) + (UTYPE)sorted[i]);              \
        }                                                                          \
        ahb_sort_##SUFFIX(actual, count);                                          \
        int ok = memcmp(actual, expected, bytes) == 0;                             \
        free(actual);                                                              \
        free(expected);                                                            \
        return ok;                                                                 \
    }

DEFINE_CHECK(i32, int32_t, uint32_t, INT32_MIN)
DEFINE_CHECK(u32, uint32_t, uint32_t, 0)
DEFINE_CHECK(i64, int64_t, uint64_t, INT64_MIN)
DEFINE_CHECK(u64, uint64_t, uint64_t, 0)

static const struct {
    const char *name;
    uint64_t top;
    int (*check)(const uint64_t *offsets, const uint64_t *sorted, size_t count);
} types[] = {
    {"i32", UINT32_MAX, check_i32},
    {"u32", UINT32_MAX, check_u32},
    {"i64", UINT64_MAX, check_i64},
    {"u64", UINT64_MAX, check_u64},
};

// AHB_Sort(int *, int) is the original entry point; check it still works,
// including sizes that are zero or negative.
static int check_legacy_entry_point(void) {
    enum { COUNT = 5000 };
    static int actual[COUNT], expected[COUNT];
    for (int i = 0; i < COUNT; i++) {
        actual[i] = (int)(uint32_t)next_u64();
        expected[i] = actual[i];
    }
    uint64_t sorted[COUNT];
    for (int i = 0; i < COUNT; i++) sorted[i] = (uint64_t)((uint32_t)expected[i] - (uint32_t)INT_MIN);
    qsort(sorted, COUNT, sizeof(uint64_t), compare_u64);
    for (int i = 0; i < COUNT; i++) expected[i] = (int)((uint32_t)INT_MIN + (uint32_t)sorted[i]);

    AHB_Sort(actual, COUNT);
    int ok = memcmp(actual, expected, sizeof(actual)) == 0;

    int untouched[3] = {3, 1, 2};
    AHB_Sort(untouched, 0);
    AHB_Sort(untouched, -5);
    ok = ok && untouched[0] == 3 && untouched[1] == 1 && untouched[2] == 2;
    return ok;
}

int main(void) {
    int failures = 0, checks = 0;

    for (size_t t = 0; t < sizeof(types) / sizeof(types[0]); t++) {
        for (size_t p = 0; p < sizeof(patterns) / sizeof(patterns[0]); p++) {
            for (size_t c = 0; c < sizeof(counts) / sizeof(counts[0]); c++) {
                size_t count = counts[c];
                size_t bytes = count * sizeof(uint64_t);
                uint64_t *offsets = malloc(bytes + 1), *sorted = malloc(bytes + 1);
                if (!offsets || !sorted) {
                    fprintf(stderr, "out of memory\n");
                    return 2;
                }

                patterns[p].fill(offsets, count, types[t].top);
                memcpy(sorted, offsets, bytes);
                qsort(sorted, count, sizeof(uint64_t), compare_u64);

                checks++;
                if (!types[t].check(offsets, sorted, count)) {
                    failures++;
                    printf("FAIL: %s, %s, count %zu\n", types[t].name, patterns[p].name, count);
                }

                free(offsets);
                free(sorted);
            }
        }
    }

    checks++;
    if (!check_legacy_entry_point()) {
        failures++;
        printf("FAIL: AHB_Sort legacy entry point\n");
    }

    printf("%d/%d checks passed\n", checks - failures, checks);
    return failures ? 1 : 0;
}
