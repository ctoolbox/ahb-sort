// Correctness tests for ahb_sort.h: every case is checked against qsort,
// for all four element types.
// Build and run with `make test` (uses AddressSanitizer + UBSan).
//
// Four groups of checks: fixed patterns at fixed sizes, a subset of them on
// large arrays, randomly shaped inputs, and runs where chosen allocations
// inside the sort are made to fail.
//
// Test inputs are generated as 64-bit offsets from the smallest value of the
// element type. Offset order is the same as element order for every type, so
// one set of patterns covers signed and unsigned, 32-bit and 64-bit.

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---- Allocation hooks ------------------------------------------------------
//
// ahb_sort.h is compiled with malloc and calloc redirected to these hooks, so
// a test can make chosen allocations inside the sort fail. Allocations made by
// the sort are numbered from 0; those numbered fail_first..fail_last fail.

static long allocation_index = 0;
static long fail_first = -1, fail_last = -1;
static long injected_failures = 0;

static int allocation_should_fail(void) {
    long index = allocation_index++;
    if (fail_first < 0 || index < fail_first || index > fail_last) return 0;
    injected_failures++;
    return 1;
}

static void *hooked_malloc(size_t size) {
    return allocation_should_fail() ? NULL : malloc(size);
}

static void *hooked_calloc(size_t count, size_t size) {
    return allocation_should_fail() ? NULL : calloc(count, size);
}

#define malloc(size) hooked_malloc(size)
#define calloc(count, size) hooked_calloc(count, size)
#include "ahb_sort.h"
#undef malloc
#undef calloc

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

// Built against the sort's sampling step: the 64 elements it samples are
// spread over the whole range, while every other element sits in one tight
// cluster. The sort then picks its evenly-spread strategy and has to back out
// of it when it finds the oversized bucket.
static void pat_cluster_hidden_from_sample(uint64_t *out, size_t count, uint64_t top) {
    uint64_t middle = top / 2 + 1;
    size_t step = count / 64;
    for (size_t i = 0; i < count; i++) {
        if (step == 0 || i % step == 0) {
            out[i] = next_u64() & top;
        } else {
            out[i] = middle + next_u64() % 50;
        }
    }
}

static void pat_powers_of_two(uint64_t *out, size_t count, uint64_t top) {
    unsigned bits = top == UINT32_MAX ? 32 : 64;
    for (size_t i = 0; i < count; i++) out[i] = (uint64_t)1 << (next_u64() % bits);
}

// 1000 distinct values over the whole range: repeats show up in the sort's
// sample, but there are too many distinct values for its sparse path
static void pat_thousand_distinct(uint64_t *out, size_t count, uint64_t top) {
    static uint64_t values[1000];
    for (size_t i = 0; i < 1000; i++) values[i] = next_u64() & top;
    for (size_t i = 0; i < count; i++) out[i] = values[next_u64() % 1000];
}

// Wide but heavily skewed towards the bottom of the range
static void pat_skewed(uint64_t *out, size_t count, uint64_t top) {
    for (size_t i = 0; i < count; i++) out[i] = (next_u64() & top) >> (next_u64() % 24);
}

// 99% of the values in a block of 1000, the rest anywhere
static void pat_tight_cluster(uint64_t *out, size_t count, uint64_t top) {
    uint64_t middle = top / 2 + 1;
    for (size_t i = 0; i < count; i++) {
        out[i] = next_u64() % 100 == 0 ? (next_u64() & top) : middle + next_u64() % 1000;
    }
}

// Ascending and widely spaced, with 1% of the elements swapped at random
static void pat_sorted_with_swaps(uint64_t *out, size_t count, uint64_t top) {
    uint64_t gap = count ? top / count : 1;
    for (size_t i = 0; i < count; i++) out[i] = i * gap;
    for (size_t k = 0; k < count / 100; k++) {
        size_t a = next_u64() % count, b = next_u64() % count;
        uint64_t t = out[a];
        out[a] = out[b];
        out[b] = t;
    }
}

// Ascending runs of 4096 elements, each starting at a random value
static void pat_sorted_runs(uint64_t *out, size_t count, uint64_t top) {
    uint64_t base = 0;
    for (size_t i = 0; i < count; i++) {
        if (i % 4096 == 0) base = next_u64() & (top / 2);
        out[i] = base + (i % 4096) * 16;
    }
}

// Rising over the first half and falling over the second
static void pat_organ_pipe(uint64_t *out, size_t count, uint64_t top) {
    const uint64_t scale = top / (count + 1);
    for (size_t i = 0; i < count; i++) out[i] = (i < count / 2 ? i : count - i) * scale;
}

// Ascending, with every tenth element replaced by a random value
static void pat_sorted_tenth_random(uint64_t *out, size_t count, uint64_t top) {
    const uint64_t scale = top / (count + 1);
    for (size_t i = 0; i < count; i++) out[i] = i % 10 == 3 ? next_u64() & top : i * scale;
}

// Ascending and bunched towards the bottom of the range, with 1% of the
// elements swapped at random
static void pat_sorted_skewed_with_swaps(uint64_t *out, size_t count, uint64_t top) {
    for (size_t i = 0; i < count; i++) {
        double x = (double)(i + 1) / (double)(count + 1);
        out[i] = (uint64_t)(x * x * x * x * (double)(top / 2));
    }
    for (size_t k = 0; k < count / 100; k++) {
        size_t a = next_u64() % count, b = next_u64() % count;
        uint64_t t = out[a];
        out[a] = out[b];
        out[b] = t;
    }
}

// 200 distinct values, then 2% of unrelated values at the very end: the
// sort's sparse path gets almost all the way through before giving up
static void pat_late_surprise(uint64_t *out, size_t count, uint64_t top) {
    uint64_t values[200];
    for (size_t i = 0; i < 200; i++) values[i] = next_u64() & top;
    for (size_t i = 0; i < count; i++) {
        out[i] = i < count - count / 50 ? values[next_u64() % 200] : (next_u64() & top);
    }
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
    {"cluster hidden from sample", pat_cluster_hidden_from_sample},
    {"all equal", pat_all_equal},
    {"1000 distinct", pat_thousand_distinct},
    {"skewed", pat_skewed},
    {"tight cluster", pat_tight_cluster},
    {"sorted with swaps", pat_sorted_with_swaps},
    {"sorted runs", pat_sorted_runs},
    {"organ pipe", pat_organ_pipe},
    {"sorted, every tenth random", pat_sorted_tenth_random},
    {"sorted, skewed, with swaps", pat_sorted_skewed_with_swaps},
    {"late surprise", pat_late_surprise},
    {"already sorted", pat_sorted},
    {"reversed", pat_reversed},
    {"nearly sorted", pat_nearly_sorted},
    {"range == 2 * count", pat_range_twice_count},
    {"range == 2 * count + 1", pat_range_twice_count_plus_one},
};

static const size_t counts[] = {0, 1, 2, 3, 15, 16, 17, 18, 19, 20, 33, 100,
                                1000, 1023, 1024, 1025, 10000, 200000};

// Arrays of more than 2^21 elements switch to fewer, larger buckets, so these
// patterns also run at sizes below and above that limit.
static const struct {
    const char *name;
    pattern fill;
} large_patterns[] = {
    {"full range", pat_full_range},
    {"upper half", pat_upper_half},
    {"1000 distinct", pat_thousand_distinct},
    {"skewed", pat_skewed},
    {"tight cluster", pat_tight_cluster},
    {"sorted with swaps", pat_sorted_with_swaps},
    {"sorted runs", pat_sorted_runs},
    {"organ pipe", pat_organ_pipe},
    {"sorted, every tenth random", pat_sorted_tenth_random},
    {"sorted, skewed, with swaps", pat_sorted_skewed_with_swaps},
    {"late surprise", pat_late_surprise},
    {"nearly sorted", pat_nearly_sorted},
    {"cluster hidden from sample", pat_cluster_hidden_from_sample},
    {"range == 2 * count", pat_range_twice_count},
    {"extremes", pat_extremes},
};

static const size_t large_counts[] = {300000, 2500000};

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

// ---- Randomly shaped inputs ------------------------------------------------

// Random count, random width of the value range (from 1 bit to the full
// type), a random position for that range, and sometimes only a few distinct
// values.
static size_t fill_random_shape(uint64_t *out, size_t capacity, uint64_t top) {
    unsigned bits = top == UINT32_MAX ? 32 : 64;
    size_t count = next_u64() % (capacity + 1);
    unsigned width_bits = 1 + (unsigned)(next_u64() % bits);
    uint64_t width_mask = width_bits == 64 ? UINT64_MAX : (((uint64_t)1 << width_bits) - 1);
    uint64_t base = (next_u64() & top) & ~width_mask;

    uint64_t palette[8];
    for (int k = 0; k < 8; k++) palette[k] = base | (next_u64() & width_mask);
    int few_distinct = next_u64() % 4 == 0;

    for (size_t i = 0; i < count; i++) {
        out[i] = few_distinct ? palette[next_u64() % 8] : (base | (next_u64() & width_mask));
    }
    return count;
}

// ---- Allocation failures ---------------------------------------------------

// Patterns that between them reach the counting path, the bucket path and
// recursion into a large bucket
static const pattern allocation_patterns[] = {
    pat_small_range_at_bottom,
    pat_full_range,
    pat_cluster_with_outlier,
    pat_powers_of_two,
    pat_thousand_distinct,
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

    // Large arrays
    for (size_t t = 0; t < sizeof(types) / sizeof(types[0]); t++) {
        for (size_t p = 0; p < sizeof(large_patterns) / sizeof(large_patterns[0]); p++) {
            for (size_t c = 0; c < sizeof(large_counts) / sizeof(large_counts[0]); c++) {
                size_t count = large_counts[c];
                size_t bytes = count * sizeof(uint64_t);
                uint64_t *offsets = malloc(bytes), *sorted = malloc(bytes);
                if (!offsets || !sorted) {
                    fprintf(stderr, "out of memory\n");
                    return 2;
                }

                large_patterns[p].fill(offsets, count, types[t].top);
                memcpy(sorted, offsets, bytes);
                qsort(sorted, count, sizeof(uint64_t), compare_u64);

                checks++;
                if (!types[t].check(offsets, sorted, count)) {
                    failures++;
                    printf("FAIL: %s, %s, count %zu\n", types[t].name, large_patterns[p].name, count);
                }

                free(offsets);
                free(sorted);
            }
        }
    }

    // Randomly shaped inputs
    enum { RANDOM_CASES = 500, RANDOM_CAPACITY = 3000 };
    for (size_t t = 0; t < sizeof(types) / sizeof(types[0]); t++) {
        for (int round = 0; round < RANDOM_CASES; round++) {
            uint64_t offsets[RANDOM_CAPACITY], sorted[RANDOM_CAPACITY];
            uint64_t seed = rng_state;
            size_t count = fill_random_shape(offsets, RANDOM_CAPACITY, types[t].top);
            memcpy(sorted, offsets, count * sizeof(uint64_t));
            qsort(sorted, count, sizeof(uint64_t), compare_u64);

            checks++;
            if (!types[t].check(offsets, sorted, count)) {
                failures++;
                printf("FAIL: %s, random shape, count %zu, generator state %llu\n",
                       types[t].name, count, (unsigned long long)seed);
            }
        }
    }

    // Allocation failures: the array must still end up sorted
    enum { ALLOCATION_COUNT = 5000 };
    for (size_t t = 0; t < sizeof(types) / sizeof(types[0]); t++) {
        for (size_t p = 0; p < sizeof(allocation_patterns) / sizeof(allocation_patterns[0]); p++) {
            static uint64_t offsets[ALLOCATION_COUNT], sorted[ALLOCATION_COUNT];
            allocation_patterns[p](offsets, ALLOCATION_COUNT, types[t].top);
            memcpy(sorted, offsets, sizeof(offsets));
            qsort(sorted, ALLOCATION_COUNT, sizeof(uint64_t), compare_u64);

            for (long first = 0; first < 6; first++) {
                for (int every_later_one = 0; every_later_one < 2; every_later_one++) {
                    allocation_index = 0;
                    fail_first = first;
                    fail_last = every_later_one ? LONG_MAX : first;
                    int ok = types[t].check(offsets, sorted, ALLOCATION_COUNT);
                    fail_first = fail_last = -1;

                    checks++;
                    if (!ok) {
                        failures++;
                        printf("FAIL: %s, allocation %ld%s failing, pattern %zu\n", types[t].name,
                               first, every_later_one ? " onwards" : "", p);
                    }
                }
            }
        }
    }

    checks++;
    if (injected_failures == 0) {
        failures++;
        printf("FAIL: no allocation failure was ever injected\n");
    }

    checks++;
    if (!check_legacy_entry_point()) {
        failures++;
        printf("FAIL: AHB_Sort legacy entry point\n");
    }

    printf("%d/%d checks passed\n", checks - failures, checks);
    return failures ? 1 : 0;
}
