// Times AHB Sort against stronger baselines than benchmark.c uses:
//
//   std::sort      the C++ standard library's comparison sort
//   pdqsort        pattern-defeating quicksort (Orson Peters)
//   ska_sort       in-place radix sort (Malte Skarupke)
//   ska_sort_copy  the same author's out-of-place radix sort
//   radix-256      the byte-wise LSD radix sort from benchmark.c
//   radix-2048     an LSD radix sort with 11-bit digits: three passes
//
// pdqsort.h and ska_sort.hpp are not part of this repository. `make baselines`
// downloads them into experiments/third_party/.
//
// Usage: baselines [size ...]      default sizes: 100000 1000000 5000000

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#include "third_party/pdqsort.h"
#include "third_party/ska_sort.hpp"

extern "C" {
#include "../ahb_sort.h"
}

// ---- Sorts -----------------------------------------------------------------

static std::vector<int> scratch;  // for the sorts that take a caller's buffer

static void run_ahb(int *a, int n) { AHB_Sort(a, n); }
static void run_std(int *a, int n) { std::sort(a, a + n); }
static void run_pdq(int *a, int n) { pdqsort(a, a + n); }
static void run_ska(int *a, int n) { ska_sort(a, a + n); }

static void run_ska_copy(int *a, int n) {
    if (ska_sort_copy(a, a + n, scratch.data())) memcpy(a, scratch.data(), (size_t)n * sizeof(int));
}

// One byte per pass, as in benchmark.c. Passes where every element has the
// same byte are skipped.
static void run_radix_256(int *arr, int n) {
    uint32_t *data = (uint32_t *)arr, *source = data, *target = (uint32_t *)scratch.data();
    static size_t count[4][256];
    memset(count, 0, sizeof(count));
    for (int i = 0; i < n; i++) {
        uint32_t v = data[i] ^ 0x80000000u;
        data[i] = v;
        count[0][v & 255]++;
        count[1][(v >> 8) & 255]++;
        count[2][(v >> 16) & 255]++;
        count[3][v >> 24]++;
    }
    for (int pass = 0; pass < 4; pass++) {
        int shift = pass * 8;
        if (count[pass][(source[0] >> shift) & 255] == (size_t)n) continue;
        size_t offsets[256], total = 0;
        for (int b = 0; b < 256; b++) {
            offsets[b] = total;
            total += count[pass][b];
        }
        for (int i = 0; i < n; i++) target[offsets[(source[i] >> shift) & 255]++] = source[i];
        std::swap(source, target);
    }
    if (source != data) memcpy(data, source, (size_t)n * sizeof(uint32_t));
    for (int i = 0; i < n; i++) data[i] ^= 0x80000000u;
}

// 11 + 11 + 10 bits: three passes instead of four, with larger tables
static void run_radix_2048(int *arr, int n) {
    uint32_t *data = (uint32_t *)arr, *source = data, *target = (uint32_t *)scratch.data();
    static const int shifts[3] = {0, 11, 22};
    static const uint32_t masks[3] = {2047, 2047, 1023};
    static size_t count[3][2048];
    memset(count, 0, sizeof(count));
    for (int i = 0; i < n; i++) {
        uint32_t v = data[i] ^ 0x80000000u;
        data[i] = v;
        count[0][v & 2047]++;
        count[1][(v >> 11) & 2047]++;
        count[2][v >> 22]++;
    }
    for (int pass = 0; pass < 3; pass++) {
        int shift = shifts[pass];
        uint32_t mask = masks[pass];
        if (count[pass][(source[0] >> shift) & mask] == (size_t)n) continue;
        size_t total = 0;
        for (uint32_t b = 0; b <= mask; b++) {
            size_t c = count[pass][b];
            count[pass][b] = total;
            total += c;
        }
        for (int i = 0; i < n; i++) target[count[pass][(source[i] >> shift) & mask]++] = source[i];
        std::swap(source, target);
    }
    if (source != data) memcpy(data, source, (size_t)n * sizeof(uint32_t));
    for (int i = 0; i < n; i++) data[i] ^= 0x80000000u;
}

// The first entry is the one the others are compared against.
static const struct {
    const char *name;
    void (*sort)(int *, int);
} sorts[] = {
    {"AHB", run_ahb},         {"std::sort", run_std},   {"pdqsort", run_pdq},      {"ska_sort", run_ska},
    {"ska_copy", run_ska_copy}, {"radix256", run_radix_256}, {"radix2048", run_radix_2048},
};
static const int SORT_COUNT = (int)(sizeof(sorts) / sizeof(sorts[0]));

// ---- Inputs ----------------------------------------------------------------

static uint64_t rng_state;

static uint32_t next_u32() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (uint32_t)(rng_state >> 16);
}

static double next_unit() { return (next_u32() + 1.0) / 4294967297.0; }

static const double PI = 3.14159265358979323846;

// The six shapes of the main benchmark. The sort's thresholds were tuned on these.
static void in_range_1000(int *a, int n) { for (int i = 0; i < n; i++) a[i] = (int)(next_u32() % 1000); }
static void in_range_1e9(int *a, int n) { for (int i = 0; i < n; i++) a[i] = (int)(next_u32() % 1000000000); }
static void in_nearly_sorted(int *a, int n) { for (int i = 0; i < n; i++) a[i] = i + (int)(next_u32() % 10); }
static void in_100_distinct(int *a, int n) {
    int values[100];
    for (int i = 0; i < 100; i++) values[i] = (int)next_u32();
    for (int i = 0; i < n; i++) a[i] = values[next_u32() % 100];
}
static void in_exponential(int *a, int n) { for (int i = 0; i < n; i++) a[i] = (int)(-log(next_unit()) * 1e6); }
static void in_cluster(int *a, int n) {
    for (int i = 0; i < n; i++) a[i] = next_u32() % 100 == 0 ? (int)(next_u32() % 1000000000) : (int)(next_u32() % 1000);
}

// Twelve shapes that played no part in tuning the sort
static void in_v_shape(int *a, int n) { for (int i = 0; i < n; i++) a[i] = (i < n / 2 ? n / 2 - i : i - n / 2) * 300; }
static void in_four_waves(int *a, int n) { for (int i = 0; i < n; i++) a[i] = (int)(1e9 + 9e8 * sin(i * (8.0 * PI / n))); }
static void in_forty_waves(int *a, int n) { for (int i = 0; i < n; i++) a[i] = (int)(1e9 + 9e8 * sin(i * (80.0 * PI / n))); }
static void in_sorted_quarter_random(int *a, int n) {
    for (int i = 0; i < n; i++) a[i] = next_u32() % 4 == 0 ? (int)(next_u32() % ((uint32_t)n * 100)) : i * 100;
}
static void in_descending_5pct(int *a, int n) {
    for (int i = 0; i < n; i++) a[i] = next_u32() % 20 == 0 ? (int)(next_u32() % ((uint32_t)n * 100)) : (n - i) * 100;
}
static void in_locally_shuffled(int *a, int n) { for (int i = 0; i < n; i++) a[i] = i * 100 + (int)(next_u32() % 6400); }
static void in_two_rising(int *a, int n) { for (int i = 0; i < n; i++) a[i] = (i & 1) ? 1000000000 + i * 50 : i * 50; }
static void in_uniform_31bit(int *a, int n) { for (int i = 0; i < n; i++) a[i] = (int)(next_u32() >> 1); }
static void in_triangular(int *a, int n) { for (int i = 0; i < n; i++) a[i] = (int)((next_unit() + next_unit()) * 1e9); }
static void in_three_runs(int *a, int n) {
    int third = n / 3 + 1;
    for (int i = 0; i < n; i++) a[i] = (i % third) * 300 + i / third;
}
static void in_sorted_prefix(int *a, int n) {
    for (int i = 0; i < n; i++) a[i] = i < n / 10 ? i * 1000 : (int)(next_u32() % 2000000000);
}
static void in_third_one_value(int *a, int n) {
    for (int i = 0; i < n; i++) a[i] = next_u32() % 3 == 0 ? 123456789 : (int)(next_u32() >> 1);
}

static const struct {
    const char *name;
    void (*generate)(int *, int);
} inputs[] = {
    {"uniform, range 1000", in_range_1000},
    {"uniform, range 1e9", in_range_1e9},
    {"nearly sorted", in_nearly_sorted},
    {"100 distinct, wide", in_100_distinct},
    {"exponential", in_exponential},
    {"cluster + 1% outliers", in_cluster},
    {"V shape", in_v_shape},
    {"four sine waves", in_four_waves},
    {"forty sine waves", in_forty_waves},
    {"sorted, 25% random", in_sorted_quarter_random},
    {"descending, 5% random", in_descending_5pct},
    {"locally shuffled", in_locally_shuffled},
    {"two rising, interleaved", in_two_rising},
    {"uniform, 31 bits", in_uniform_31bit},
    {"triangular", in_triangular},
    {"three sorted runs", in_three_runs},
    {"10% sorted prefix", in_sorted_prefix},
    {"one third one value", in_third_one_value},
};
static const int INPUT_COUNT = (int)(sizeof(inputs) / sizeof(inputs[0]));
static const int TUNED_INPUTS = 6;  // the first six

// ---- Measurement -----------------------------------------------------------

static double now_ms() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

static int failures = 0;

// Best time of several rounds for each sort. The sorts take turns within a
// round, so a busy spell on the machine affects all of them.
static void measure(int input, int n, int rounds, double best[]) {
    std::vector<int> source(n), expected, work(n);
    rng_state = 88172645463325252ULL + (uint64_t)input;
    inputs[input].generate(source.data(), n);
    expected = source;
    std::sort(expected.begin(), expected.end());

    for (int s = 0; s < SORT_COUNT; s++) best[s] = 1e300;
    for (int round = 0; round < rounds; round++) {
        for (int s = 0; s < SORT_COUNT; s++) {
            work = source;
            double start = now_ms();
            sorts[s].sort(work.data(), n);
            double elapsed = now_ms() - start;
            if (elapsed < best[s]) best[s] = elapsed;
            if (round == 0 && work != expected) {
                failures++;
                printf("WRONG OUTPUT: %s on %s, size %d\n", sorts[s].name, inputs[input].name, n);
            }
        }
    }
}

static void print_mean(const char *label, const double *log_sum, int count) {
    printf("%-24s %8s", label, "");
    for (int s = 1; s < SORT_COUNT; s++) printf(" %9.2fx", exp(log_sum[s] / count));
    printf("\n");
}

static void run_size(int n) {
    int rounds = n >= 1000000 ? 9 : 31;
    scratch.assign(n, 0);

    printf("\nn = %d   (best of %d rounds; AHB in ms, then each sort's time / AHB's: above 1 means AHB is faster)\n", n, rounds);
    printf("%-24s %8s", "input", sorts[0].name);
    for (int s = 1; s < SORT_COUNT; s++) printf(" %10s", sorts[s].name);
    printf("\n");

    std::vector<double> tuned(SORT_COUNT, 0.0), unseen(SORT_COUNT, 0.0);
    for (int in = 0; in < INPUT_COUNT; in++) {
        if (in == TUNED_INPUTS) print_mean("  mean, tuning inputs", tuned.data(), TUNED_INPUTS);
        std::vector<double> best(SORT_COUNT);
        measure(in, n, rounds, best.data());
        printf("%-24s %8.3f", inputs[in].name, best[0]);
        for (int s = 1; s < SORT_COUNT; s++) {
            printf(" %9.2fx", best[s] / best[0]);
            (in < TUNED_INPUTS ? tuned : unseen)[s] += log(best[s] / best[0]);
        }
        printf("\n");
        fflush(stdout);
    }
    print_mean("  mean, other inputs", unseen.data(), INPUT_COUNT - TUNED_INPUTS);
}

int main(int argc, char **argv) {
    static const int default_sizes[] = {100000, 1000000, 5000000};
    if (argc > 1) {
        for (int a = 1; a < argc; a++) {
            int n = atoi(argv[a]);
            if (n < 1) {
                fprintf(stderr, "usage: %s [size ...]\n", argv[0]);
                return 2;
            }
            run_size(n);
        }
    } else {
        for (int n : default_sizes) run_size(n);
    }
    if (failures) printf("\n%d outputs were WRONG\n", failures);
    return failures ? 1 : 0;
}
