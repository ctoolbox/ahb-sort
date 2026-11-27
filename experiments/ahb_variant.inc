/*
 * Template for experimental AHB_Sort variants. Include it once per variant
 * with these macros defined (all are #undef'd at the end). The including file
 * must provide ahbv_insertion_sort, ahbv_compare_ints and ahbv_clamp_index.
 *
 *   AHB_FN              name of the generated function (required)
 *   AHB_OPT_COUNTING    0 = off; C > 0 = plain counting sort when
 *                       range <= size * C / 4
 *   AHB_OPT_POW2        0 = bucket index by multiply;
 *                       1 = power-of-two bucket width, rounded up (shift);
 *                       2 = power-of-two bucket width, rounded down (shift)
 *   AHB_OPT_DIVISOR     0 = variance-based bucket count;
 *                       K > 0 = bucket count is size / K, no variance pass
 *   AHB_OPT_MAX_BUCKETS 0 = no limit; M > 0 = at most M buckets per level
 *   AHB_OPT_SHARED      0 = every recursion level allocates its own scratch;
 *                       1 = one scratch buffer for the whole sort (a level
 *                       sorting a bucket uses the matching part of the
 *                       caller's array as its scratch)
 *   AHB_OPT_BUCKET_MULT with AHB_OPT_DIVISOR = K: bucket count is
 *                       size * MULT / K (default 1)
 *   AHB_OPT_THRESHOLD   buckets larger than this are sorted recursively,
 *                       smaller ones by insertion sort (default 64; arrays
 *                       under 1024 elements always use size / 16)
 *   AHB_OPT_FINAL_PASS  0 = insertion-sort each small bucket separately;
 *                       1 = sort only the large buckets, then finish with a
 *                       single insertion-sort pass over the whole array;
 *                       M > 1 = do that only when the largest bucket holds
 *                       at most M elements
 *   AHB_OPT_SAMPLE      0 = off; K > 0 = look at 64 evenly spaced elements
 *                       first. If they are all different and spread evenly
 *                       over the range, use size * K buckets and allow the
 *                       final pass (AHB_OPT_FINAL_PASS = M, required);
 *                       otherwise use the normal bucket count and no final
 *                       pass
 *   AHB_OPT_COUNT_SPREAD with AHB_OPT_SAMPLE: 1 = when the sample says the
 *                       data is evenly spread, use counting sort only up to
 *                       range <= size (the bucket route with a final pass is
 *                       faster above that)
 *   AHB_OPT_SPARSE      S > 0 (a power of two) = when the sample holds a
 *                       repeated value, try to count the distinct values in
 *                       a hash table of S slots (gives up past 256 of them);
 *                       needs AHB_OPT_SAMPLE
 *   AHB_OPT_CAP_SKEW    M > 0 = at most M buckets per level for data the
 *                       sample does not call evenly spread, at levels of at
 *                       least AHB_OPT_CAP_SKEW_AT elements
 *   AHB_OPT_CAP_SPREAD  M > 0 = at most M buckets per level for evenly spread
 *                       data of more than AHB_OPT_CAP_SPREAD_AT elements
 *   AHB_OPT_CAP_SPREAD_DIV  D > 0 = size / D buckets for evenly spread data
 *                       of more than AHB_OPT_CAP_SPREAD_AT elements
 *   AHB_OPT_ORDERED_MIN    how many of the sample comparisons must agree for
 *                          the array to count as ordered (default 56).
 *   AHB_OPT_ORDERED_TURNS  also count the array as ordered when the samples
 *                          change direction at most this many times (0 = off).
 *   AHB_OPT_CAP_GUARDS  1 = no cap when the sample is already in order, and
 *                       no AHB_OPT_CAP_SKEW cap when the middle half of the
 *                       sample would fit in one capped bucket;
 *                       2 = no cap when the sample is in order or the array
 *                       is made of sorted runs, and no AHB_OPT_CAP_SKEW cap
 *                       when the sample falls into fewer than 32 capped
 *                       buckets or the sparse attempt got more than half way
 *   AHB_OPT_WC          B > 0 = at levels with at most 4096 buckets and at
 *                       least 262144 elements, scatter
 *                       through per-bucket buffers of B elements that are
 *                       flushed with one copy (software write combining)
 *   AHB_OPT_COMPACT     0 = three bookkeeping arrays per level (offsets,
 *                       sizes, write positions); 1 = a single array
 *
 * With every option at its default the generated code is the original
 * algorithm.
 */

#ifndef AHB_FN
#error "define AHB_FN before including ahb_variant.inc"
#endif
#ifndef AHB_OPT_COUNTING
#define AHB_OPT_COUNTING 0
#endif
#ifndef AHB_OPT_POW2
#define AHB_OPT_POW2 0
#endif
#ifndef AHB_OPT_DIVISOR
#define AHB_OPT_DIVISOR 0
#endif
#ifndef AHB_OPT_MAX_BUCKETS
#define AHB_OPT_MAX_BUCKETS 0
#endif
#ifndef AHB_OPT_SHARED
#define AHB_OPT_SHARED 0
#endif
#ifndef AHB_OPT_BUCKET_MULT
#define AHB_OPT_BUCKET_MULT 1
#endif
#ifndef AHB_OPT_THRESHOLD
#define AHB_OPT_THRESHOLD 64
#endif
#ifndef AHB_OPT_FINAL_PASS
#define AHB_OPT_FINAL_PASS 0
#endif
#ifndef AHB_OPT_COMPACT
#define AHB_OPT_COMPACT 0
#endif
#ifndef AHB_OPT_SAMPLE
#define AHB_OPT_SAMPLE 0
#endif
#ifndef AHB_OPT_COUNT_SPREAD
#define AHB_OPT_COUNT_SPREAD 0
#endif
#ifndef AHB_OPT_SPARSE
#define AHB_OPT_SPARSE 0
#endif
#ifndef AHB_OPT_CAP_SKEW
#define AHB_OPT_CAP_SKEW 0
#endif
#ifndef AHB_OPT_CAP_SPREAD
#define AHB_OPT_CAP_SPREAD 0
#endif
#ifndef AHB_OPT_CAP_SPREAD_AT
#define AHB_OPT_CAP_SPREAD_AT 0
#endif
#ifndef AHB_OPT_CAP_SKEW_AT
#define AHB_OPT_CAP_SKEW_AT 0
#endif
#ifndef AHB_OPT_CAP_SPREAD_DIV
#define AHB_OPT_CAP_SPREAD_DIV 0
#endif
#ifndef AHB_OPT_ORDERED_MIN
#define AHB_OPT_ORDERED_MIN 56
#endif
#ifndef AHB_OPT_ORDERED_TURNS
#define AHB_OPT_ORDERED_TURNS 0
#endif
#ifndef AHB_OPT_CAP_GUARDS
#define AHB_OPT_CAP_GUARDS 0
#endif
#ifndef AHB_OPT_WC
#define AHB_OPT_WC 0
#endif

#if AHB_OPT_SPARSE && AHB_OPT_SPARSE < 512
#error "AHB_OPT_SPARSE is the hash table size: a power of two, at least 512"
#endif

#if AHB_OPT_COMPACT
#define AHBV_ARRAYS 1
#else
#define AHBV_ARRAYS 3
#endif

#define AHBV_CAT2(a, b) a##b
#define AHBV_CAT(a, b) AHBV_CAT2(a, b)
#define AHBV_REC AHBV_CAT(AHB_FN, _rec)

#if AHB_OPT_POW2
#define AHBV_INDEX(x) ((int)(((uint32_t)(x) - min_u) >> shift))
#else
#define AHBV_INDEX(x) ahbv_clamp_index((int)(((uint32_t)(x) - min_u) * bucket_size_inv), bucket_count)
#endif

// Sorts arr in place. scratch is NULL or a buffer of at least size ints.
static void AHBV_REC(int *arr, int *scratch, int size) {
    (void)scratch;
    if (size <= 1) return;
    if (size <= 16) {
        ahbv_insertion_sort(arr, size);
        return;
    }

    int is_sorted = 1;
    for (int i = 1; i < size; i++) {
        if (arr[i] < arr[i-1]) {
            is_sorted = 0;
            break;
        }
    }
    if (is_sorted) return;

    int max_value = arr[0], min_value = arr[0];
    int i = 0;

#if AHB_OPT_DIVISOR == 0
    const double base = (double)arr[0];
    double sum = 0.0, sum_sq = 0.0;
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

        double d1 = v1 - base, d2 = v2 - base, d3 = v3 - base, d4 = v4 - base;
        sum += d1 + d2 + d3 + d4;
        sum_sq += d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4;
    }
    for (; i < size; i++) {
        int val = arr[i];
        if (val > max_value) max_value = val;
        if (val < min_value) min_value = val;
        double d = val - base;
        sum += d;
        sum_sq += d * d;
    }

    double mean = sum / size;
    double variance = (sum_sq / size) - (mean * mean);
    if (!(variance > 0.0)) variance = 0.0;

    int64_t range = (int64_t)max_value - (int64_t)min_value + 1;
    double ideal_count = sqrt(variance) / (double)range * size + 1.0;
    int bucket_count = ideal_count >= (double)size ? size : (int)ideal_count;
#else
    for (i = 1; i < size; i++) {
        int val = arr[i];
        max_value = val > max_value ? val : max_value;
        min_value = val < min_value ? val : min_value;
    }
    int64_t range = (int64_t)max_value - (int64_t)min_value + 1;
    int bucket_count = (int)((int64_t)size * AHB_OPT_BUCKET_MULT / AHB_OPT_DIVISOR);
#endif
    bucket_count = bucket_count < 2 ? 2 : bucket_count;
#if AHB_OPT_MAX_BUCKETS
    if (bucket_count > AHB_OPT_MAX_BUCKETS) bucket_count = AHB_OPT_MAX_BUCKETS;
#endif

    const uint32_t min_u = (uint32_t)min_value;

#if AHB_OPT_SAMPLE
    // Decide from a small sample whether the values are spread out evenly,
    // and whether any value repeats.
    // ordered: the array is close to sorted or made of sorted runs. Either
    // at least 56 of the 63 neighbouring samples are in ascending order (or
    // 56 in descending order), or, with AHB_OPT_CAP_GUARDS = 2, at least 56
    // of the 64 samples are in order with the element right after them.
    // bulk: the width of the middle half of the sample.
    int sample[64];
    int spread = 0, repeats = 0, ordered = 0;
    uint32_t bulk = 0;
    if (size >= 1024) {
        int bins[8] = {0};
        int step = size / 64;
        int rising = 0, falling = 0, steps_up = 0, steps_down = 0;
        for (int k = 0; k < 64; k++) {
            int value = arr[k * step], next = arr[k * step + 1];
            sample[k] = value;
            bins[(int)(((uint64_t)((uint32_t)value - min_u) * 8) / (uint64_t)range)]++;
            if (next >= value) steps_up++;
            if (next <= value) steps_down++;
        }
        for (int k = 1; k < 64; k++) {
            if (sample[k] >= sample[k - 1]) rising++;
            if (sample[k] <= sample[k - 1]) falling++;
        }
        (void)steps_up; (void)steps_down;
        ordered = rising >= AHB_OPT_ORDERED_MIN || falling >= AHB_OPT_ORDERED_MIN;
#if AHB_OPT_CAP_GUARDS == 2
        ordered = ordered || steps_up >= AHB_OPT_ORDERED_MIN || steps_down >= AHB_OPT_ORDERED_MIN;
#endif
#if AHB_OPT_ORDERED_TURNS
        {
            // Long rising and falling stretches (organ pipe, a few big runs):
            // the samples change direction only a few times.
            int turns = 0, direction = 0;
            for (int k = 1; k < 64; k++) {
                int d = (sample[k] > sample[k - 1]) - (sample[k] < sample[k - 1]);
                if (d && direction && d != direction) turns++;
                if (d) direction = d;
            }
            ordered = ordered || turns <= AHB_OPT_ORDERED_TURNS;
        }
#endif
        ahbv_insertion_sort(sample, 64);
        for (int k = 1; k < 64; k++) {
            if (sample[k] == sample[k - 1]) repeats = 1;
        }
        spread = !repeats;
        for (int k = 0; k < 8; k++) {
            if (bins[k] < 3) spread = 0;
        }
        bulk = (uint32_t)sample[47] - (uint32_t)sample[16];
    }
    (void)sample;
    (void)repeats; (void)ordered; (void)bulk;
#endif

#if AHB_OPT_COUNTING
    // Narrow range: count each value and write the values straight back.
#if AHB_OPT_COUNT_SPREAD
    if (range <= (int64_t)size * (spread ? 4 : AHB_OPT_COUNTING) / 4) {
#else
    if (range <= (int64_t)size * AHB_OPT_COUNTING / 4) {
#endif
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
    }
#endif

#if AHB_OPT_SPARSE
    // Few distinct values: count them in a small hash table, sort the
    // distinct values, and write each back as many times as it occurred.
    int sparse_scanned = 0;
    (void)sparse_scanned;
    if (repeats) {
        enum { SLOTS = AHB_OPT_SPARSE, MAX_DISTINCT = 256 };
        int keys[SLOTS];
        uint32_t counts[SLOTS];
        int distinct[MAX_DISTINCT];
        int distinct_count = 0;
        memset(counts, 0, sizeof counts);
        for (i = 0; i < size; i++) {
            int value = arr[i];
            uint32_t slot = (uint32_t)(((uint64_t)(uint32_t)value * 0x9E3779B97F4A7C15ULL) >> 40) & (SLOTS - 1);
            while (counts[slot] && keys[slot] != value) slot = (slot + 1) & (SLOTS - 1);
            if (!counts[slot]) {
                if (distinct_count == MAX_DISTINCT) break;
                keys[slot] = value;
                distinct[distinct_count++] = value;
            }
            counts[slot]++;
        }
        if (i == size) {
            ahbv_insertion_sort(distinct, distinct_count);
            int *out = arr;
            for (int d = 0; d < distinct_count; d++) {
                int value = distinct[d];
                uint32_t slot = (uint32_t)(((uint64_t)(uint32_t)value * 0x9E3779B97F4A7C15ULL) >> 40) & (SLOTS - 1);
                while (keys[slot] != value) slot = (slot + 1) & (SLOTS - 1);
                for (uint32_t c = counts[slot]; c > 0; c--) *out++ = value;
            }
            return;
        }
        sparse_scanned = i;
    }
#endif

#if AHB_OPT_SAMPLE
    if (spread) {
        int64_t wanted = (int64_t)size * AHB_OPT_SAMPLE;
        bucket_count = wanted > INT32_MAX / 2 ? INT32_MAX / 2 : (int)wanted;
    }
#if AHB_OPT_CAP_GUARDS == 2
    // A cap only pays if the data really spreads over the buckets: not when
    // the array is nearly sorted, when the sample falls into fewer than 32
    // of the capped buckets, or when the sparse attempt got through more
    // than half the array before giving up.
    int occupied = 1;
    {
        int64_t wide = (range - 1) / (AHB_OPT_CAP_SKEW ? AHB_OPT_CAP_SKEW : 1) + 1;
        for (int k = 1; k < 64; k++) {
            if (((uint32_t)sample[k] - min_u) / wide != ((uint32_t)sample[k - 1] - min_u) / wide) occupied++;
        }
    }
    const int skew_cap_ok = !ordered && occupied >= 32 && sparse_scanned <= size / 2;
    const int spread_cap_ok = !ordered;
#elif AHB_OPT_CAP_GUARDS
    // A cap only pays if it splits the bulk of the data over many buckets,
    // and nearly sorted data needs no cap at all.
    const int skew_cap_ok = !ordered && (int64_t)bulk >= (range - 1) / (AHB_OPT_CAP_SKEW ? AHB_OPT_CAP_SKEW : 1) + 1;
    const int spread_cap_ok = !ordered;
#else
    const int skew_cap_ok = 1, spread_cap_ok = 1;
#endif
    (void)skew_cap_ok; (void)spread_cap_ok;
#if AHB_OPT_CAP_SKEW
    if (!spread && skew_cap_ok && size >= AHB_OPT_CAP_SKEW_AT && bucket_count > AHB_OPT_CAP_SKEW) {
        bucket_count = AHB_OPT_CAP_SKEW;
    }
#endif
#if AHB_OPT_CAP_SPREAD
    if (spread && spread_cap_ok && size > AHB_OPT_CAP_SPREAD_AT && bucket_count > AHB_OPT_CAP_SPREAD) {
        bucket_count = AHB_OPT_CAP_SPREAD;
    }
#endif
#if AHB_OPT_CAP_SPREAD_DIV
    if (spread && spread_cap_ok && size > AHB_OPT_CAP_SPREAD_AT) bucket_count = size / AHB_OPT_CAP_SPREAD_DIV;
#endif
#endif

    int64_t bucket_size = (range - 1) / bucket_count + 1;

#if AHB_OPT_POW2
    // Smallest power of two >= bucket_size; it is still <= range - 1, so the
    // minimum and maximum keep landing in different buckets.
    int shift = 0;
    while (((int64_t)1 << shift) < bucket_size) shift++;
#if AHB_OPT_POW2 == 2
    if (((int64_t)1 << shift) > bucket_size) shift--;
#endif
    bucket_count = (int)((range - 1) >> shift) + 1;
#else
    const double bucket_size_inv = 1.0 / (double)bucket_size;
#endif

#if AHB_OPT_SHARED
    int owns_scratch = 0;
    if (!scratch) {
        scratch = malloc(size * sizeof(int));
        owns_scratch = 1;
    }
    int *bucket_data = malloc(bucket_count * sizeof(int) * AHBV_ARRAYS);
    if (!scratch || !bucket_data) {
        if (owns_scratch) free(scratch);
        free(bucket_data);
        qsort(arr, size, sizeof(int), ahbv_compare_ints);
        return;
    }
    int *buckets = scratch;
#else
    size_t total_size = size * sizeof(int) + bucket_count * sizeof(int) * AHBV_ARRAYS;
    char *memory = malloc(total_size);
    if (!memory) {
        qsort(arr, size, sizeof(int), ahbv_compare_ints);
        return;
    }

    int *buckets = (int*)memory;
    int *bucket_data = (int*)(buckets + size);
#endif

#if AHB_OPT_COMPACT
    // One array: it holds the counts first, then each bucket's write position.
    // After the scatter, positions[i] is the end of bucket i, which is also
    // the start of bucket i + 1.
    int *positions = bucket_data;
    int *bucket_sizes = bucket_data;
#else
    int *bucket_offsets = bucket_data;
    int *bucket_sizes = bucket_data + bucket_count;
    int *positions = bucket_data + 2 * bucket_count;
#endif

    memset(bucket_data, 0, bucket_count * sizeof(int) * AHBV_ARRAYS);

    i = 0;
    for (; i + 4 <= size; i += 4) {
        int b1 = AHBV_INDEX(arr[i]);
        int b2 = AHBV_INDEX(arr[i+1]);
        int b3 = AHBV_INDEX(arr[i+2]);
        int b4 = AHBV_INDEX(arr[i+3]);
        bucket_sizes[b1]++;
        bucket_sizes[b2]++;
        bucket_sizes[b3]++;
        bucket_sizes[b4]++;
    }
    for (; i < size; i++) bucket_sizes[AHBV_INDEX(arr[i])]++;

    int running_total = 0, max_bucket = 0;
    for (i = 0; i < bucket_count; i++) {
        int bsize = bucket_sizes[i];
        max_bucket = bsize > max_bucket ? bsize : max_bucket;
#if !AHB_OPT_COMPACT
        bucket_offsets[i] = running_total;
#endif
        positions[i] = running_total;
        running_total += bsize;
    }

#if AHB_OPT_WC
    // Software write combining: collect AHB_OPT_WC elements per bucket in a
    // small buffer and move them to the bucket with one copy.
    int *wc = NULL;
    if (bucket_count <= 4096 && size >= 262144) {
        wc = malloc((size_t)bucket_count * (AHB_OPT_WC + 1) * sizeof(int));
    }
    if (wc) {
        int *fill = wc + (size_t)bucket_count * AHB_OPT_WC;
        memset(fill, 0, (size_t)bucket_count * sizeof(int));
        for (i = 0; i < size; i++) {
            int value = arr[i];
            int b = AHBV_INDEX(value);
            int *block = wc + (size_t)b * AHB_OPT_WC;
            block[fill[b]++] = value;
            if (fill[b] == AHB_OPT_WC) {
                memcpy(&buckets[positions[b]], block, AHB_OPT_WC * sizeof(int));
                positions[b] += AHB_OPT_WC;
                fill[b] = 0;
            }
        }
        for (i = 0; i < bucket_count; i++) {
            memcpy(&buckets[positions[i]], wc + (size_t)i * AHB_OPT_WC, (size_t)fill[i] * sizeof(int));
            positions[i] += fill[i];
        }
        free(wc);
    } else
#endif
    for (i = 0; i < size; i++) buckets[positions[AHBV_INDEX(arr[i])]++] = arr[i];

    int threshold = size < 1024 ? size / 16 : AHB_OPT_THRESHOLD;

#if AHB_OPT_SAMPLE
    const int final_pass = spread && max_bucket <= AHB_OPT_FINAL_PASS;
#elif AHB_OPT_FINAL_PASS == 1
    const int final_pass = 1;
#elif AHB_OPT_FINAL_PASS > 1
    const int final_pass = max_bucket <= AHB_OPT_FINAL_PASS;
#else
    const int final_pass = 0;
#endif

    // With a final pass and no bucket large enough to recurse into, there is
    // nothing to do per bucket.
    if (!final_pass || max_bucket > threshold) {
        int next_start = 0;
        (void)next_start;
        for (i = 0; i < bucket_count; i++) {
#if AHB_OPT_COMPACT
            int boffset = next_start;
            int bsize = positions[i] - next_start;
            next_start = positions[i];
#else
            int boffset = bucket_offsets[i];
            int bsize = bucket_sizes[i];
#endif
            if (bsize > threshold) {
#if AHB_OPT_SHARED
                AHBV_REC(&buckets[boffset], arr + boffset, bsize);
#else
                AHBV_REC(&buckets[boffset], NULL, bsize);
#endif
            } else if (!final_pass && bsize > 1) {
                ahbv_insertion_sort(&buckets[boffset], bsize);
            }
        }
    }

    if (final_pass) {
        // Buckets are already in order relative to each other, so one
        // insertion sort over everything only ever moves elements within a
        // small bucket. It reads from the scratch buffer and writes into arr,
        // which also replaces the copy back.
        arr[0] = buckets[0];
        for (i = 1; i < size; i++) {
            int key = buckets[i];
            int j = i;
            while (j > 0 && arr[j - 1] > key) {
                arr[j] = arr[j - 1];
                j--;
            }
            arr[j] = key;
        }
    } else {
        memcpy(arr, buckets, size * sizeof(int));
    }

#if AHB_OPT_SHARED
    free(bucket_data);
    if (owns_scratch) free(scratch);
#else
    free(memory);
#endif
}

static void AHB_FN(int *arr, int size) {
    AHBV_REC(arr, NULL, size);
}

#undef AHBV_INDEX
#undef AHB_FN
#undef AHB_OPT_COUNTING
#undef AHB_OPT_POW2
#undef AHB_OPT_DIVISOR
#undef AHB_OPT_MAX_BUCKETS
#undef AHB_OPT_SHARED
#undef AHB_OPT_BUCKET_MULT
#undef AHB_OPT_THRESHOLD
#undef AHB_OPT_FINAL_PASS
#undef AHB_OPT_COMPACT
#undef AHB_OPT_SAMPLE
#undef AHB_OPT_COUNT_SPREAD
#undef AHB_OPT_SPARSE
#undef AHB_OPT_CAP_SKEW
#undef AHB_OPT_CAP_SPREAD
#undef AHB_OPT_CAP_SPREAD_AT
#undef AHB_OPT_CAP_SKEW_AT
#undef AHB_OPT_CAP_SPREAD_DIV
#undef AHB_OPT_CAP_GUARDS
#undef AHB_OPT_ORDERED_MIN
#undef AHB_OPT_ORDERED_TURNS
#undef AHB_OPT_WC
#undef AHBV_ARRAYS
#undef AHBV_REC
#undef AHBV_CAT
#undef AHBV_CAT2
