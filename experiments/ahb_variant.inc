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
 *
 * With every option at 0 the generated code is the original algorithm.
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
    int bucket_count = size / AHB_OPT_DIVISOR;
#endif
    bucket_count = bucket_count < 2 ? 2 : bucket_count;
#if AHB_OPT_MAX_BUCKETS
    if (bucket_count > AHB_OPT_MAX_BUCKETS) bucket_count = AHB_OPT_MAX_BUCKETS;
#endif

    const uint32_t min_u = (uint32_t)min_value;

#if AHB_OPT_COUNTING
    // Narrow range: count each value and write the values straight back.
    if (range <= (int64_t)size * AHB_OPT_COUNTING / 4) {
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
    int *bucket_data = malloc(bucket_count * sizeof(int) * 3);
    if (!scratch || !bucket_data) {
        if (owns_scratch) free(scratch);
        free(bucket_data);
        qsort(arr, size, sizeof(int), ahbv_compare_ints);
        return;
    }
    int *buckets = scratch;
#else
    size_t total_size = size * sizeof(int) + bucket_count * sizeof(int) * 3;
    char *memory = malloc(total_size);
    if (!memory) {
        qsort(arr, size, sizeof(int), ahbv_compare_ints);
        return;
    }

    int *buckets = (int*)memory;
    int *bucket_data = (int*)(buckets + size);
#endif
    int *bucket_offsets = bucket_data;
    int *bucket_sizes = bucket_data + bucket_count;
    int *positions = bucket_data + 2 * bucket_count;

    memset(bucket_data, 0, bucket_count * sizeof(int) * 3);

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

    int running_total = 0;
    for (i = 0; i < bucket_count; i++) {
        bucket_offsets[i] = running_total;
        positions[i] = running_total;
        running_total += bucket_sizes[i];
    }

    for (i = 0; i < size; i++) buckets[positions[AHBV_INDEX(arr[i])]++] = arr[i];

    int threshold = size < 1024 ? size / 16 : 64;
    for (i = 0; i < bucket_count; i++) {
        int bsize = bucket_sizes[i];
        if (bsize <= 1) continue;

        int *bucket = &buckets[bucket_offsets[i]];
        if (bsize > threshold) {
#if AHB_OPT_SHARED
            AHBV_REC(bucket, arr + bucket_offsets[i], bsize);
#else
            AHBV_REC(bucket, NULL, bsize);
#endif
        } else {
            ahbv_insertion_sort(bucket, bsize);
        }
    }

    memcpy(arr, buckets, size * sizeof(int));
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
#undef AHBV_REC
#undef AHBV_CAT
#undef AHBV_CAT2
