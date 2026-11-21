/*
 * Template for experimental AHB_Sort variants. Include it once per variant,
 * after "ahb_sort.h", with these macros defined (all are #undef'd at the end):
 *
 *   AHB_FN            name of the generated function (required)
 *   AHB_OPT_COUNTING  0 = off; C > 0 = plain counting sort when
 *                     range <= size * C / 4
 *   AHB_OPT_POW2      0 = bucket index by multiply (the old behaviour);
 *                     1 = power-of-two bucket width, rounded up (shift);
 *                     2 = power-of-two bucket width, rounded down (shift)
 *   AHB_OPT_DIVISOR   0 = variance-based bucket count (the old behaviour);
 *                     K > 0 = bucket count is size / K, no variance pass
 *
 * With every option at 0 the generated code is the algorithm as it was before
 * these options were adopted. ahb_sort.h now corresponds to
 * COUNTING 8, POW2 1, DIVISOR 2.
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

#if AHB_OPT_POW2
#define AHBV_INDEX(x) ((int)(((uint32_t)(x) - min_u) >> shift))
#else
#define AHBV_INDEX(x) ahbv_clamp_index((int)(((uint32_t)(x) - min_u) * bucket_size_inv), bucket_count)
#endif

static void AHB_FN(int *arr, int size) {
    if (size <= 1) return;
    if (size <= 16) {
        insertion_sort(arr, size);
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

    size_t total_size = size * sizeof(int) + bucket_count * sizeof(int) * 3;
    char *memory = malloc(total_size);
    if (!memory) {
        qsort(arr, size, sizeof(int), ahb_compare_ints);
        return;
    }

    int *buckets = (int*)memory;
    int *bucket_data = (int*)(buckets + size);
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
            AHB_FN(bucket, bsize);
        } else {
            insertion_sort(bucket, bsize);
        }
    }

    memcpy(arr, buckets, size * sizeof(int));
    free(memory);
}

#undef AHBV_INDEX
#undef AHB_FN
#undef AHB_OPT_COUNTING
#undef AHB_OPT_POW2
#undef AHB_OPT_DIVISOR
