/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// for compare_floats_asc
#include "permutedsort.h"
// for QSORT_R
#include "os-features.h"

/* k-th smallest (0-based) of a[0..n-1], rearranging a. Quickselect with a median-of-three pivot:
   O(n) instead of the O(n log n) qsort this used to be, which dominated image2xy on large frames
   (dmedsmooth takes one median per 201x201 box). Returns exactly what sorted[k] would. */
static float select_kth(float *a, unsigned long n, unsigned long k) {
    unsigned long lo = 0, hi = n - 1;
    while (hi - lo > 16) {
        unsigned long mid = lo + (hi - lo) / 2, i = lo, j = hi;
        float t, pivot;
        if (a[mid] < a[lo]) { t = a[mid]; a[mid] = a[lo]; a[lo] = t; }
        if (a[hi] < a[lo])  { t = a[hi];  a[hi]  = a[lo]; a[lo] = t; }
        if (a[hi] < a[mid]) { t = a[hi];  a[hi]  = a[mid]; a[mid] = t; }
        pivot = a[mid];
        // Hoare partition; a[lo] <= pivot <= a[hi] act as sentinels so the scans stay in range
        for (;;) {
            while (a[i] < pivot) i++;
            while (a[j] > pivot) j--;
            if (i >= j) break;
            t = a[i]; a[i] = a[j]; a[j] = t;
            i++; j--;
        }
        // a[lo..j] <= pivot <= a[j+1..hi]
        if (k <= j) hi = j;
        else lo = j + 1;
    }
    // small range: insertion sort
    for (unsigned long i = lo + 1; i <= hi; i++) {
        float v = a[i];
        unsigned long j = i;
        while (j > lo && a[j - 1] > v) { a[j] = a[j - 1]; j--; }
        a[j] = v;
    }
    return a[k];
}

// k-th smallest of arr[0..n-1], rearranging arr; for callers whose buffer is scratch anyway
// (dmedsmooth boxes) - no copy, no shared state, so it is safe from several threads
float dselip_inplace(unsigned long k, unsigned long n, float *arr) {
    return select_kth(arr, n, k);
}

#ifdef SIMPLEXY_REENTRANT

// this is slower, because each call needs to malloc, but it is reentrant
float dselip(unsigned long k, unsigned long n, const float *arr) {
    float* sorted_data = malloc(sizeof(float) * n);
    memcpy(sorted_data, arr, sizeof(float)*n);
    float kth_item = select_kth(sorted_data, n, k);
    free(sorted_data);
    return kth_item;
}

void dselip_cleanup() {
}

#else

static int high_water_mark = 0;
static float* past_data = NULL;

float dselip(unsigned long k, unsigned long n, float *arr) {
    if (n > high_water_mark) {
        free(past_data);
        past_data = malloc(sizeof(float) * n);
        high_water_mark = n;
        //printf("dselip watermark=%lu\n",n);
    }
    memcpy(past_data, arr, sizeof(float) * n);
    return select_kth(past_data, n, k);
}

void dselip_cleanup() {
    free(past_data);
    past_data = NULL;
    high_water_mark = 0;
}

#endif
