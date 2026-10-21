/*
 # This file is part of the Astrometry.net suite.
 # Licensed under a 3-clause BSD style license - see LICENSE
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "os-features.h"
#include "simplexy-common.h"

/*
 * dmedsmooth.c
 *
 * Median smooth an image -- actually, compute median values for a grid of points,
 * then interpolate.
 *
 * Mike Blanton
 * 1/2006 */


float dselip_inplace(unsigned long k, unsigned long n, float *arr);


int dmedsmooth_gridpoints(int nx, int halfbox, int* p_nxgrid, int** p_xgrid,
                          int** p_xlo, int** p_xhi) {
    int nxgrid;
    int* xgrid;
    int* xlo;
    int* xhi;
    int xoff;
    int i;
    nxgrid = MAX(1, nx / halfbox) + 2;
    *p_nxgrid = nxgrid;
    // "xgrid" are the centers.
    // "xlo" are the (inclusive) lower-bounds
    // "xhi" are the (inclusive) upper-bounds
    // the grid cells may overlap.
    *p_xgrid = xgrid = (int *) malloc((size_t)nxgrid * sizeof(int));
    *p_xlo   = xlo   = (int *) malloc((size_t)nxgrid * sizeof(int));
    *p_xhi   = xhi   = (int *) malloc((size_t)nxgrid * sizeof(int));
    xoff = (nx - 1 - (nxgrid - 3) * halfbox) / 2;
    for (i = 1; i < nxgrid - 1; i++)
        xgrid[i] = (i - 1) * halfbox + xoff;
    xgrid[0] = xgrid[1] - halfbox;
    xgrid[nxgrid - 1] = xgrid[nxgrid - 2] + halfbox;
    for (i = 0; i < nxgrid; i++) {
        xlo[i] = MAX(xgrid[i] - halfbox, 0);
        xhi[i] = MIN(xgrid[i] + halfbox, nx-1);
    }
    return 0;
}

int dmedsmooth_grid(const float* image,
                    const uint8_t *masked,
                    int nx,
                    int ny,
                    int halfbox,
                    float **p_grid, int** p_xgrid, int** p_ygrid,
                    int* p_nxgrid, int* p_nygrid) {
    float* grid = NULL;
    int *xlo = NULL;
    int *xhi = NULL;
    int *ylo = NULL;
    int *yhi = NULL;
    int nxgrid, nygrid;
    int i, j;

    if (dmedsmooth_gridpoints(nx, halfbox, &nxgrid, p_xgrid, &xlo, &xhi)) {
        return 1;
    }
    if (dmedsmooth_gridpoints(ny, halfbox, &nygrid, p_ygrid, &ylo, &yhi)) {
        FREEVEC(xlo);
        FREEVEC(xhi);
        FREEVEC(*p_xgrid);
        return 1;
    }
    *p_nxgrid = nxgrid;
    *p_nygrid = nygrid;

    /*
     for (i=0; i<nxgrid; i++)
     printf("xgrid %i, xlo %i, xhi %i\n", (*p_xgrid)[i], xlo[i], xhi[i]);
     for (i=0; i<nygrid; i++)
     printf("ygrid %i, ylo %i, yhi %i\n", (*p_ygrid)[i], ylo[i], yhi[i]);
     */

    // the median-filtered image (subsampled on a grid).
    *p_grid = grid = (float *) malloc((size_t)(nxgrid * nygrid) *
                                      sizeof(float));

    // one box per grid point, boxes are independent: each thread selects the median in
    // its own scratch copy, in the same order as before
#pragma omp parallel
    {
        float* arr = (float *) malloc((size_t)((halfbox * 2 + 5) *
                                               (halfbox * 2 + 5)) * sizeof(float));
        int i, j;
#pragma omp for collapse(2)
        for (j=0; j<nygrid; j++) {
            for (i=0; i<nxgrid; i++) {
                int nb = 0, jp, ip;
                for (jp=ylo[j]; jp<=yhi[j]; jp++) {
                    const float* imageptr = image + xlo[i] + jp * nx;
                    float f;
                    if (masked) {
                        const uint8_t* maskptr = masked + xlo[i] + jp * nx;
                        for (ip=xlo[i]; ip<=xhi[i]; ip++, imageptr++, maskptr++) {
                            if (*maskptr)
                                continue;
                            f = (*imageptr);
                            if (!isfinite(f))
                                continue;
                            arr[nb] = f;
                            nb++;
                        }
                    } else {
                        for (ip=xlo[i]; ip<=xhi[i]; ip++, imageptr++) {
                            f = (*imageptr);
                            if (!isfinite(f))
                                continue;
                            arr[nb] = f;
                            nb++;
                        }
                    }
                }
                if (nb > 1) {
                    grid[i + j*nxgrid] = dselip_inplace(nb / 2, nb, arr);
                } else {
                    grid[i + j*nxgrid] = 0.0;
                }
            }
        }
        FREEVEC(arr);
    }
    FREEVEC(xlo);
    FREEVEC(ylo);
    FREEVEC(xhi);
    FREEVEC(yhi);
    return 0;
}

int dmedsmooth_interpolate(const float* grid,
                           int nx, int ny,
                           int nxgrid, int nygrid,
                           const int* xgrid, const int* ygrid,
                           int halfbox,
                           float* smooth) {
    // Per grid row / column: the image rows / columns it touches and its +/- spans, exactly as
    // the input-driven loop used to compute them for every grid point.
    int *jst = malloc(nygrid * sizeof(int)), *jnd = malloc(nygrid * sizeof(int));
    int *ypsize = malloc(nygrid * sizeof(int)), *ymsize = malloc(nygrid * sizeof(int));
    int *ist = malloc(nxgrid * sizeof(int)), *ind = malloc(nxgrid * sizeof(int));
    int *xpsize = malloc(nxgrid * sizeof(int)), *xmsize = malloc(nxgrid * sizeof(int));
    int i, j, jp;
    for (j = 0; j < nygrid; j++) {
        jst[j] = (int) ( (float) ygrid[j] - halfbox * 1.5);
        jnd[j] = (int) ( (float) ygrid[j] + halfbox * 1.5);
        if (jst[j] < 0)
            jst[j] = 0;
        if (jnd[j] > ny - 1)
            jnd[j] = ny - 1;
        ypsize[j] = halfbox;
        ymsize[j] = halfbox;
        if (j == 0)
            ypsize[j] = ygrid[1] - ygrid[0];
        if (j == 1)
            ymsize[j] = ygrid[1] - ygrid[0];
        if (j == nygrid - 2)
            ypsize[j] = ygrid[nygrid - 1] - ygrid[nygrid - 2];
        if (j == nygrid - 1)
            ymsize[j] = ygrid[nygrid - 1] - ygrid[nygrid - 2];
    }
    for (i = 0; i < nxgrid; i++) {
        ist[i] = (long) ( (float) xgrid[i] - halfbox * 1.5);
        ind[i] = (long) ( (float) xgrid[i] + halfbox * 1.5);
        if (ist[i] < 0)
            ist[i] = 0;
        if (ind[i] > nx - 1)
            ind[i] = nx - 1;
        xpsize[i] = halfbox;
        xmsize[i] = halfbox;
        if (i == 0)
            xpsize[i] = xgrid[1] - xgrid[0];
        if (i == 1)
            xmsize[i] = xgrid[1] - xgrid[0];
        if (i == nxgrid - 2)
            xpsize[i] = xgrid[nxgrid - 1] - xgrid[nxgrid - 2];
        if (i == nxgrid - 1)
            xmsize[i] = xgrid[nxgrid - 1] - xgrid[nxgrid - 2];
    }
    // Output-driven: every image row gathers its contributions in the original (j, i) order,
    // so the sums are bit-identical to the input-driven version and the rows are independent.
#pragma omp parallel for private(i, j)
    for (jp = 0; jp < ny; jp++) {
        float* row = smooth + (size_t)jp * nx;
        for (i = 0; i < nx; i++)
            row[i] = 0.;
        for (j = 0; j < nygrid; j++) {
            // Interpolate with a kernel that is two parabolas spliced
            // together: in [-1.5, -0.5] and [0.5, 1.5], 0.5 * (|y|-1.5)^2
            // so at +- 0.5 it has value 0.5.
            // at +- 1.5 it has value 0.
            // in [-0.5, 0.5]: 0.75 - (y^2)
            // so at +- 0.5 it has value 0.5
            // at 0 it has value 0.75
            float dy, ykernel;
            if (jp < jst[j] || jp > jnd[j])
                continue;
            dy = (float)(jp - ygrid[j]);
            if (dy >= 0) {
                dy /= (float)ypsize[j];
            } else {
                dy /= (float)(-ymsize[j]);
            }
            if ((dy >= 0.5) && (dy < 1.5))
                ykernel = 0.5 * (dy - 1.5) * (dy - 1.5);
            else if (dy < 0.5)
                ykernel = 0.75 - (dy * dy);
            else
                // ykernel = 0
                continue;
            for (i = 0; i < nxgrid; i++) {
                int ip;
                for (ip = ist[i]; ip <= ind[i]; ip++) {
                    float dx, xkernel;
                    dx = (float)(ip - xgrid[i]);
                    if (dx >= 0) {
                        dx /= (float)xpsize[i];
                    } else {
                        dx /= (float)(-xmsize[i]);
                    }
                    if ((dx >= 0.5) && (dx < 1.5))
                        xkernel = 0.5 * (dx - 1.5) * (dx - 1.5);
                    else if (dx < 0.5)
                        xkernel = 0.75 - (dx * dx);
                    else
                        // xkernel = 0
                        continue;
                    row[ip] += xkernel * ykernel * grid[i + j * nxgrid];
                }
            }
        }
    }
    FREEVEC(jst); FREEVEC(jnd); FREEVEC(ypsize); FREEVEC(ymsize);
    FREEVEC(ist); FREEVEC(ind); FREEVEC(xpsize); FREEVEC(xmsize);
    return 0;
}

int dmedsmooth(const float *image,
               const uint8_t *masked,
               int nx,
               int ny,
               int halfbox,
               float *smooth)
{
    float *grid = NULL;
    int *xgrid = NULL;
    int *ygrid = NULL;
    int nxgrid, nygrid;

    if (dmedsmooth_grid(image, masked, nx, ny, halfbox,
                        &grid, &xgrid, &ygrid, &nxgrid, &nygrid)) {
        return 0;
    }
    if (dmedsmooth_interpolate(grid, nx, ny, nxgrid, nygrid,
                               xgrid, ygrid, halfbox, smooth)) {
        return 0;
    }

    FREEVEC(grid);
    FREEVEC(xgrid);
    FREEVEC(ygrid);

    return 1;
}
