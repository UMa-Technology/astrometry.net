/*
# This file is part of the Astrometry.net suite.
# Licensed under a 3-clause BSD style license - see LICENSE
*/


#define GLUE2(a,b) a ## b
#define GLUE(a,b) GLUE2(a, b)

// Optimize version of dsmooth, with a separated Gaussian convolution.
void GLUE(dsmooth2, SUFFIX)(IMGTYPE *image,
							int nx,
							int ny,
							float sigma,
							float *smooth) {
#undef GLUE
#undef GLUE2
	int i, npix, half;
	float neghalfinvvar, total, scale, dx;
	float* kernel1D;
    float* kernel_shifted;

	// make the kernel
	npix = 2 * ((int) ceilf(3. * sigma)) + 1;
	half = npix / 2;
	kernel1D =  malloc(npix * sizeof(float));
	neghalfinvvar = -1.0 / (2.0 * sigma * sigma);
	for (i=0; i<npix; i++) {
        dx = ((float) i - 0.5 * ((float)npix - 1.));
        kernel1D[i] = exp((dx * dx) * neghalfinvvar);
	}

	// normalize the kernel
	total = 0.0;
	for (i=0; i<npix; i++)
        total += kernel1D[i];
	scale = 1. / total;
	for (i=0; i<npix; i++)
        kernel1D[i] *= scale;

    // Here's some trickery: we set "kernel_shifted" to be an array where:
    //   kernel_shifted[0] is the middle of the array,
    //   kernel_shifted[-half] is the left edge (ie the first sample),
    //   kernel_shifted[half] is the right edge (last sample)
	kernel_shifted = kernel1D + half;
	// rows of the x pass and columns of the y pass are independent; every thread keeps its own
	// scratch line and the barrier between the two "omp for" loops orders the passes
#pragma omp parallel
	{
		float* smooth_temp = malloc(sizeof(float) * MAX(nx, ny));
		int i, j, start, end, sample;
		float sum;
		// convolve in x direction, dumping results into smooth_temp
#pragma omp for
		for (j=0; j<ny; j++) {
			IMGTYPE* imagerow = image + j*nx;
			for (i=0; i<nx; i++) {
				/*
				 The outer loops are over OUTPUT pixels;
				 the "sample" loop is over INPUT pixels.
				 We're summing over the input pixels that contribute to the value
				 of the output pixel.
				 */
				start = MAX(0, i - half);
				end = MIN(nx-1, i + half);
				sum = 0.0;
				for (sample=start; sample <= end; sample++)
					sum += imagerow[sample] * kernel_shifted[sample - i];
				smooth_temp[i] = sum;
			}
			memcpy(smooth + j*nx, smooth_temp, nx * sizeof(float));
		}
		// convolve in the y direction, dumping results into smooth
#pragma omp for
		for (i=0; i<nx; i++) {
			float* imagecol = smooth + i;
			for (j=0; j<ny; j++) {
				start = MAX(0, j - half);
				end = MIN(ny-1, j + half);
				sum = 0.0;
				for (sample=start; sample<=end; sample++)
					sum += imagecol[sample*nx] * kernel_shifted[sample - j];
				smooth_temp[j] = sum;
			}
			for (j=0; j<ny; j++)
				smooth[i + j*nx] = smooth_temp[j];
		}
		FREEVEC(smooth_temp);
	}
	FREEVEC(kernel1D);
}


