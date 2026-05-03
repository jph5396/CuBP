#include <math.h>
#include "bp_kernels.cuh"



__global__ void accumulatePulseKernel(
    float2* image,
    const cufftComplex* signal,
    const double* grid,
    const double* srcPos,
    double srpX, double srpY, double srpZ,
    double kc,
    double rangeStart,
    double rangeStep,
    int rangeBinLen,
    int numPixels
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= numPixels) return;

    // Range from sensor to SRP (same for all pixels in this pulse).
    double sdx = srcPos[0] - srpX;
    double sdy = srcPos[1] - srpY;
    double sdz = srcPos[2] - srpZ;
    double r_center = sqrt(sdx*sdx + sdy*sdy + sdz*sdz);

    // Range from sensor to this pixel.
    double pdx = grid[k*3+0] - srcPos[0];
    double pdy = grid[k*3+1] - srcPos[1];
    double pdz = grid[k*3+2] - srcPos[2];
    double r_pixel = sqrt(pdx*pdx + pdy*pdy + pdz*pdz);

    double dr = r_center - r_pixel;

    // Linear interpolation of signal at dr, clamped to array bounds.
    double t = (dr - rangeStart) / rangeStep;
    t = fmax(0.0, fmin((double)(rangeBinLen - 1), t));

    int idx = (int)t;
    if (idx >= rangeBinLen - 1) idx = rangeBinLen - 2;
    double frac = t - idx;

    float2 s0 = signal[idx];
    float2 s1 = signal[idx + 1];
    float re = (float)((1.0 - frac) * s0.x + frac * s1.x);
    float im = (float)((1.0 - frac) * s0.y + frac * s1.y);

    // Apply phase correction and accumulate.
    double phase = -kc * dr;
    float cos_p = (float)cos(phase);
    float sin_p = (float)sin(phase);

    image[k].x += re * cos_p - im * sin_p;
    image[k].y += re * sin_p + im * cos_p;
}

__global__ void finalCorrectionKernel(
    float2* image,
    const double* grid,
    const double* centerSrcPos,
    double srpX, double srpY, double srpZ,
    double kc,
    int numPixels
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= numPixels) return;

    double sdx = centerSrcPos[0] - srpX;
    double sdy = centerSrcPos[1] - srpY;
    double sdz = centerSrcPos[2] - srpZ;
    double r_center = sqrt(sdx*sdx + sdy*sdy + sdz*sdz);

    double pdx = grid[k*3+0] - centerSrcPos[0];
    double pdy = grid[k*3+1] - centerSrcPos[1];
    double pdz = grid[k*3+2] - centerSrcPos[2];
    double r_pixel = sqrt(pdx*pdx + pdy*pdy + pdz*pdz);

    double dr = r_center - r_pixel;
    double phase = kc * dr;  // positive for final correction

    float cos_p = (float)cos(phase);
    float sin_p = (float)sin(phase);

    float2 px = image[k];
    image[k].x = px.x * cos_p - px.y * sin_p;
    image[k].y = px.x * sin_p + px.y * cos_p;
}

void launchAccumulatePulse(
    float2* image,
    const cufftComplex* signal,
    const double* grid,
    const double* srcPos,
    double srpX, double srpY, double srpZ,
    double kc,
    double rangeStart,
    double rangeStep,
    int rangeBinLen,
    int numPixels
) {
    int blockSize = 256;
    int gridSize = (numPixels + blockSize - 1) / blockSize;
    accumulatePulseKernel<<<gridSize, blockSize>>>(
        image, signal, grid, srcPos,
        srpX, srpY, srpZ,
        kc, rangeStart, rangeStep,
        rangeBinLen, numPixels
    );
    cudaDeviceSynchronize();
}

void launchFinalCorrection(
    float2* image,
    const double* grid,
    const double* centerSrcPos,
    double srpX, double srpY, double srpZ,
    double kc,
    int numPixels
) {
    int blockSize = 256;
    int gridSize = (numPixels + blockSize - 1) / blockSize;
    finalCorrectionKernel<<<gridSize, blockSize>>>(
        image, grid, centerSrcPos,
        srpX, srpY, srpZ,
        kc, numPixels
    );
    cudaDeviceSynchronize();
}
