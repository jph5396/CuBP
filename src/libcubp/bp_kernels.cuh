#pragma once
#include <cufft.h>

/*
    Accumulate one FFT'd pulse into the image.
*/
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
);

/*
    Apply the final center-pulse phase correction:
*/
void launchFinalCorrection(
    float2* image,
    const double* grid,
    const double* centerSrcPos,
    double srpX, double srpY, double srpZ,
    double kc,
    int numPixels
);
