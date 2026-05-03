#pragma once
#include <complex>
#include <vector>
#include <cufft.h>
#include "geodesy.h"
#include "grid_manager.h"

class BPManager {
public:
    /*
        bandwidth   : channel bandwidth in Hz (FxBW)
        fc          : center frequency in Hz (FxC)
        srpEcef     : scene reference point in ECEF
        pulseLimit  : number of pulses to process
        rangeBinLen : number of range bins per pulse
        gridMgr     : coordinate grid (must outlive BPManager)
        srcPos      : sensor positions, host array of shape (pulseLimit, 3), row-major doubles
    */
    BPManager(
        int xSize,
        int ySize,
        int pulseLimit,
        int rangeBinLen,
        double bandwidth,
        double fc,
        WGS84::ECEFCoord srpEcef,
        const CoordinateGridManager& gridMgr,
        const double* srcPos
    );
    ~BPManager();

    /*
        Copy raw I/Q for pulse pulseIdx from pulseData to device,
        apply fftshift, run cuFFT, apply fftshift, then accumulate into the image.
        pulseData : host array of shape (rangeBinLen, 2), row-major float32 I/Q, little-endian.
    */
    void processPulse(int pulseIdx, const float* pulseData);

    // Apply the final center-pulse phase correction after all pulses are accumulated.
    void finalizeImage();

    std::vector<std::complex<float>> imageToHost() const;

    int numPixels() const { return xSize_ * ySize_; }
    int xSize() const { return xSize_; }
    int ySize() const { return ySize_; }

private:
    int xSize_;
    int ySize_;
    int pulseLimit_;
    int rangeBinLen_;
    double bandwidth_;
    double fc_;
    WGS84::ECEFCoord srpEcef_;

    const CoordinateGridManager& gridMgr_;  // not owned
    cufftHandle plan_;                      // 1D C2C FFT plan for one pulse

    double* d_srcPos_ = nullptr;            // (pulseLimit, 3)
    cufftComplex* d_signal_ = nullptr;      // (rangeBinLen,) — working buffer per pulse
    std::complex<float>* d_image_ = nullptr;// (xSize * ySize)
};
