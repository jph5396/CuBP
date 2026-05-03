#include "bp_manager.h"
#include "bp_kernels.cuh"

static constexpr double SPEED_OF_LIGHT = 2.99792458e8;

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
} while(0)

/*
    Multiplying by (-1)^n is equivalent to fftshift for a 1D array.
    Applying this before and after the FFT replicates fftshift(fft(fftshift(x))).
*/
__global__ void applyFftShift(cufftComplex* data, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    if (n % 2 == 1) {
        data[n].x = -data[n].x;
        data[n].y = -data[n].y;
    }
}

BPManager::BPManager(
    int xSize,
    int ySize,
    int pulseLimit,
    int rangeBinLen,
    double bandwidth,
    double fc,
    WGS84::ECEFCoord srpEcef,
    const CoordinateGridManager& gridMgr,
    const double* srcPos
):
    xSize_(xSize),
    ySize_(ySize),
    pulseLimit_(pulseLimit),
    rangeBinLen_(rangeBinLen),
    bandwidth_(bandwidth),
    fc_(fc),
    srpEcef_(srpEcef),
    gridMgr_(gridMgr)
{
    size_t srcPosBytes = pulseLimit * 3 * sizeof(double);
    cudaMalloc(&d_srcPos_, srcPosBytes);
    cudaMemcpy(d_srcPos_, srcPos, srcPosBytes, cudaMemcpyHostToDevice);

    cudaMalloc(&d_signal_, rangeBinLen * sizeof(cufftComplex));

    cudaMalloc(&d_image_, numPixels() * sizeof(std::complex<float>));
    cudaMemset(d_image_, 0, numPixels() * sizeof(std::complex<float>));

    cufftPlan1d(&plan_, rangeBinLen, CUFFT_C2C, 1);
}

BPManager::~BPManager() {
    cufftDestroy(plan_);
    if (d_srcPos_ != nullptr) cudaFree(d_srcPos_);
    if (d_signal_ != nullptr) cudaFree(d_signal_);
    if (d_image_  != nullptr) cudaFree(d_image_);
}

void BPManager::processPulse(int pulseIdx, const float* pulseData) {
    CUDA_CHECK(cudaMemcpy(d_signal_, pulseData, rangeBinLen_ * sizeof(cufftComplex), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (rangeBinLen_ + blockSize - 1) / blockSize;

    applyFftShift<<<gridSize, blockSize>>>(d_signal_, rangeBinLen_);
    cufftExecC2C(plan_, d_signal_, d_signal_, CUFFT_FORWARD);
    applyFftShift<<<gridSize, blockSize>>>(d_signal_, rangeBinLen_);

    double kc         = 4.0 * M_PI * fc_ / SPEED_OF_LIGHT;
    double rangeStep  = SPEED_OF_LIGHT / (2.0 * bandwidth_);
    double rangeStart = -(rangeBinLen_ / 2) * rangeStep;

    const double* d_pulseSrcPos = d_srcPos_ + pulseIdx * 3;
    launchAccumulatePulse(
        reinterpret_cast<float2*>(d_image_),
        d_signal_,
        gridMgr_.deviceGrid(),
        d_pulseSrcPos,
        srpEcef_.x, srpEcef_.y, srpEcef_.z,
        kc, rangeStart, rangeStep,
        rangeBinLen_, numPixels()
    );
    CUDA_CHECK(cudaGetLastError());
}

void BPManager::finalizeImage() {
    double kc = 4.0 * M_PI * fc_ / SPEED_OF_LIGHT;
    int centerPulse = pulseLimit_ / 2;
    const double* d_centerSrcPos = d_srcPos_ + centerPulse * 3;
    launchFinalCorrection(
        reinterpret_cast<float2*>(d_image_),
        gridMgr_.deviceGrid(),
        d_centerSrcPos,
        srpEcef_.x, srpEcef_.y, srpEcef_.z,
        kc, numPixels()
    );
    CUDA_CHECK(cudaGetLastError());
}

std::vector<std::complex<float>> BPManager::imageToHost() const {
    size_t n = numPixels();
    std::vector<std::complex<float>> host(n);
    CUDA_CHECK(cudaMemcpy(host.data(), d_image_, n * sizeof(std::complex<float>), cudaMemcpyDeviceToHost));
    return host;
}
