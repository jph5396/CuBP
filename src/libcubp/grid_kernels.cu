#include "grid_kernels.cuh"
#include "grid_manager.h"

// WGS 1984 constants for coordinate transformations. 
__constant__ double cu_A = WGS84::A; 
__constant__ double cu_B = WGS84::B; 
__constant__ double cu_F = WGS84::F; 
__constant__ double cu_E2 = WGS84::E2; 
__constant__ double cu_A2 = WGS84::A2; 
__constant__ double cu_E4 = WGS84::E4;

// Constants to be set at runtime.
/*
    Target centerpoint in output Image in ENU coordinates. 
    used when generating grid points. 
*/
__constant__ double e_ref; 
__constant__ double n_ref; 
__constant__ double u_ref; 

/*
    Target centerpoint in  output image in ECEF coordinates. 
    used when converting grid to ECEF coordinates. 
*/

__constant__ double x_ref; 
__constant__ double y_ref; 
__constant__ double z_ref; 

// spacing in meters
__constant__ double d_e;
__constant__ double d_n;
__constant__ double d_u;

// Enu matrix terms
__constant__ double sla;
__constant__ double cla;
__constant__ double slo;
__constant__ double clo;
__constant__ double sla_clo;
__constant__ double sla_slo;
__constant__ double cla_clo;
__constant__ double cla_slo;


void setCoordinateReferences(
    const ENUMatrixTerms& matTerms,
    const WGS84::ECEFCoord& referencePoint,
    const WGS84::ENUCoord& enuReferencePoint,
    const ENUSpacing& enuSpacing
) {
    // set enu ref points 
    cudaMemcpyToSymbol(e_ref, &enuReferencePoint.e, sizeof(double)); 
    cudaMemcpyToSymbol(n_ref, &enuReferencePoint.n, sizeof(double));
    cudaMemcpyToSymbol(u_ref, &enuReferencePoint.u, sizeof(double));

    // ECEF reference point.
    cudaMemcpyToSymbol(x_ref, &referencePoint.x, sizeof(double));
    cudaMemcpyToSymbol(y_ref, &referencePoint.y, sizeof(double)); 
    cudaMemcpyToSymbol(z_ref, &referencePoint.z, sizeof(double)); 

    //spacing in meters.
    cudaMemcpyToSymbol(d_e, &enuSpacing.d_e, sizeof(double)); 
    cudaMemcpyToSymbol(d_n, &enuSpacing.d_n, sizeof(double));
    cudaMemcpyToSymbol(d_u, &enuSpacing.d_u, sizeof(double));

    //mat terms
    cudaMemcpyToSymbol(sla, &matTerms.sla, sizeof(double)); 
    cudaMemcpyToSymbol(cla, &matTerms.cla, sizeof(double));
    cudaMemcpyToSymbol(slo, &matTerms.slo, sizeof(double));
    cudaMemcpyToSymbol(clo, &matTerms.clo, sizeof(double));
    cudaMemcpyToSymbol(sla_clo, &matTerms.sla_clo, sizeof(double));
    cudaMemcpyToSymbol(sla_slo, &matTerms.sla_slo, sizeof(double));
    cudaMemcpyToSymbol(cla_clo, &matTerms.cla_clo, sizeof(double));
    cudaMemcpyToSymbol(cla_slo, &matTerms.cla_slo, sizeof(double)); 
};


// this currently isn't used but I figured it should be implemented anyway.
__device__ WGS84::GeodeticCoord cu_ecefToGeodetic(double x, double y, double z) {
    double x_y_euclid = sqrt(x * x + y * y);

    double p = (x * x + y * y) / cu_A2;
    double q = ((1.0 - cu_E2) / cu_A2) * (z * z);
    double r = (p + q - cu_E4) / 6.0;
    double s = cu_E4 * ((p * q) / (4.0 * r * r * r));
    double t = cbrt(1.0 + s + sqrt(s * (2.0 + s)));
    double u = r * (1.0 + t + 1.0 / t);
    double v = sqrt(u * u + cu_E4 * q);
    double w = cu_E2 * ((u + v - q) / (2.0 * v));
    double k = sqrt(u + v + w * w) - w;
    double D = (k * x_y_euclid) / (k + cu_E2);

    double lon = 2.0 * atan2(y, x + x_y_euclid);
    double lat = 2.0 * atan2(z, D + sqrt(D * D + z * z));
    double height = ((k + cu_E2 - 1.0) / k) * sqrt(D * D + z * z);

    return WGS84::GeodeticCoord{ lat, lon, height };
}

__device__ WGS84::ECEFCoord cu_enuToEcef(double e, double n, double u) {
    return WGS84::ECEFCoord{
        x_ref + (-slo * e - sla_clo * n + cla_clo * u),
        y_ref + ( clo * e - sla_slo * n + cla_slo * u),
        z_ref + (           cla    * n  + sla     * u)
    };
}

__global__ void genECEFCoordGrid(double* grid, int xSize, int ySize) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= xSize * ySize) return;

    int col = k % xSize;
    int row = k / xSize;

    double e = e_ref + (col - (xSize - 1) * 0.5) * d_e;
    double n = n_ref + (row - (ySize - 1) * 0.5) * d_n;

    WGS84::ECEFCoord ecef = cu_enuToEcef(e, n, 0.0);

    grid[k * 3 + 0] = ecef.x;
    grid[k * 3 + 1] = ecef.y;
    grid[k * 3 + 2] = ecef.z;
}

void launchGenECEFCoordGrid(double* grid, int xSize, int ySize) {
    int total = xSize * ySize;
    int blockSize = 256;
    int gridSize = (total + blockSize - 1) / blockSize;
    genECEFCoordGrid<<<gridSize, blockSize>>>(grid, xSize, ySize);
    cudaDeviceSynchronize();
}