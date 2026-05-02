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


__global__ void genECEFCoordGrid(double* grid, int xSize, int ySize) {};

void launchGenECEFCoordGrid(double* grid, int xSize, int ySize) {};