#include "coorindate_conversions.cuh"
#include "coordinate_conversions.h"



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
    ENUMatrixTerms matTerms, 
    WGS84::GeodeticCoord referencePoint
) {};