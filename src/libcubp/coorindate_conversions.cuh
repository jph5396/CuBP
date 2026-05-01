#pragma once 
#include "coordinate_conversions.h"


// WGS 1984 constants for coordinate transformations. 
__constant__ double cu_A = WGS84::A; 
__constant__ double cu_B = WGS84::B; 
__constant__ double cu_F = WGS84::F; 
__constant__ double cu_E2 = WGS84::E2; 
__constant__ double cu_A2 = WGS84::A2; 
__constant__ double cu_E4 = WGS84::E4;


//cu_ecefToGeodetic converts coordniate forms 
// using the algorithm described in 
// vermeille (2002) https://link.springer.com/article/10.1007/s00190-002-0273-6

__device__ WGS84::GeodeticCoord cu_ecefToGeodetic(
    double x,
    double y,
    double z
);

__device__  WGS84::ECEFCoord cu_enuToEcef(
    double e, 
    double n,
    double u
);