#pragma once 
#include "coordinate_conversions.h"


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

__global__ void genECEFCoordGrid();

void setCoordinateReferences(
    ENUMatrixTerms matTerms,
    WGS84::GeodeticCoord referencePoint
);