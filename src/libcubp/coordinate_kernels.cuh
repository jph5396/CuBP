#pragma once
#include "coordinate_conversions.h"

void setCoordinateReferences(
    const ENUMatrixTerms& matTerms,
    const WGS84::ECEFCoord& referencePoint,
    const WGS84::ENUCoord& enuReferencePoint,
    const ENUSpacing& enuSpacing
);

//cu_ecefToGeodetic converts coordinate forms
// using the algorithm described in
// vermeille (2002) https://link.springer.com/article/10.1007/s00190-002-0273-6

__device__ WGS84::GeodeticCoord cu_ecefToGeodetic(
    double x,
    double y,
    double z
);

__device__ WGS84::ECEFCoord cu_enuToEcef(
    double e,
    double n,
    double u
);

__global__ void genECEFCoordGrid(double* grid, int xSize, int ySize);

void launchGenECEFCoordGrid(double* grid, int xSize, int ySize);
