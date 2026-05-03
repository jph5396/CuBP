#pragma once
#include <optional>
#include <vector>
#include "geodesy.h"

class CoordinateGridManager {
public:
    /*
        xSize          : number of grid points along the x (east) axis
        ySize          : number of grid points along the y (north) axis
        spacing        : grid spacing in meters (applied equally in x and y)
        referencePoint : scene center in ECEF — grid is centered here
        targetPoint    : optional geodetic point to shift the ENU origin;
                        defaults to the referencePoint's own geodetic position
    */
    CoordinateGridManager(
        int xSize,
        int ySize,
        double spacing,
        WGS84::ECEFCoord referencePoint,
        std::optional<WGS84::GeodeticCoord> targetPoint = std::nullopt
    );
    ~CoordinateGridManager();

    // Allocate device memory and run the GPU kernel to build the ECEF coordinate grid.
    void createGrid();

    // Copy the device grid to a host vector of shape (xSize*ySize, 3), row-major doubles.
    std::vector<double> gridToHost() const;

    int numPoints() const { return xSize_ * ySize_; }

    // Device pointer to the flat (xSize*ySize, 3) ECEF grid — valid after createGrid().
    const double* deviceGrid() const { return d_grid_; }

private:
    ENUMatrixTerms terms_;
    int xSize_;
    int ySize_;
    double spacing_;
    WGS84::ECEFCoord referencePoint_;
    WGS84::ENUCoord targetEnu_;
    double* d_grid_ = nullptr;

    static ENUMatrixTerms computeEnuTerms(WGS84::ECEFCoord coord);
    static WGS84::ENUCoord optionallyResolveTargetCoordinates(
        WGS84::ECEFCoord referencePoint,
        const ENUMatrixTerms& matTerms,
        std::optional<WGS84::GeodeticCoord> targetPoint
    );
};
