#pragma once
#include <optional>
#include <vector>
#include "geodesy.h"

class CoordinateGridManager {
public:
    CoordinateGridManager(
        int xSize,
        int ySize,
        double spacing,
        WGS84::ECEFCoord referencePoint,
        std::optional<WGS84::GeodeticCoord> targetPoint = std::nullopt
    );
    ~CoordinateGridManager();

    void createGrid();
    std::vector<double> gridToHost() const;
    int numPoints() const { return xSize_ * ySize_; }
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
