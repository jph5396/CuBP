#include <optional>
#include "grid_manager.h"
#include "grid_kernels.cuh"

ENUMatrixTerms CoordinateGridManager::computeEnuTerms(WGS84::ECEFCoord coords) {
    WGS84::GeodeticCoord converted = WGS84::ecefToGeodetic(coords);
    return ENUMatrixTerms(
        converted.lat * deg_to_rad_multipler,
        converted.lon * deg_to_rad_multipler
    );
}

/*
    A centering target coordinate can optionally be provided, but it will be
    provided in llh and the coordinate grid needs to know its place in ENU
    with the ECEF coordinate provided as the reference. This defaults us to the
    center if it isn't there.
*/
WGS84::ENUCoord CoordinateGridManager::optionallyResolveTargetCoordinates(
    WGS84::ECEFCoord referencePoint,
    const ENUMatrixTerms& matTerms,
    std::optional<WGS84::GeodeticCoord> targetPoint
) {
    if (targetPoint.has_value()) {
        WGS84::ECEFCoord targEcef = WGS84::geodeticToEcef(targetPoint.value());
        return WGS84::ecefToEnu(targEcef, matTerms, referencePoint);
    }
    return WGS84::ENUCoord{ 0.0, 0.0, 0.0 };
}

CoordinateGridManager::CoordinateGridManager(
    int xSize,
    int ySize,
    double spacing,
    WGS84::ECEFCoord referencePoint,
    std::optional<WGS84::GeodeticCoord> targetPoint
):
    xSize_(xSize),
    ySize_(ySize),
    spacing_(spacing),
    referencePoint_(referencePoint),
    terms_(computeEnuTerms(referencePoint)),
    targetEnu_(optionallyResolveTargetCoordinates(referencePoint, terms_, targetPoint))
{}

CoordinateGridManager::~CoordinateGridManager() {
    if (d_grid_ != nullptr) {
        cudaFree(d_grid_);
    }
}

void CoordinateGridManager::createGrid() {
    if (d_grid_ != nullptr) {
        cudaFree(d_grid_);
    }
    cudaMalloc(&d_grid_, numPoints() * 3 * sizeof(double));
    setCoordinateReferences(
        terms_,
        referencePoint_,
        targetEnu_,
        ENUSpacing{ spacing_, spacing_, 0.0 }
    );
    launchGenECEFCoordGrid(d_grid_, xSize_, ySize_);
}

std::vector<double> CoordinateGridManager::gridToHost() const {
    size_t n = numPoints() * 3;
    std::vector<double> host(n, 0.0);
    if (d_grid_ != nullptr) {
        cudaMemcpy(host.data(), d_grid_, n * sizeof(double), cudaMemcpyDeviceToHost);
    }
    return host;
}
