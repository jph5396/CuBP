#pragma once
#include <cmath>
#include <optional>
#include <vector>


const double deg_to_rad_multipler = M_PI / 180.0;

struct ENUMatrixTerms {
    double slo;     // sin(lon)
    double clo;     // cos(lon)
    double sla;     // sin(lat)
    double cla;     // cos(lat)
    double sla_clo; // sin(lat)*cos(lon)
    double sla_slo; // sin(lat)*sin(lon)
    double cla_clo; // cos(lat)*cos(lon)
    double cla_slo; // cos(lat)*sin(lon)

    ENUMatrixTerms(double lat_rad, double lon_rad) {
        sla = std::sin(lat_rad); 
        cla = std::cos(lat_rad); 
        slo = std::sin(lon_rad); 
        clo = std::cos(lon_rad); 

        sla_clo = sla * clo; 
        sla_slo = sla * slo;
        cla_clo = cla * clo; 
        cla_slo = cla * slo; 
    };
};

// Anything related to WGS 1984 and coordinate system conversions around it
// should be placed in this namespace. 
// 
// See section 3 in the standards document linked here: 
// https://nsgreg.nga.mil/doc/view?i=4597
namespace WGS84 {
    // semi-major axis
    constexpr double A = 6378137.0;
    // semi-minor axis
    constexpr double B = 6356752.3142;
    // Flattening Factor of the Earth (1/f)
    constexpr double F = 1.0 / 298.257223563; 
    // First Eccentriricty squared
    constexpr double E2 = 6.694379990141e-3;

    //These are some constants not directly from the derived document, 
    // but are commonly used in some coordinate conversions so we keep them here 
    constexpr double A2 = A * A;
    
    constexpr double E4 = E2 * E2;

    // coordinate structs 
    //GeodeticCoord
    struct GeodeticCoord {
        double lat, lon, alt; 
    };

    // ECEF
    struct ECEFCoord{
        double x, y, z;
    };

    //ENU 
    struct ENUCoord {
        double e, n, u; 
    };

    // Takes in a geodetic coordinate and converts it to an ECEF coordinate
    ECEFCoord geodeticToEcef(GeodeticCoord coord);

    // Takes in a ECEF coordinate and converts it to a geodetic coordinate
    GeodeticCoord ecefToGeodetic(ECEFCoord coord);

    // this assumes a version of the enu conversion that has a precalculated 
    // ENU Matrix. 
    ENUCoord ecefToEnu(ECEFCoord coord, const ENUMatrixTerms& matTerms, ECEFCoord referenceCoord);
}

/*
    this struct generates the terms needed for an ENU matrix. The terms are 
    reusable for conversions to and from, they would just need to be transposed. 
*/ 



/*
    ENUSpacing contains the offset per pixel in meters for 
    request in each direction.
*/
struct ENUSpacing {
    double d_e;
    double d_n; 
    double d_u; 
};

class CoordinateGridManager {
public: 
    CoordinateGridManager(
        int xSize, 
        int ySize, 
        double spacing, 
        WGS84::ECEFCoord referncePoint, 
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