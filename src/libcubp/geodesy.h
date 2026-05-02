#pragma once
#include <cmath>
#include <optional>

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

// ENUSpacing contains the offset per pixel in meters in each direction.
struct ENUSpacing {
    double d_e;
    double d_n;
    double d_u;
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
    // First Eccentricity squared
    constexpr double E2 = 6.694379990141e-3;

    constexpr double A2 = A * A;
    constexpr double E4 = E2 * E2;

    struct GeodeticCoord {
        double lat, lon, alt;
    };

    struct ECEFCoord {
        double x, y, z;
    };

    struct ENUCoord {
        double e, n, u;
    };

    ECEFCoord geodeticToEcef(GeodeticCoord coord);
    GeodeticCoord ecefToGeodetic(ECEFCoord coord);
    ENUCoord ecefToEnu(ECEFCoord coord, const ENUMatrixTerms& matTerms, ECEFCoord referenceCoord);
}
