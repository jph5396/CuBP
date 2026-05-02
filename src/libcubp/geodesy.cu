#include <cmath>
#include "geodesy.h"

/*
    Based on the algorithm defined in Vermille (2002). Access here:
    https://link.springer.com/article/10.1007/s00190-002-0273-6
*/
WGS84::GeodeticCoord WGS84::ecefToGeodetic(WGS84::ECEFCoord coord) {

    double x_y_euclid = std::sqrt(std::pow(coord.x, 2.0) + std::pow(coord.y, 2.0));

    double p = (std::pow(coord.x, 2.0) + std::pow(coord.y, 2.0)) / WGS84::A2;
    double q = ((1.0 - WGS84::E2) / WGS84::A2) * std::pow(coord.z, 2.0);
    double r = (p + q - WGS84::E4) / 6.0;
    double s = WGS84::E4 * ((p * q) / (4.0 * std::pow(r, 3.0)));
    double t = std::cbrt(1.0 + s + std::sqrt(s * (2.0 + s)));
    double u = r * (1.0 + t + (1.0 / t));
    double v = std::sqrt(std::pow(u, 2.0) + (WGS84::E4 * q));
    double w = WGS84::E2 * ((u + v - q) / (2.0 * v));
    double k = std::sqrt(u + v + std::pow(w, 2.0)) - w;
    double D = (k * x_y_euclid) / (k + WGS84::E2);

    double lon = 2.0 * std::atan2(coord.y, coord.x + x_y_euclid);
    double lat = 2.0 * std::atan2(coord.z, D + std::sqrt(D * D + coord.z * coord.z));
    double height = ((k + WGS84::E2 - 1.0) / k) * std::sqrt(D * D + coord.z * coord.z);

    return WGS84::GeodeticCoord{ lat, lon, height };
}

WGS84::ECEFCoord WGS84::geodeticToEcef(WGS84::GeodeticCoord coord) {

    double cla_clo = std::cos(coord.lat * deg_to_rad_multipler) * std::cos(coord.lon * deg_to_rad_multipler);
    double cla_slo = std::cos(coord.lat * deg_to_rad_multipler) * std::sin(coord.lon * deg_to_rad_multipler);
    double sla = std::sin(coord.lat * deg_to_rad_multipler);

    double r = WGS84::A / (std::sqrt(1.0 - WGS84::E2 * sla * sla));

    return WGS84::ECEFCoord{
        (r + coord.alt) * cla_clo,
        (r + coord.alt) * cla_slo,
        (r + coord.alt - WGS84::E2 * r) * sla
    };
}

WGS84::ENUCoord WGS84::ecefToEnu(
    WGS84::ECEFCoord coord,
    const ENUMatrixTerms& matTerms,
    WGS84::ECEFCoord referenceCoord
) {
    double d_x = coord.x - referenceCoord.x;
    double d_y = coord.y - referenceCoord.y;
    double d_z = coord.z - referenceCoord.z;

    return WGS84::ENUCoord{
        -matTerms.slo * d_x + matTerms.clo * d_y,
        -matTerms.sla_clo * d_x - matTerms.sla_slo * d_y + matTerms.cla * d_z,
        matTerms.cla_clo * d_x + matTerms.cla_slo * d_y + matTerms.sla * d_z
    };
}
