#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include "coordinate_conversions.h"

namespace py = pybind11;

// Example binding: element-wise scale a float32 numpy array on the GPU.
py::array_t<float> scale(py::array_t<float, py::array::c_style | py::array::forcecast> arr,
                         float scalar) {
    py::buffer_info buf = arr.request();
    auto result = py::array_t<float>(buf.size);
    py::buffer_info res_buf = result.request();
    return result;
}

void test(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
    std::cout << "received!\n";
}

PYBIND11_MODULE(_libcubp, m) {
    m.doc() = "CuBP: cuda backprojection library";

    //WGS84 structs

    py::class_<WGS84::GeodeticCoord>(m, "GeodeticCoord")
        .def(py::init<double, double, double>())
        .def_readwrite("lat", &WGS84::GeodeticCoord::lat)
        .def_readwrite("lon", &WGS84::GeodeticCoord::lon)
        .def_readwrite("alt", &WGS84::GeodeticCoord::alt);

    py::class_<WGS84::ECEFCoord>(m, "ECEFCoord")
        .def(py::init<double, double, double>())
        .def_readwrite("x", &WGS84::ECEFCoord::x)
        .def_readwrite("y", &WGS84::ECEFCoord::y)
        .def_readwrite("z", &WGS84::ECEFCoord::z);

    py::class_<ENUMatrixTerms>(m, "ENUMatrixTerms")
        .def(py::init<double, double>(), py::arg("lat_rad"), py::arg("lon_rad"))
        .def_readonly("slo", &ENUMatrixTerms::slo)
        .def_readonly("clo", &ENUMatrixTerms::clo)
        .def_readonly("sla", &ENUMatrixTerms::sla)
        .def_readonly("cla", &ENUMatrixTerms::cla)
        .def_readonly("sla_clo", &ENUMatrixTerms::sla_clo)
        .def_readonly("sla_slo", &ENUMatrixTerms::sla_slo)
        .def_readonly("cla_clo", &ENUMatrixTerms::cla_clo)
        .def_readonly("cla_slo", &ENUMatrixTerms::cla_slo);

    py::class_<WGS84::ENUCoord>(m, "ENUCoord")
        .def(py::init<double, double, double>())
        .def_readwrite("e", &WGS84::ENUCoord::e)
        .def_readwrite("n", &WGS84::ENUCoord::n)
        .def_readwrite("u", &WGS84::ENUCoord::u);
    
    //WGS84 functions 
    m.def(
        "ecef_to_geodetic", 
        &WGS84::ecefToGeodetic, 
        "converts an ECEF coordinate to a geodetic coordinate"
    ); 
    
    m.def(
        "geodetic_to_ecef", 
        &WGS84::geodeticToEcef, 
        "converts a geodetic coordinate to a ECEF coordinate"
    );

    m.def(
        "ecef_to_enu",
        &WGS84::ecefToEnu,
        "Converts an ecef coordinate to its enu coordinate"
    );

    py::class_<CoordinateGridManager>(m, "CoordinateGridManager")
        .def(
            py::init<int, int, double, WGS84::ECEFCoord, std::optional<WGS84::GeodeticCoord>>(),
            py::arg("x_size"),
            py::arg("y_size"),
            py::arg("spacing"),
            py::arg("reference_point"),
            py::arg("target_point") = std::nullopt
        )
        .def(
            "create_grid",
            &CoordinateGridManager::createGrid,
            "Allocate device memory and run the GPU kernel to build the ECEF coordinate grid"
        )
        .def(
            "grid_to_numpy",
            [](CoordinateGridManager& self) {
                auto host = self.gridToHost();
                auto result = py::array_t<double>({self.numPoints(), 3});
                std::copy(host.begin(), host.end(), result.mutable_data());
                return result;
            },
            "Copy the device grid to a (x_size*y_size, 3) numpy array (primarily for testing)"
        );
}
