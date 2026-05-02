#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
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
}
