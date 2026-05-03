#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <iostream>
#include "grid_manager.h"
#include "bp_manager.h"

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

    using complex64 = std::complex<float>;
    using srcpos_array = py::array_t<double, py::array::c_style | py::array::forcecast>;
    using memmap_array = py::array_t<float, py::array::c_style | py::array::forcecast>;

    py::class_<BPManager>(m, "BPManager")
        .def(
            py::init([](
                int xSize, int ySize,
                int pulseLimit, int rangeBinLen,
                double bandwidth, double fc,
                WGS84::ECEFCoord srpEcef,
                CoordinateGridManager& gridMgr,
                srcpos_array srcPos
            ) {
                auto srcPosBuf = srcPos.request();
                return new BPManager(
                    xSize, ySize, pulseLimit, rangeBinLen,
                    bandwidth, fc, srpEcef,
                    gridMgr,
                    static_cast<const double*>(srcPosBuf.ptr)
                );
            }),
            py::arg("x_size"), py::arg("y_size"),
            py::arg("pulse_limit"), py::arg("range_bin_len"),
            py::arg("bandwidth"), py::arg("fc"),
            py::arg("srp_ecef"),
            py::arg("grid_manager"),
            py::arg("src_pos"),
            
            // Keep grid_manager alive for the lifetime of BPManager
            py::keep_alive<0, 8>()
        )
        .def(
            "process_pulse",
            [](BPManager& self, int pulseIdx, memmap_array pulseData) {
                auto buf = pulseData.request();
                self.processPulse(pulseIdx, static_cast<const float*>(buf.ptr));
            },
            py::arg("pulse_idx"), py::arg("pulse_data"),
            "FFT and accumulate one pulse"
        )
        .def(
            "finalize_image",
            &BPManager::finalizeImage,
            "Apply the final center-pulse phase correction"
        )
        .def(
            "image_to_numpy",
            [](BPManager& self) {
                auto host = self.imageToHost();
                auto result = py::array_t<complex64>({self.xSize(), self.ySize()});
                std::copy(host.begin(), host.end(), result.mutable_data());
                return result;
            },
            "Copy the device image to a (x_size, y_size) complex64 numpy array"
        );
}
