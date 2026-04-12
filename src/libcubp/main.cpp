#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "kernels.h"
#include <iostream> 

namespace py = pybind11;

// Example binding: element-wise scale a float32 numpy array on the GPU.
py::array_t<float> scale(py::array_t<float, py::array::c_style | py::array::forcecast> arr,
                         float scalar) {
    py::buffer_info buf = arr.request();
    auto result = py::array_t<float>(buf.size);
    py::buffer_info res_buf = result.request();

    cuda_scale(static_cast<const float*>(buf.ptr),
               static_cast<float*>(res_buf.ptr),
               scalar,
               static_cast<size_t>(buf.size));

    result.resize(buf.shape);
    return result;
}

void test(py::array_t<float, py::array::c_style | py::array::forcecast> arr) {
    std::cout << "received!\n";
}

PYBIND11_MODULE(_libcubp, m) {
    m.doc() = "CuBP: cuda backprojection library";

    m.def("scale", &scale,
          "Element-wise multiply a float32 numpy array by a scalar on the GPU",
          py::arg("arr"), py::arg("scalar"));

    m.def("test", &test,
        "testing func", 
        py::arg("arr")
    );
}
