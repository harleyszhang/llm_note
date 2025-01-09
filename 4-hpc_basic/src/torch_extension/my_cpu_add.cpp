// my_extension.cpp

#include <torch/extension.h>

// 定义一个简单的加法操作（CPU 实现）
torch::Tensor add_cpu(torch::Tensor a, torch::Tensor b) {
    return a + b;
}

#ifdef WITH_CUDA
// CUDA 声明
torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b);
#endif

// 绑定操作到 Python
torch::Tensor add(torch::Tensor a, torch::Tensor b) {
    if (a.device().is_cuda()) {
#ifdef WITH_CUDA
        return add_cuda(a, b);
#else
        throw std::runtime_error("CUDA is not available");
#endif
    }
    return add_cpu(a, b);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add", &add, "Add two tensors");
}
