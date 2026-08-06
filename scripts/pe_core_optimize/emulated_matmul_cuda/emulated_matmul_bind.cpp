#include <torch/extension.h>

torch::Tensor emulated_matmul_cuda(torch::Tensor Xh, torch::Tensor Xl, torch::Tensor Wh, torch::Tensor Wl);

torch::Tensor emulated_matmul(torch::Tensor Xh, torch::Tensor Xl, torch::Tensor Wh, torch::Tensor Wl) {
    TORCH_CHECK(Xh.is_cuda() && Xl.is_cuda() && Wh.is_cuda() && Wl.is_cuda(), "all inputs must be CUDA tensors");
    TORCH_CHECK(Xh.dtype() == torch::kHalf && Xl.dtype() == torch::kHalf &&
                Wh.dtype() == torch::kHalf && Wl.dtype() == torch::kHalf,
                "all inputs must be fp16");
    TORCH_CHECK(Xh.size(0) % 16 == 0 && Xh.size(1) % 16 == 0, "M and K must be multiples of 16");
    TORCH_CHECK(Wh.size(1) % 16 == 0, "N must be a multiple of 16");
    TORCH_CHECK(Xh.size(1) == Wh.size(0), "K mismatch between X and W");
    return emulated_matmul_cuda(Xh.contiguous(), Xl.contiguous(), Wh.contiguous(), Wl.contiguous());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("emulated_matmul", &emulated_matmul, "Emulated FP32 matmul via WMMA (CUDA)");
}
