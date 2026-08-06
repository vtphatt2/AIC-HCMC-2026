from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="emulated_matmul_cuda",
    ext_modules=[
        CUDAExtension(
            name="emulated_matmul_cuda",
            sources=["emulated_matmul_bind.cpp", "emulated_matmul_kernel.cu"],
            extra_compile_args={
                "cxx": [],
                # T4 == Turing == sm_75. Locking this avoids nvcc guessing/multi-arch bloat.
                "nvcc": ["-gencode=arch=compute_75,code=sm_75"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
