"""
Build system for Engram CUDA extension.

Usage:
    pip install -e .                    # install with CUDA extension
    ENGRAM_NO_CUDA=1 pip install -e .   # install without CUDA (PyTorch fallback)

    python setup.py build_ext --inplace  # build extension in-place
"""

import os
from setuptools import setup, find_packages

# Allow building without CUDA for development/CPU-only machines
NO_CUDA = os.environ.get("ENGRAM_NO_CUDA", "0") == "1"

ext_modules = []
cmdclass = {}

if not NO_CUDA:
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension

        ext_modules = [
            CUDAExtension(
                name="engram_cuda",
                sources=[
                    "csrc/engram_kernels.cu",
                    "csrc/engram_bindings.cpp",
                ],
                include_dirs=["csrc"],
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                    "nvcc": [
                        "-O3",
                        "-std=c++17",
                        "--use_fast_math",
                        "--expt-relaxed-constexpr",
                        "-gencode=arch=compute_80,code=sm_80",  # A100
                        "-gencode=arch=compute_89,code=sm_89",  # L40/4090
                        "-gencode=arch=compute_90,code=sm_90",  # H100
                    ],
                },
            )
        ]
        cmdclass = {"build_ext": BuildExtension}
    except ImportError:
        print("WARNING: torch.utils.cpp_extension not available, skipping CUDA build")

setup(
    name="engram",
    version="0.1.0",
    description="Engram: Conditional Memory via Scalable N-gram Lookup for LLMs",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "numpy>=1.24.0",
        "transformers>=4.37.0",
        "tokenizers>=0.15.0",
        "sympy>=1.12",
    ],
)
