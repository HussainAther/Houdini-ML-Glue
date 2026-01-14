import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

# We use the helper from torch to find the correct include and library paths
# This ensures it works on your MacBook Air (M1/M2/Intel)
setup(
    name='deepvex_bindings',
    ext_modules=[
        CppExtension(
            name='deepvex_bindings',
            sources=[
                'src/cpp_bindings/core_interface.cpp'
            ],
            include_dirs=[
                # Points to your header files
                os.path.abspath('src/cpp_bindings')
            ],
            extra_compile_args=['-std=c++17', '-O3']
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
