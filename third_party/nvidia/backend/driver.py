from collections.abc import Callable
import functools
import os
import subprocess
from triton.runtime import _allocation
from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver
from triton._utils import parse_list_string
from ._C import cuda_utils

dirname = os.path.dirname(os.path.realpath(__file__))
include_dir = [os.path.join(dirname, "include")]
libdevice_dir = os.path.join(dirname, "lib")


@functools.lru_cache()
def libcuda_dirs():
    env_libcuda_path = os.getenv("TRITON_LIBCUDA_PATH")
    if env_libcuda_path:
        return [env_libcuda_path]

    libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode()
    # each line looks like the following:
    # libcuda.so.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcuda.so.1
    locs = [line.split()[-1] for line in libs.splitlines() if "libcuda.so.1" in line]
    dirs = [os.path.dirname(loc) for loc in locs]
    env_ld_library_path = os.getenv("LD_LIBRARY_PATH")
    if env_ld_library_path and not dirs:
        dirs = [dir for dir in env_ld_library_path.split(":") if os.path.exists(os.path.join(dir, "libcuda.so.1"))]
    msg = 'libcuda.so cannot found!\n'
    if locs:
        msg += 'Possible files are located at %s.' % str(locs)
        msg += 'Please create a symlink of libcuda.so to any of the files.'
    else:
        msg += 'Please make sure GPU is set up and then run "/sbin/ldconfig"'
        msg += ' (requires sudo) to refresh the linker cache.'
    assert any(os.path.exists(os.path.join(path, 'libcuda.so.1')) for path in dirs), msg
    return dirs


@functools.lru_cache()
def library_dirs():
    return [libdevice_dir, *libcuda_dirs()]


# ------------------------
# Utils
# ------------------------


class CudaUtils(object):

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(CudaUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.load_binary = cuda_utils.load_binary
        self.get_device_properties = cuda_utils.get_device_properties
        self.cuOccupancyMaxActiveClusters = cuda_utils.cuOccupancyMaxActiveClusters
        self.set_printf_fifo_size = cuda_utils.set_printf_fifo_size
        self.fill_1d_tma_descriptor = cuda_utils.fill_1d_tma_descriptor
        self.fill_2d_tma_descriptor = cuda_utils.fill_2d_tma_descriptor


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty):
    if ty[0] == '*' or ty == "none":
        return "CUdeviceptr"
    return {
        "i1": "int32_t",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
        "u1": "uint32_t",
        "u8": "uint8_t",
        "u16": "uint16_t",
        "u32": "uint32_t",
        "u64": "uint64_t",
        "fp16": "float",
        "bf16": "float",
        "fp32": "float",
        "f32": "float",
        "fp64": "double",
        "nvTmaDesc": "CUtensorMap",
    }[ty]


def flatten_tuples(xs):
    """Recursively flattens tuple elements in xs."""
    for x in xs:
        if isinstance(x, tuple):
            yield from flatten_tuples(x)
        else:
            yield x


def make_launcher(constants : dict[int, str], signature : dict[int, any], ids : dict[str, tuple]) -> Callable[..., None]:

    signature = {k: v for k, v in signature.items() if v != 'constexpr'}
    signature = ','.join(signature.values()).replace('[', '').replace(']', '')
    signature = list(filter(bool, signature.split(',')))
    signature = {i: s for i, s in enumerate(signature)}

    # We seem to have 3  categories of arguments:
    # 1. arguments listed in signature
    # 2. arguments listed in constants
    # 3. those present in both signature and constants
    # TODO(gflegar): why is that?
    # Category (2) does not get passed to the launcher, but category (3) does.
    # However, the launcher is supposed to ignore the arguments passed from
    # category (3). The generic C++ launcher currently does not do that, so we
    # are doing it in the python wrapper.
    signature_metadata = cuda_utils.build_signature_metadata(
            ty for ty in signature.values())

    def wrapper(grid_dim_x: int, grid_dim_y: int, grid_dim_z: int,
                stream: int, kernel: int, global_scratch: any,
                packed_metadata: tuple[int, int, int, int, int, int],
                hook_args: any,
                launch_enter_hook: Callable[..., None],
                launch_exit_hook: Callable[..., None],
                *args: any) -> None:
        cuda_utils.launch(grid_dim_x, grid_dim_y, grid_dim_z, stream, kernel,
                          packed_metadata, hook_args, launch_enter_hook,
                          launch_exit_hook, signature_metadata, global_scratch,
                          flatten_tuples(args))
    return wrapper


class CudaLauncher(object):

    def __init__(self, src, metadata):
        ids = {"ids_of_const_exprs": src.fn.constexprs if hasattr(src, "fn") else tuple()}
        constants = src.constants if hasattr(src, "constants") else dict()
        constants = {idx: value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}
        self.launch = make_launcher(constants, signature, ids)
        self.global_scratch_size = metadata.global_scratch_size
        self.global_scratch_align = metadata.global_scratch_align

    def __call__(self, gridX, gridY, gridZ, stream, function, *args):
        if self.global_scratch_size > 0:
            grid_size = gridX * gridY * gridZ
            alloc_size = grid_size * self.global_scratch_size
            global_scratch = _allocation._allocator(alloc_size, self.global_scratch_align, stream)
        else:
            global_scratch = None
        self.launch(gridX, gridY, gridZ, stream, function, global_scratch, *args)


class CudaDriver(GPUDriver):

    def __init__(self):
        self.utils = CudaUtils()  # TODO: make static
        self.launcher_cls = CudaLauncher
        super().__init__()

    def get_current_target(self):
        device = self.get_current_device()
        capability = self.get_device_capability(device)
        capability = capability[0] * 10 + capability[1]
        warp_size = 32
        return GPUTarget("cuda", capability, warp_size)

    def get_active_torch_device(self):
        import torch
        return torch.device("cuda", self.get_current_device())

    def get_device_interface(self):
        import torch
        return torch.cuda

    @staticmethod
    def is_active():
        return True
        import torch
        return torch.cuda.is_available() and (torch.version.hip is None)

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_empty_cache_for_benchmark(self):
        import torch

        # We maintain a buffer of 256 MB that we clear
        # before each kernel call to make sure that the L2 cache
        # doesn't contain any input data before the run
        cache_size = 256 * 1024 * 1024
        return torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')
