from collections.abc import Callable
import functools
import operator
import os
import subprocess
import triton
from triton.runtime import _allocation
from triton.backends.compiler import GPUTarget
from triton.backends.driver import GPUDriver, platform_key
from ._C import cuda_utils

from triton.tools.tensor_descriptor import TensorDescriptor

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
        self.fill_tma_descriptor = cuda_utils.fill_tma_descriptor


# ------------------------
# Launcher
# ------------------------


def ty_to_cpp(ty):
    if ty[0] == '*' or ty == "none":
        return "CUdeviceptr"
    if ty.startswith("tensordesc"):
        return "CUtensorMap"
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


def make_launcher(constants : dict[int, str], signature : dict[int, any]) -> Callable[..., None]:

    # Here, signature can look like:
    #  {'_0': 'i32',
    #   'Ptrs': (),
    #   '_1': 'constexpr',
    #   'values': '[*f32, constexpr]',
    #   'out_tuple': 'constexpr'}
    # We want to remove the constexprs, flatten the tuples, and remove any more
    # constexprs. If we remove them all at the end, we won't be able to remove
    # entire tuples that are a single constexpr. If we remove them before
    # flattening, we will miss mixed-tuples. So we do it twice.

    def _serialize_signature(sig):
        if isinstance(sig, tuple):
            return ','.join(map(_serialize_signature, sig))
        return sig

    # Remember & remove all the constexpr before flattening.
    constant_indices_before_flattening = {i for i, [k, v] in enumerate(signature.items()) if v == 'constexpr'}
    # constant_indices_before_flattening = [2, 4]
    signature = {k: v for k, v in signature.items() if v != 'constexpr'}
    # signature = {'_0': 'i32', 'Ptrs': (), 'values': '[*f32, constexpr]'}

    # Flatten.
    signature = ','.join(map(_serialize_signature, signature.values()))
    # signature = 'i32,,*f32,constexpr'
    signature = list(filter(bool, signature.split(',')))
    # signature = ['i32', '*f32', 'constexpr']

    # Remove any constexprs after flattening.
    constant_indices_after_flattening = {i for i, s in enumerate(signature) if s == 'constexpr'}
    # constant_indices_after_flattening = [2]
    signature = {i: s for i, s in enumerate(signature) if s != 'constexpr'}
    # signature = {0: 'i32', 1: '*f32'}

    signature_metadata = cuda_utils.build_signature_metadata(
            ty for ty in signature.values())

    def wrapper(grid_dim_x: int, grid_dim_y: int, grid_dim_z: int,
                stream: int, kernel: int, global_scratch: any,
                packed_metadata: tuple[int, int, int, int, int, int],
                hook_args: any,
                launch_enter_hook: Callable[..., None],
                launch_exit_hook: Callable[..., None],
                *args: any) -> None:
        # Given the example above, args would look something like:
        # args = [8, (), 5, (3, 4), (2, 2, 2)]
        # constant_indices_before_flattening = [2, 4]
        # Remove constantexprs before flattening:
        non_const_args = [arg
            for idx, arg in enumerate(args)
            if idx not in constant_indices_before_flattening
        ]
        # non_const_args = [8, (), (3, 4)]
        non_const_args = flatten_tuples(non_const_args)
        # non_const_args = [8, 3, 4]
        # constant_indices_after_flattening = [2]
        non_const_args = [arg
            for idx, arg in enumerate(non_const_args)
            if idx not in constant_indices_after_flattening
        ]
        # non_const_args = [8, 3]
        cuda_utils.launch(grid_dim_x, grid_dim_y, grid_dim_z, stream, kernel,
                          packed_metadata, hook_args, launch_enter_hook,
                          launch_exit_hook, signature_metadata, global_scratch,
                          non_const_args)
    return wrapper


class TmaDescKernelParam:
    TMA_DESC_SIZE = 128

    def __init__(self):
        import torch
        self.desc = torch.empty(self.TMA_DESC_SIZE, dtype=torch.uint8, device="cpu")

    # Return a CUtensorMap* pointer in host memory
    def tma_desc_cpu_ptr(self):
        return self.desc.data_ptr()


# The TMA dtype enum values are slightly different on host vs device...
TMA_DTYPE_DEVICE_TO_HOST = dict((i, i) for i in range(16))
TMA_DTYPE_DEVICE_TO_HOST[8] = 10
TMA_DTYPE_DEVICE_TO_HOST[9] = 8
TMA_DTYPE_DEVICE_TO_HOST[10] = 9


def make_tensordesc_arg(arg, metadata):
    assert isinstance(arg, TensorDescriptor)
    swizzle = metadata["swizzle"]
    elem_size = metadata["elem_size"]
    elem_type = metadata["elem_type"]
    block_size = metadata["block_size"]
    fp4_padded = metadata["fp4_padded"]

    data_ptr = arg.base.data_ptr()
    shape = arg.shape
    strides = arg.strides
    assert strides[-1] == 1

    desc = TmaDescKernelParam()
    result = [desc, *shape, *strides]

    if fp4_padded:
        shape[-1] *= 2
    triton.runtime.driver.active.utils.fill_tma_descriptor(
        desc.tma_desc_cpu_ptr(),
        data_ptr,
        swizzle,
        elem_size,
        TMA_DTYPE_DEVICE_TO_HOST[elem_type],
        block_size,
        shape,
        strides,
    )
    return result


def wrap_handle_tensordesc(launcher, tensordesc_meta):
    if not tensordesc_meta:
        return launcher

    def inner(*args):
        meta_args = args[:11]
        raw_kernel_args = args[11:]
        tensordesc_idx = 0
        final_args = []
        for i, arg in enumerate(raw_kernel_args):
            if isinstance(arg, TensorDescriptor):
                meta = tensordesc_meta[tensordesc_idx]
                tensordesc_idx += 1
                final_args.extend(make_tensordesc_arg(arg, meta))
            else:
                final_args.append(arg)
        assert tensordesc_idx == len(tensordesc_meta)
        return launcher(*meta_args, *final_args)

    return inner


class CudaLauncher(object):

    def __init__(self, src, metadata):
        constants = src.constants if hasattr(src, "constants") else dict()
        arg_idx = lambda x: (src.fn.arg_names.index(x), ) if isinstance(x, str) else x
        constants = {arg_idx(idx): value for idx, value in constants.items()}
        signature = {idx: value for idx, value in src.signature.items()}
        self.num_ctas = functools.reduce(operator.mul, metadata.cluster_dims, 1)
        self.launch = make_launcher(constants, signature)
        self.global_scratch_size = metadata.global_scratch_size
        self.global_scratch_align = metadata.global_scratch_align
        self.launch_cooperative_grid = metadata.launch_cooperative_grid

    def __call__(self, gridX, gridY, gridZ, stream, function, *args):
        if self.global_scratch_size > 0:
            grid_size = gridX * gridY * gridZ
            alloc_size = grid_size * self.num_ctas * self.global_scratch_size
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
        try:
            import torch
            return torch.cuda.is_available() and (torch.version.hip is None)
        except ImportError:
            return True

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

    def clear_cache(self, cache):
        cache.zero_()
