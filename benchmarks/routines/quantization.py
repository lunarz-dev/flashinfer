from collections import defaultdict

import numpy as np
import torch

import flashinfer
from flashinfer import SfLayout
from flashinfer.testing.utils import bench_gpu_time

from .flashinfer_benchmark_utils import (
    dtype_str_to_torch_dtype,
    filter_backends_by_compute_capability,
    get_device,
    print_perf_metrics,
)


def run_quantization_test(args):
    if args.routine == "nvfp4_quantize":
        return testNvfp4Quantize(args)
    raise ValueError(f"Unsupported routine: {args.routine}")


def parse_quantization_args(line, parser):
    parser.add_argument("--m", type=int, required=True, help="Number of input rows.")
    parser.add_argument(
        "--k",
        type=int,
        required=True,
        help="Number of input columns; must be divisible by sf_vec_size.",
    )
    parser.add_argument(
        "--input_dtype",
        type=str,
        required=False,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Input tensor dtype.",
    )
    parser.add_argument(
        "--global_scale",
        type=float,
        required=False,
        default=1.0,
        help="Global scale factor for nvfp4_quantize.",
    )
    parser.add_argument(
        "--sf_layout",
        type=str,
        required=False,
        default="128x4",
        choices=["128x4", "8x4"],
        help="Scale factor layout.",
    )
    parser.add_argument(
        "--do_shuffle",
        action="store_true",
        default=False,
        help="Enable scale-factor shuffle.",
    )
    parser.add_argument(
        "--sf_vec_size",
        type=int,
        required=False,
        default=16,
        help="Scale factor vector size.",
    )
    parser.add_argument(
        "--backends",
        type=str,
        required=False,
        nargs="+",
        default=["cuda"],
        choices=["cuda"],
        help="Backend to test.",
    )
    args = parser.parse_args(line)
    if args.verbose >= 1:
        print(f"[INFO] {args = }")
    return args


def testNvfp4Quantize(args):
    if args.verbose >= 1:
        print("[INFO] Running testNvfp4Quantize")
        print(f"[INFO] FlashInfer version: {flashinfer.__version__}")

    device = get_device(args)
    if args.generate_repro_command:
        print(
            f"[INFO] To reproduce this test case, run the following command: {args.repro_command}"
        )

    backends = args.backends[:]
    m = args.m
    k = args.k
    global_scale = args.global_scale
    sf_layout_str = args.sf_layout
    do_shuffle = args.do_shuffle
    sf_vec_size = args.sf_vec_size
    is_cuda_graph_compatible = not args.no_cuda_graph
    res = []

    if k % sf_vec_size != 0:
        raise ValueError(
            f"k ({k}) must be divisible by sf_vec_size ({sf_vec_size}) for nvfp4_quantize"
        )
    if do_shuffle and is_cuda_graph_compatible:
        print(
            "[WARNING] do_shuffle=True is not CUDA graph compatible. Disabling CUDA graph."
        )
        is_cuda_graph_compatible = False

    sf_layout = {
        "128x4": SfLayout.layout_128x4,
        "8x4": SfLayout.layout_8x4,
    }[sf_layout_str]

    backends = filter_backends_by_compute_capability(backends, args.routine, device)
    if len(backends) == 0:
        print("[ERROR] No backends to test. Exiting.")
        return res

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    input_tensor = torch.randn((m, k), dtype=input_dtype, device=device)
    global_sf_tensor = torch.tensor(global_scale, dtype=torch.float32, device=device)

    if args.verbose >= 2:
        print(f"[VVERBOSE] {input_tensor.shape = }")
        print(f"[VVERBOSE] {input_tensor.dtype = }")
        print(f"[VVERBOSE] {global_scale = }")
        print(f"[VVERBOSE] {sf_layout_str = }")
        print(f"[VVERBOSE] {do_shuffle = }")
        print(f"[VVERBOSE] {sf_vec_size = }")

    def run_backend(backend):
        return flashinfer.nvfp4_quantize(
            input_tensor,
            global_sf_tensor,
            sfLayout=sf_layout,
            do_shuffle=do_shuffle,
            sf_vec_size=sf_vec_size,
        )

    backend_times = {}
    for backend in backends:
        if backend != "cuda":
            raise ValueError(f"Unsupported nvfp4_quantize backend: {backend}")
        backend_times[backend] = bench_gpu_time(
            fn=lambda: run_backend(backend),
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            l2_flush=True,
            l2_flush_size_mb=256,
            l2_flush_device=device,
            sleep_after_run=True,
            enable_cupti=args.use_cupti,
            use_cuda_graph=is_cuda_graph_compatible,
        )

    for backend in backends:
        if len(backend_times[backend]) > 0:
            median_time = np.median(backend_times[backend])
            std_time = np.std(backend_times[backend])
            num_elements = m * k
            num_scale_factors = num_elements // sf_vec_size
            problem_bytes = (
                num_elements * input_dtype.itemsize
                + 4
                + num_elements // 2
                + num_scale_factors
            )
            problem_flops = num_elements * 3
            tflops = problem_flops / (10**9 * median_time)
            tb_per_sec = problem_bytes / (10**9 * median_time)
            print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec)

            cur_res = defaultdict(str)
            cur_res["routine"] = args.routine
            cur_res["median_time"] = median_time
            cur_res["std_time"] = std_time
            cur_res["tflops"] = tflops
            cur_res["tb_per_sec"] = tb_per_sec
            cur_res["backend"] = backend
            cur_res["m"] = m
            cur_res["k"] = k
            cur_res["input_dtype"] = str(input_dtype)
            cur_res["global_scale"] = global_scale
            cur_res["sf_layout"] = sf_layout_str
            cur_res["do_shuffle"] = do_shuffle
            cur_res["sf_vec_size"] = sf_vec_size
            cur_res["case_tag"] = args.case_tag
            res.append(cur_res)
    return res
