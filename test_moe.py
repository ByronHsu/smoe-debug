from transformers.models.mixtral import MixtralConfig
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
import torch
from smoe_debug.moe import GLUMLP, SMoeBlock
import random
import os
import pytest 
import triton
from typing import Callable

def get_current_file_directory() -> str:
    """
    Returns the directory path of the current Python file.
    """
    # Get the absolute path of the current file
    current_file_path = os.path.abspath(__file__)

    # Get the directory path of the current file
    return os.path.dirname(current_file_path)


def _test_memory_once(func: Callable) -> float:

    torch.cuda.memory._record_memory_history()
    torch.cuda.memory.reset_peak_memory_stats()

    func()

    mem = torch.cuda.max_memory_allocated()

    # uncomment to save the visual memory snapshot
    # torch.cuda.memory._dump_snapshot(f"{func.__name__}.pickle")

    torch.cuda.memory._record_memory_history(enabled=None)
    return mem


def _test_memory(func: Callable, _iter: int = 100) -> float:
    total_mem = []

    for _ in range(_iter):
        mem = _test_memory_once(func)
        total_mem.append(mem)

    return sum(total_mem) / len(total_mem)

def set_seed(seed=42):
    """
    Fix all random seeds we use for reproducibility.
    """
    # Python random seed
    random.seed(seed)

    # PyTorch random seed
    torch.manual_seed(seed)

    # If you are using CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # PyTorch backend settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

set_seed()


def assert_verbose_allclose(tensor1, tensor2, rtol=1e-05, atol=1e-08, max_print=5):
    """
    Assert that two tensors are element-wise equal within a tolerance, providing detailed information about mismatches.

    Parameters:
    tensor1 (torch.Tensor): First tensor to compare.
    tensor2 (torch.Tensor): Second tensor to compare.
    rtol (float): Relative tolerance.
    atol (float): Absolute tolerance.
    max_print (int): Maximum number of mismatched elements to print.

    Raises:
    AssertionError: If the tensors are not all close within the given tolerance.
    """
    # Check if the shapes of the tensors match
    if tensor1.shape != tensor2.shape:
        raise AssertionError("Input tensors must have the same shape.")

    # Calculate the difference between the tensors
    diff = torch.abs(tensor1 - tensor2)

    # Determine the tolerance
    tolerance = atol + rtol * torch.abs(tensor2)

    # Find mismatched elements
    mismatched = diff > tolerance

    # Get the indices of mismatched elements
    mismatched_indices = torch.nonzero(mismatched)

    # Count the number of mismatched elements
    num_mismatched = mismatched.sum().item()

    # Check if all elements are close
    all_close = num_mismatched == 0

    # Raise AssertionError with detailed information if there are mismatches
    if not all_close and num_mismatched > 1:
        mismatch_details = [f"Number of mismatched elements: {num_mismatched}"]
        print_count = min(max_print, num_mismatched)
        for index in mismatched_indices[:print_count]:
            i = tuple(index.tolist())
            mismatch_details.append(
                f"Mismatch at index {i}: tensor1[{i}] = {tensor1[i]}, tensor2[{i}] = {tensor2[i]}"
            )
        if num_mismatched > max_print:
            mismatch_details.append(
                f"... and {num_mismatched - max_print} more mismatched elements."
            )

        raise AssertionError("\n".join(mismatch_details))


@pytest.mark.parametrize(
    "bsz, seq_len, hidden_size, intermediate_size, num_local_experts, num_experts_per_tok",
    [
        (4, 16, 256, 512, 8, 2),
        # (4, 1024, 4096, 14436, 8, 2)
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-2, 1e-5),
    ],
)
def test_correctness(
    bsz, seq_len, hidden_size, intermediate_size, num_local_experts, num_experts_per_tok, dtype, atol, rtol
):
    config = MixtralConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=num_local_experts,
        num_experts_per_tok=num_experts_per_tok,
    )

    moe_block = MixtralSparseMoeBlock(config).to("cuda").to(dtype)

    # init moe block weights as randn for gate and experts
    # moe_block.gate.weight.data = torch.randn_like(moe_block.gate.weight.data)
    # for expert in moe_block.experts:
    #     expert.w1.weight.data = torch.randn_like(expert.w1.weight.data)
    #     expert.w2.weight.data = torch.randn_like(expert.w2.weight.data)
    #     expert.w3.weight.data = torch.randn_like(expert.w3.weight.data)


    smoe_block = SMoeBlock(config).to("cuda").to(dtype)

    smoe_block.copy_weights_from_hf(moe_block)

    assert_verbose_allclose(moe_block.gate.weight.data, smoe_block.gate.weight.data)
    
    _tensor = torch.randn(bsz, seq_len, hidden_size, device="cuda", dtype=dtype)
    x1 = _tensor.clone().requires_grad_(True)
    x2 = _tensor.clone().requires_grad_(True)
    y1, r1 = moe_block(x1)
    y2, r2 = smoe_block(x2)
    assert_verbose_allclose(r1, r2, atol=atol, rtol=rtol)
    assert_verbose_allclose(y1, y2, atol=atol, rtol=rtol)

    # backward
    (x1_grad,)  = torch.autograd.grad(y1, x1, torch.ones_like(y1), retain_graph=True)
    (x2_grad,)  = torch.autograd.grad(y2, x2, torch.ones_like(y2), retain_graph=True)
    # import pdb; pdb.set_trace()
    assert_verbose_allclose(x1_grad, x2_grad, atol=atol, rtol=rtol)



@triton.testing.perf_report(
    [
    triton.testing.Benchmark(
        x_names=["bsz"],
        x_vals=[i for i in range(4, 8, 2)],
        line_arg="provider",
        line_vals=["smoe", "torch"],
        line_names=["Smoe", "PyTorch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="time (ms)",
        plot_name=f"moe-full-fp16-speed-benchmark",
        # args={"dtype": torch.bfloat16, "mode": mode, "seq_len": 2048},
        args={"seq_len": 2048, "hidden_size": 4096, "intermediate_size": 14336, "num_local_experts": 8, "num_experts_per_tok": 2, "dtype": torch.float16, "mode": "full"},
    ),
    triton.testing.Benchmark(
        x_names=["bsz"],
        x_vals=[i for i in range(4, 8, 2)],
        line_arg="provider",
        line_vals=["smoe", "torch"],
        line_names=["Smoe", "PyTorch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="time (ms)",
        plot_name=f"moe-full-fp32-speed-benchmark",
        # args={"dtype": torch.bfloat16, "mode": mode, "seq_len": 2048},
        args={"seq_len": 2048, "hidden_size": 4096, "intermediate_size": 14336, "num_local_experts": 8, "num_experts_per_tok": 2, "dtype": torch.float32, "mode": "full"},
    ),
    # triton.testing.Benchmark(
    #     x_names=["bsz"],
    #     x_vals=[i for i in range(4, 6, 2)],
    #     line_arg="provider",
    #     line_vals=["smoe", "torch"],
    #     line_names=["Smoe", "PyTorch"],
    #     styles=[("blue", "-"), ("green", "-")],
    #     ylabel="time (ms)",
    #     plot_name=f"moe-fwd-speed-benchmark",
    #     # args={"dtype": torch.bfloat16, "mode": mode, "seq_len": 2048},
    #     args={"seq_len": 2048, "hidden_size": 4096, "intermediate_size": 14336, "num_local_experts": 8, "num_experts_per_tok": 2, "dtype": torch.bfloat16, "mode": "forward"},
    # ),
    # triton.testing.Benchmark(
    #     x_names=["bsz"],
    #     x_vals=[i for i in range(4, 6, 2)],
    #     line_arg="provider",
    #     line_vals=["smoe", "torch"],
    #     line_names=["Smoe", "PyTorch"],
    #     styles=[("blue", "-"), ("green", "-")],
    #     ylabel="time (ms)",
    #     plot_name=f"moe-bwd-speed-benchmark",
    #     # args={"dtype": torch.bfloat16, "mode": mode, "seq_len": 2048},
    #     args={"seq_len": 2048, "hidden_size": 4096, "intermediate_size": 14336, "num_local_experts": 8, "num_experts_per_tok": 2, "dtype": torch.bfloat16, "mode": "backward"},
    # ),
    # triton.testing.Benchmark(
    #     x_names=["bsz"],
    #     x_vals=[i for i in range(4, 8, 2)],
    #     line_arg="provider",
    #     line_vals=["smoe", "torch"],
    #     line_names=["Smoe", "PyTorch"],
    #     styles=[("blue", "-"), ("green", "-")],
    #     ylabel="time (ms)",
    #     plot_name=f"moe-fwd-speed-benchmark",
    #     # args={"dtype": torch.bfloat16, "mode": mode, "seq_len": 2048},
    #     args={"seq_len": 256, "hidden_size": 1024, "intermediate_size": 2048, "num_local_experts": 8, "num_experts_per_tok": 2, "dtype": torch.bfloat16, "mode": "forward"},
    # ),
    # triton.testing.Benchmark(
    #     x_names=["bsz"],
    #     x_vals=[i for i in range(4, 8, 2)],
    #     line_arg="provider",
    #     line_vals=["smoe", "torch"],
    #     line_names=["Smoe", "PyTorch"],
    #     styles=[("blue", "-"), ("green", "-")],
    #     ylabel="time (ms)",
    #     plot_name=f"moe-full-speed-benchmark",
    #     # args={"dtype": torch.bfloat16, "mode": mode, "seq_len": 2048},
    #     args={"seq_len": 256, "hidden_size": 1024, "intermediate_size": 2048, "num_local_experts": 8, "num_experts_per_tok": 2, "dtype": torch.bfloat16, "mode": "full"},
    # ),
    # triton.testing.Benchmark(
    #     x_names=["bsz"],
    #     x_vals=[i for i in range(4, 6, 2)],
    #     line_arg="provider",
    #     line_vals=["smoe", "torch"],
    #     line_names=["Smoe", "PyTorch"],
    #     styles=[("blue", "-"), ("green", "-")],
    #     ylabel="time (ms)",
    #     plot_name=f"moe-full-speed-benchmark",
    #     # args={"dtype": torch.bfloat16, "mode": mode, "seq_len": 2048},
    #     args={"seq_len": 2048, "hidden_size": 4096, "intermediate_size": 14336, "num_local_experts": 8, "num_experts_per_tok": 2, "dtype": torch.bfloat16, "mode": "full"},
    # ),
    ]
)
def bench_speed_moe(bsz, seq_len, hidden_size, intermediate_size, num_local_experts, num_experts_per_tok, provider, mode, dtype):
    print(f"Running: bsz={bsz}, seq_len={seq_len}, hidden_size={hidden_size}, intermediate_size={intermediate_size}, num_local_experts={num_local_experts}, num_experts_per_tok={num_experts_per_tok}, provider={provider}, mode={mode}, dtype={dtype}")

    config = MixtralConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=num_local_experts,
        num_experts_per_tok=num_experts_per_tok,
    )

    if provider == "torch":
        moe_block = MixtralSparseMoeBlock(config).to("cuda").to(dtype)
    elif provider == "smoe":
        moe_block = SMoeBlock(config).to("cuda").to(dtype)
        # moe_block._reconstruct_experts()
    else:
        raise ValueError(f"Invalid provider: {provider} for MoE block")

    x = torch.randn(bsz, seq_len, hidden_size, device="cuda", dtype=dtype, requires_grad=True)
    dy = torch.randn_like(x)

    quantiles = [0.5, 0.2, 0.8]

    def fwd():
        return moe_block(x)

    if mode == "forward":
        ms, min_ms, max_ms = triton.testing.do_bench(
            fwd, quantiles=quantiles, grad_to_none=[x], warmup=5, rep=10
        )
    elif mode == "backward":
        y, r = fwd()
        ms, min_ms, max_ms = triton.testing.do_bench(
            lambda: torch.autograd.grad(y, x, dy, allow_unused=True, retain_graph=True),
            quantiles=quantiles,
            grad_to_none=[x],
            warmup=5, rep=10
        )
    elif mode == "full":
        def full():
            y, r = fwd()
            torch.autograd.grad(y, x, dy, allow_unused=True)

        ms, min_ms, max_ms = triton.testing.do_bench(
            full, quantiles=quantiles, grad_to_none=[x], warmup=5, rep=10
        )
    return ms, max_ms, min_ms


@pytest.mark.speed
def test_bench_speed_moe_wrapper():

    curr_dir = get_current_file_directory()
    dir_name = "moe_speed"
    output_dir = os.path.join(curr_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)

    bench_speed_moe.run(save_path=output_dir, print_data=True)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["bsz"],
        x_vals=[i for i in range(4, 8, 2)],
        line_arg="provider",
        line_vals=["smoe", "torch"],
        line_names=["Smoe", "PyTorch"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="Memory (MB)",
        plot_name=f"moe-full-memory-benchmark",
        args={"seq_len": 2048, "hidden_size": 4096, "intermediate_size": 14336, "num_local_experts": 8, "num_experts_per_tok": 2, "dtype": torch.float16, "mode": "full"},
    ),
)
def bench_memory_moe(bsz, seq_len, hidden_size, intermediate_size, num_local_experts, num_experts_per_tok, provider, mode, dtype):
    print(f"Running: bsz={bsz}, seq_len={seq_len}, hidden_size={hidden_size}, intermediate_size={intermediate_size}, num_local_experts={num_local_experts}, num_experts_per_tok={num_experts_per_tok}, provider={provider}, mode={mode}, dtype={dtype}")
    config = MixtralConfig(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=num_local_experts,
        num_experts_per_tok=num_experts_per_tok,
    )

    if provider == "torch":
        moe_block = MixtralSparseMoeBlock(config).to("cuda").to(dtype)
    elif provider == "smoe":
        moe_block = SMoeBlock(config).to("cuda").to(dtype)
        # moe_block._reconstruct_experts()
    else:
        raise ValueError(f"Invalid provider: {provider} for MoE block")

    x = torch.randn(bsz, seq_len, hidden_size, device="cuda", dtype=dtype, requires_grad=True)
    dx = torch.randn_like(x)

    def full():
        y, r = moe_block(x)
        torch.autograd.grad(y, x, dx, allow_unused=True, retain_graph=True)

    mem = _test_memory(full, _iter=10)
    return mem / 2**20


@pytest.mark.memory
def test_bench_memory_moe_wrapper():

    curr_dir = get_current_file_directory()
    output_dir = os.path.join(curr_dir, "moe_memory")
    os.makedirs(output_dir, exist_ok=True)

    bench_memory_moe.run(save_path=output_dir, print_data=True)\

def test_bm_torch_matmul():

    import torch
    import time

    # Set device to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set matrix dimensions
    n = 4096

    # Create random matrices
    mat1_fp32 = torch.randn(n, n, dtype=torch.float32).to(device)
    mat2_fp32 = torch.randn(n, n, dtype=torch.float32).to(device)
    mat1_fp16 = mat1_fp32.half()
    mat2_fp16 = mat2_fp32.half()

    # Perform matrix multiplication and measure execution time
    num_iterations = 10

    # FP32
    start_time = time.time()
    for _ in range(num_iterations):
        result_fp32 = torch.matmul(mat1_fp32, mat2_fp32)
    end_time = time.time()
    fp32_time = (end_time - start_time) / num_iterations

    # FP16
    start_time = time.time()
    for _ in range(num_iterations):
        result_fp16 = torch.matmul(mat1_fp16, mat2_fp16)
    end_time = time.time()
    fp16_time = (end_time - start_time) / num_iterations

    # Print benchmark results
    print(f"Matrix dimensions: {n} x {n}")
    print(f"FP32 matmul time: {fp32_time:.5f} seconds")
    print(f"FP16 matmul time: {fp16_time:.5f} seconds")
    print(f"Speedup: {fp32_time / fp16_time:.2f}x")