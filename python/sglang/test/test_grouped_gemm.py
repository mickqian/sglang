import math
import os

import torch
import pytest
from torch import load
torch.ops.load_library("/sgl-workspace/sglang/sgl-kernel/dev/build/dev_ops.so")

print(dir(torch.ops))


print(f"{torch.ops.abs.__dict__=}")
print(f"{torch.ops.dev_ops=}")
# dev_ops = load(
#     name="dev_ops",
#     sources=["fp8_blockwise_moe_kernel.cu", "cutlass_moe_helper.cu"],
#     verbose=True,
# )

from sglang.srt.utils import is_cuda

if is_cuda():
    import sgl_kernel
    from sglang.srt.layers.quantization.fp8_kernel import (
        sglang_per_token_group_quant_fp8,
    )


def dequantize_fp8_blockwise(
    q_tensor: torch.Tensor,
    scales: torch.Tensor,
    block_size_n: int,
    block_size_k: int,
    output_dtype: torch.dtype = torch.bfloat16,
):
    """
    Dequantizes a tensor that was quantized with block-wise scaling.
    This version is robust to edge cases where tensor dimensions are not
    perfectly divisible by block sizes, using an efficient advanced indexing method.

    Args:
        q_tensor: The quantized tensor of shape (..., N, K).
        scales: The scales tensor of shape (..., num_blocks_n, num_blocks_k).
        block_size_n: The block size along the N dimension.
        block_size_k: The block size along the K dimension.
        output_dtype: The desired output data type.
    """
    assert q_tensor.dtype == torch.float8_e4m3fn

    # Get the dimensions of the quantized tensor
    n_dim, k_dim = q_tensor.shape[-2], q_tensor.shape[-1]

    # Get the dimensions of the scales tensor
    # The ... part of the shape must be broadcastable.
    scales_n_dim, scales_k_dim = scales.shape[-2], scales.shape[-1]

    # 1. Create index grids for the N and K dimensions
    # For each element in the final tensor, this finds which block it belongs to.
    indices_n = torch.arange(n_dim, device=q_tensor.device) // block_size_n
    indices_k = torch.arange(k_dim, device=q_tensor.device) // block_size_k

    # 2. *** THE CRITICAL FIX ***
    # Clamp the indices to be within the bounds of the scales tensor.
    # This handles cases where q_tensor's dimensions are smaller than what
    # the scales were calculated for, or other dimension mismatches.
    indices_n = torch.clamp(indices_n, max=scales_n_dim - 1)
    indices_k = torch.clamp(indices_k, max=scales_k_dim - 1)

    # 3. Use advanced indexing to gather the scales.
    # The `[:, None]` trick broadcasts the 1D index vectors into a 2D grid
    # that matches the shape of the last two dimensions of q_tensor.
    if len(scales.shape) == len(q_tensor.shape) + 1:
        # Per-token quantization case: scales are (N, 1, 1), q_tensor is (N, K).
        # We just need to make scales broadcastable with q_tensor.
        scales_expanded = scales.squeeze(-1)
    else:
        scales_expanded = scales[..., indices_n[:, None], indices_k[None, :]]

    # 4. Perform the dequantization
    out = q_tensor.to(output_dtype) * scales_expanded.to(output_dtype)

    # This assertion is now safe because the logic guarantees matching shapes.
    assert out.shape == q_tensor.shape

    # This print is for debugging, can be removed in production.
    # print(f"dequantizing: input shape: {q_tensor.shape}, output shape: {out.shape}")

    return out


def torch_grouped_gemm(
    a_q: torch.Tensor,
    b_q: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    problem_sizes: torch.Tensor,
    expert_offsets: torch.Tensor,
    block_size_k: int,
    block_size_n: int,
    output_dtype: torch.dtype,
):
    """
    A naive PyTorch implementation of grouped GEMM with block-wise dequantization.

    Args:
        a_q: Quantized activation tensor of shape (m_total, k)
        b_q: Quantized activation tensor of shape (e, k, n)
    """
    m_total = a_q.shape[0]
    e, k, n = b_q.shape
    c_torch = torch.empty((m_total, n), dtype=output_dtype, device="cuda")
    print(f"{expert_offsets=}")
    print(f"expert count {e=}")
    print(f"activation dim {k=}")
    print(f"hidden dim {n=}")
    print(f"token count {m_total=}")
    print(f"{a_scales.shape=}, {b_scales.shape=}")
    print(f"{problem_sizes=}")
    for expert in range(e):
        m_i, k_i, n_i = problem_sizes[expert].tolist()
        if m_i == 0:
            continue

        start_idx = expert_offsets[expert]
        end_idx = expert_offsets[expert + 1]
        print(f"{start_idx=}, {end_idx=}")
        a_q_expert = a_q[start_idx:end_idx]
        a_scales_expert = a_scales[start_idx:end_idx]

        b_q_expert = b_q[expert, :k_i, :n_i]
        b_scales_expert = b_scales[
                          expert, : (n_i + block_size_n - 1) // block_size_n, :
                          ]

        print(f"{a_q_expert.shape=}, {a_scales_expert.shape=}")
        print(f"{b_q_expert.shape=}, {b_scales_expert.shape=}")
        # Dequantize
        a_dequant = dequantize_fp8_blockwise(
            a_q_expert,
            a_scales_expert.unsqueeze(1),
            block_size_k=k_i,  # Per-token quantization
            block_size_n=1,
            output_dtype=output_dtype,
        )
        # Dequantize the transpose of b_q_expert, then transpose back
        b_dequant_T = dequantize_fp8_blockwise(
            b_q_expert.T,
            b_scales_expert,
            block_size_k=block_size_k,
            block_size_n=block_size_n,
            output_dtype=output_dtype,
        )
        b_dequant = b_dequant_T.T

        print(f"{a_dequant.shape=}, {b_dequant.shape=}")

        # Matmul
        c_torch[start_idx:end_idx] = torch.matmul(a_dequant, b_dequant)

    return c_torch


def sglang_blockwise_quant_fp8(weight, block_size_n, block_size_k):
    """
    Quantizes a weight tensor with block-wise scaling.
    Args:
        weight: The weight tensor of shape (N, K).
        block_size_n: The block size along the N dimension.
        block_size_k: The block size along the K dimension.
    """
    assert weight.dim() == 2
    n, k = weight.shape

    # Pad the tensor to be divisible by block sizes
    pad_n = (block_size_n - n % block_size_n) % block_size_n
    pad_k = (block_size_k - k % block_size_k) % block_size_k
    padded_weight = torch.nn.functional.pad(weight, (0, pad_k, 0, pad_n))

    num_blocks_n = (n + pad_n) // block_size_n
    num_blocks_k = (k + pad_k) // block_size_k

    # Reshape for block-wise quantization
    reshaped_weight = padded_weight.reshape(
        num_blocks_n, block_size_n, num_blocks_k, block_size_k
    ).permute(0, 2, 1, 3)

    # Flatten each block to use the per-token quantizer
    flattened_blocks = reshaped_weight.reshape(-1, block_size_n * block_size_k)

    # Quantize each block
    q_blocks, scales = sglang_per_token_group_quant_fp8(
        flattened_blocks, group_size=block_size_n * block_size_k
    )

    # Reshape quantized tensor and scales back
    q_reshaped = q_blocks.reshape(
        num_blocks_n, num_blocks_k, block_size_n, block_size_k
    )
    q_permuted = q_reshaped.permute(0, 2, 1, 3)
    q_weight = q_permuted.reshape(n + pad_n, k + pad_k)

    # Unpad the quantized tensor
    q_final = q_weight[:n, :k]
    scales_final = scales.reshape(num_blocks_n, num_blocks_k)

    return q_final, scales_final





@pytest.mark.skipif(not is_cuda(), reason="CUDA required for this test")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
# @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
# @pytest.mark.parametrize("m_total", [1, 16, 128])
@pytest.mark.parametrize("m_total", [16])
@pytest.mark.parametrize("k", [4096])
@pytest.mark.parametrize("n", [14336])
@pytest.mark.parametrize("e", [4])
# @pytest.mark.parametrize("e", [1, 4, 8])
def test_fp8_grouped_gemm_accuracy(dtype, m_total, k, n, e):
    print("test starting...")
    torch.cuda.manual_seed_all(42)

    # For blockwise quantization, in accordance with dsv3
    block_size_n = 128
    block_size_k = 128

    # --- Generate Problem Sizes ---
    problem_sizes = torch.zeros((e, 3), dtype=torch.int32, device="cuda")
    # Assign tokens to experts somewhat randomly
    tokens_per_expert = torch.randint(
        0, m_total // e + 1, (e,), device="cuda", dtype=torch.int32
    )
    tokens_per_expert[0] += m_total - tokens_per_expert.sum()  # Ensure sum is m_total
    assert tokens_per_expert.sum() == m_total

    problem_sizes[:, 0] = tokens_per_expert
    problem_sizes[:, 1] = k
    problem_sizes[:, 2] = n

    expert_offsets = torch.zeros((e + 1), dtype=torch.int32, device="cuda")
    expert_offsets[1:] = torch.cumsum(problem_sizes[:, 0], dim=0)
    print(f"{expert_offsets=}")


    # --- Create Tensors ---
    a = torch.randn((m_total, k), dtype=dtype, device="cuda")
    b = torch.randn((e, k, n), dtype=dtype, device="cuda")

    # --- Quantize Tensors ---
    # Per-token quantization for activation
    print(f"quant group_size {k=}")
    a_q, a_scales = sglang_per_token_group_quant_fp8(a, group_size=k)

    # Block-wise quantization for weight
    b_q = torch.empty_like(b, dtype=torch.float8_e4m3fn)
    b_scales = torch.empty((e, (n + block_size_n - 1) // block_size_n, (k + block_size_k - 1) // block_size_k), device="cuda")

    # The test dimensions are perfectly divisible, so no padding is needed.
    assert n % block_size_n == 0
    assert k % block_size_k == 0

    for expert in range(e):
        q_expert, scales_expert = sglang_blockwise_quant_fp8(
            b[expert].T, block_size_n=block_size_n, block_size_k=block_size_k
        )
        b_q[expert] = q_expert.T
        b_scales[expert] = scales_expert

    print(f"{b_scales.shape=}")

    # --- CUTLASS Path ---
    c_cutlass = torch.empty((m_total, n), dtype=dtype, device="cuda")
    workspace = torch.empty((1 << 25), dtype=torch.uint8, device="cuda")
    a_ptrs = torch.empty((e,), dtype=torch.int64, device="cuda")
    b_ptrs = torch.empty((e,), dtype=torch.int64, device="cuda")
    out_ptrs = torch.empty((e,), dtype=torch.int64, device="cuda")
    a_scales_ptrs = torch.empty((e,), dtype=torch.int64, device="cuda")
    b_scales_ptrs = torch.empty((e,), dtype=torch.int64, device="cuda")
    a_sf_layout = torch.empty((e, 5), device="cuda", dtype=torch.int)
    w_sf_layout = torch.empty((e, 5), device="cuda", dtype=torch.int)

    # Note: Strides are not used by the kernel, but required for API
    a_strides = torch.full((e,), k, dtype=torch.int64, device="cuda")
    b_strides = torch.full((e,), k, dtype=torch.int64, device="cuda")
    c_strides = torch.full((e,), n, dtype=torch.int64, device="cuda")

    torch.ops.dev_ops.fp8_blockwise_scaled_grouped_mm(
    # sgl_kernel.fp8_blockwise_scaled_grouped_mm(
        c_cutlass,
        a_ptrs,
        b_ptrs,
        out_ptrs,
        a_scales_ptrs,
        b_scales_ptrs,
        a_q,
        b_q,
        a_scales,
        b_scales,
        a_strides,
        b_strides,
        c_strides,
        a_sf_layout,
        w_sf_layout,
        problem_sizes,
        expert_offsets[:-1],
        workspace,
    )

    # --- Naive PyTorch Path ---
    c_torch = torch_grouped_gemm(
        a_q, b_q, a_scales, b_scales, problem_sizes, expert_offsets, block_size_k, block_size_n, dtype
    )

    # --- Compare ---
    assert torch.allclose(c_cutlass, c_torch, atol=1e-2, rtol=1e-2)

if __name__ == "__main__":
    pytest.main([__file__])
