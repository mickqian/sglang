"""A pynccl all-to-all must be replay-safe inside a CUDA graph.

ProcessGroupNCCL's collectives are not: their host-side per-op bookkeeping
advances once at capture, so replays leave the ranks disagreeing about which
collective is in flight and both hang. That is why the full-forward DiT graph
cannot capture `dist.all_to_all_single`. Raw ncclSend/ncclRecv inside a group
carries no such state, so this exchange can live in a captured region -- this
test is what keeps that true.

    pytest -v python/sglang/multimodal_gen/test/single_test_file/test_pynccl_a2a_capture_2_gpu.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest

import torch

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.test.test_utils import CustomTestCase

_WORLD = 2


def _worker() -> int:
    import torch.distributed as dist

    from sglang.multimodal_gen.runtime.distributed.device_communicators.pynccl import (
        PyNcclCommunicator,
    )

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    cpu_group = dist.new_group(ranks=list(range(world)), backend="gloo")
    comm = PyNcclCommunicator(group=cpu_group, device=torch.device(f"cuda:{rank}"))
    if not comm.available:
        print("SKIP pynccl unavailable", flush=True)
        return 0

    failures = []

    def make(seed: int) -> torch.Tensor:
        g = torch.Generator(device="cuda").manual_seed(seed)
        return (
            torch.randn(world * 4096, dtype=torch.bfloat16, device="cuda", generator=g)
            + rank
        )

    def reference(x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        dist.all_to_all_single(out, x.contiguous())
        return out

    x = make(11)
    got = torch.empty_like(x)
    with comm.change_state(enable=True):
        comm.all_to_all_single(got, x)
    torch.cuda.synchronize()
    if not torch.equal(reference(x), got):
        failures.append("eager result differs from dist.all_to_all_single")

    # the point of the test: capture once, then replay against fresh inputs
    static_in = make(22)
    static_out = torch.empty_like(static_in)
    with comm.change_state(enable=True):
        comm.all_to_all_single(static_out, static_in)  # NCCL wants a warm path
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode="thread_local"):
            comm.all_to_all_single(static_out, static_in)
    torch.cuda.synchronize()

    for i, seed in enumerate((33, 44, 55)):
        fresh = make(seed)
        static_in.copy_(fresh)
        expected = reference(fresh)
        graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(expected, static_out):
            failures.append(f"replay {i} (seed {seed}) diverged")

    # a disabled communicator must not silently no-op into a wrong result
    try:
        comm.all_to_all_single(static_out, static_in)
        failures.append("disabled communicator did not refuse the exchange")
    except RuntimeError:
        pass

    verdict = torch.tensor([len(failures)], device="cuda")
    dist.all_reduce(verdict)
    if failures:
        print(f"rank{rank} FAIL {failures}", flush=True)
    if rank == 0:
        print(
            f"PYNCCL_A2A_CAPTURE {'FAIL' if verdict.item() else 'PASS'}",
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()
    return 1 if verdict.item() else 0


class TestPyncclA2ACapture(CustomTestCase):
    def test_all_to_all_survives_graph_replay(self):
        if not current_platform.is_cuda():
            self.skipTest("pynccl graph capture is exercised on CUDA only")
        if torch.cuda.device_count() < _WORLD:
            self.skipTest(f"needs {_WORLD} GPUs")
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc-per-node={_WORLD}",
                "--master-port=29519",
                __file__,
                "--worker",
            ],
            capture_output=True,
            text=True,
            timeout=1200,
        )
        print(proc.stdout[-4000:])
        if proc.returncode != 0:
            print(proc.stderr[-4000:], file=sys.stderr)
        self.assertEqual(proc.returncode, 0, "pynccl all-to-all is not replay-safe")
        self.assertIn("PYNCCL_A2A_CAPTURE PASS", proc.stdout)


if __name__ == "__main__":
    if "--worker" in sys.argv:
        raise SystemExit(_worker())
    unittest.main()
