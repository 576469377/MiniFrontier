from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn

from minifrontier.models.minideepseekv41.layers import Gate
from minifrontier.training.runtime import RouterBalance


def _compare_ranks(rank, rendezvous):
    torch.set_num_threads(1)
    torch.manual_seed(118)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        args = SimpleNamespace(
            dim=8, n_activated_experts=2, n_routed_experts=8, route_scale=1.0, score_func="softmax"
        )
        gates = nn.ModuleList([Gate(i, args) for i in range(12)])
        for gate in gates:
            nn.init.normal_(gate.weight)
            nn.init.zeros_(gate.bias)
            gate.bias_vl = nn.Parameter(torch.zeros_like(gate.bias))
        router = RouterBalance(gates)
        # Start from separate tensors, as in checkpoints from the original collector.
        expected = [torch.arange(8, dtype=torch.float32) for _ in router.gates]
        router.load_state_dict({"counts": expected})
        for microbatch in range(2):
            x = torch.randn(11, 8)
            valid = (
                torch.arange(11) % 3 != rank
                if microbatch == 0
                else torch.zeros(11, dtype=torch.bool)
            )
            image = torch.arange(11) % 2 == 0 if rank else torch.zeros(11, dtype=torch.bool)
            with router.capture(valid):
                for i, gate in enumerate(gates):
                    gate.route_image_mask = image
                    _, selected = gate(x)
                    for j, mask in ((i, valid & ~image), (len(gates) + i, valid & image)):
                        expected[j].add_(torch.bincount(selected[mask].flatten(), minlength=8))
        for observed, target in zip(router.counts, expected, strict=True):
            torch.testing.assert_close(observed, target, rtol=0, atol=0)
            dist.all_reduce(target)
        with patch.object(dist, "all_reduce", wraps=dist.all_reduce) as collect:
            router.update(0.01)
            assert collect.call_count == 1
        for (_, bias, _), counts, target in zip(router.gates, router.counts, expected, strict=True):
            torch.testing.assert_close(bias, 0.01 * (target.mean() - target).sign(), atol=0, rtol=0)
            assert not counts.any()
    finally:
        dist.destroy_process_group()


def test_two_rank_router_histograms_and_biases(tmp_path):
    torch.multiprocessing.spawn(_compare_ranks, (f"file://{tmp_path}/rendezvous",), nprocs=2)
