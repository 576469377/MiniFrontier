"""Two-rank source-model optimizer/update/resume acceptance, not a training run."""

import json
from copy import deepcopy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_miniqwen4 import tiny_config
from torch.nn.parallel import DistributedDataParallel

from minifrontier.models.miniqwen4 import MiniQwen4ForCausalLM
from minifrontier.training.miniqwen4_optim import MiniQwen4Optimizer


def _worker(rank, rendezvous, cuda, phase):
    torch.set_num_threads(1)
    torch.manual_seed(149)
    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    dist.init_process_group(
        "nccl" if cuda else "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=90),
    )
    try:
        cfg = tiny_config(gradient_checkpointing=True)
        model = MiniQwen4ForCausalLM(cfg, training_phase=phase).to(device)
        ddp = DistributedDataParallel(
            model, device_ids=[rank] if cuda else None, broadcast_buffers=False
        )
        optimizer = MiniQwen4Optimizer(model, lr=0.001, adam_lr=0.0002)
        ids = torch.tensor([[5 + rank, 6, 2, 8, 9 + rank, 10, 11]], device=device)
        original_head = model.lm_head.weight.detach().clone()
        original_indexer = (
            model.model.layers[3].self_attn.indexer.index_qk_proj.weight.detach().clone()
        )

        def update(wrapped, opt):
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cuda):
                result = wrapped(ids, labels=ids)
            assert torch.isfinite(result.loss)
            result.loss.backward()
            opt.step()
            return result.loss.detach().float()

        first_loss = update(ddp, optimizer)
        assert torch.equal(original_head, model.lm_head.weight) == (phase == "dense_distill")
        assert torch.equal(
            original_indexer, model.model.layers[3].self_attn.indexer.index_qk_proj.weight
        ) == (phase == "dense_pretrain")
        saved_model = deepcopy(model.state_dict())
        saved_optimizer = deepcopy(optimizer.state_dict())
        expected_loss = update(ddp, optimizer)
        resumed = MiniQwen4ForCausalLM(cfg, training_phase=phase).to(device)
        resumed.load_state_dict(saved_model, strict=True)
        resumed_ddp = DistributedDataParallel(
            resumed, device_ids=[rank] if cuda else None, broadcast_buffers=False
        )
        resumed_optimizer = MiniQwen4Optimizer(resumed, lr=0.001, adam_lr=0.0002)
        resumed_optimizer.load_state_dict(saved_optimizer)
        actual_loss = update(resumed_ddp, resumed_optimizer)
        torch.testing.assert_close(expected_loss, actual_loss, atol=0, rtol=0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, resumed.state_dict()[key], atol=0, rtol=0)
        state = torch.cat([p.detach().float().flatten() for p in resumed.parameters()])
        reference = state.clone()
        dist.broadcast(reference, src=0)
        torch.testing.assert_close(state, reference, atol=0, rtol=0)
        print(
            json.dumps(
                {
                    "kind": "engineering_acceptance",
                    "rank": rank,
                    "device": str(device),
                    "phase": phase,
                    "first_loss": first_loss.item(),
                    "second_loss": expected_loss.item(),
                    "exact_resume": True,
                    "max_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20
                    if cuda
                    else 0,
                }
            ),
            flush=True,
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.distributed
@pytest.mark.parametrize("phase", ["dense_pretrain", "dense_distill", "sparse_cpt"])
def test_source_qwen_two_cpu_ranks_update_and_resume(tmp_path, phase):
    mp.spawn(_worker, args=(str(tmp_path / "rendezvous"), False, phase), nprocs=2, join=True)


@pytest.mark.cuda
@pytest.mark.distributed
@pytest.mark.parametrize("phase", ["dense_pretrain", "dense_distill", "sparse_cpt"])
def test_source_qwen_two_gpu_bf16_ranks_update_and_resume(tmp_path, monkeypatch, phase):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two available CUDA devices")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    mp.spawn(_worker, args=(str(tmp_path / "rendezvous"), True, phase), nprocs=2, join=True)
