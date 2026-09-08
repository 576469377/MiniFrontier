"""Differentiable-equation counterpart for no-grad CSA/HCA single-token decode.

Only the unfinished/previous compression block and the sliding-window KV tail
are retained uncompressed. Completed compressed keys remain visible causally.
"""

import torch

from .kernels import rotary


def initialize_state(attn, state, x, raw_kv, compressed_kv):
    ratio = attn.compress_ratio
    state.update(
        length=x.shape[1],
        kv=raw_kv[:, -attn.window_size :].detach(),
        tail=x[:, -2 * ratio :].detach() if ratio else None,
        compressed=compressed_kv.detach() if compressed_kv is not None else None,
        index_keys=None,
    )
    if attn.indexer is not None:
        state["index_keys"] = attn.indexer.compressor(x, attn.freqs_cis[: x.shape[1]]).detach()


def decode(attn, x, state):
    output = []
    for token in x.split(1, dim=1):
        offset = state["length"]
        freqs = attn.freqs_cis[offset : offset + 1]
        qr = attn.q_norm(attn.wq_a(token))
        q = attn.wq_b(qr).unflatten(-1, (attn.n_heads, attn.head_dim))
        q = (q.float() * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + attn.eps)).to(
            q.dtype
        )
        kv = attn.kv_norm(attn.wkv(token))
        rd = attn.rope_head_dim
        q = torch.cat((q[..., :-rd], rotary(q[..., -rd:], freqs)), dim=-1)
        kv = torch.cat((kv[..., :-rd], rotary(kv[..., -rd:], freqs)), dim=-1)
        state["kv"] = torch.cat((state["kv"], kv), 1)[:, -attn.window_size :]
        state["length"] += 1
        ratio = attn.compress_ratio
        if ratio:
            state["tail"] = torch.cat((state["tail"], token), 1)[:, -2 * ratio :]
            if state["length"] % ratio == 0:
                # At a boundary, tail starts at a complete block boundary. CSA
                # sees the preceding block as well, including its learned gates.
                tail = state["tail"]
                base = state["length"] - tail.shape[1]
                time = attn.freqs_cis[base : state["length"]]
                new = attn.compressor(tail, time)[:, -1:].to(kv.dtype)
                state["compressed"] = (
                    new if state["compressed"] is None else torch.cat((state["compressed"], new), 1)
                )
                if attn.indexer is not None:
                    new_index = attn.indexer.compressor(tail, time)[:, -1:]
                    state["index_keys"] = torch.cat((state["index_keys"], new_index), 1)
        keys = state["kv"]
        compressed = state["compressed"]
        if compressed is not None and compressed.shape[1]:
            if attn.indexer is not None and attn.training_phase == "sparse_cpt":
                idx = attn.indexer
                iq = idx.wq_b(qr).unflatten(-1, (idx.n_heads, idx.head_dim))
                iq = torch.cat((iq[..., :-rd], rotary(iq[..., -rd:], freqs)), -1)
                weights = idx.weights_proj(token).float() * (idx.head_dim * idx.n_heads) ** -0.5
                scores = (
                    torch.einsum("bthd,bcd->bthc", iq.float(), state["index_keys"].float()).relu()
                    * weights[..., None]
                ).sum(2)
                if getattr(idx, "qat_enabled", False):
                    from minifrontier.training.deepseek_qat import index_scores

                    scores = index_scores(iq, state["index_keys"], weights)
                choice = scores[:, 0].argsort(dim=-1, descending=True, stable=True)[
                    ..., : min(attn.index_topk, compressed.shape[1])
                ]
                compressed = compressed.gather(1, choice[..., None].expand(-1, -1, attn.head_dim))
            keys = torch.cat((keys, compressed), 1)
        logits = torch.einsum("bthd,bcd->bhtc", q.float(), keys.float()) * attn.head_dim**-0.5
        sink = attn.attn_sink.view(1, -1, 1, 1).expand(token.shape[0], -1, 1, 1)
        probs = torch.cat((logits, sink), -1).softmax(-1)[..., :-1]
        result = torch.einsum("bhtc,bcd->bthd", probs.to(keys.dtype), keys)
        result = torch.cat((result[..., :-rd], rotary(result[..., -rd:], freqs, inverse=True)), -1)
        result = result.reshape(token.shape[0], 1, attn.n_groups, -1)
        weight = attn.wo_a.weight.view(attn.n_groups, attn.o_lora_rank, -1)
        output.append(attn.wo_b(torch.einsum("btgd,grd->btgr", result, weight).flatten(2)))
    attn.indexer_loss = x.new_zeros((), dtype=torch.float32)
    return torch.cat(output, 1)
