"""Trainable, deterministic Engram lookup; adapted from official dba1be0 (MIT).

Tables and projections are unquantized. Hashing stops at padding, sample
boundaries and every image-span token. The normalized token map is a checkpoint
buffer; bind it to the training tokenizer before starting a new run.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


def compressed_token_map(tokenizer: Any, vocab_size: int) -> list[int]:
    from tokenizers import Regex, normalizers

    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = getattr(tokenizer, "backend_tokenizer", tokenizer)
    backend = getattr(backend, "tokenizer", backend)
    keys: dict[str, int] = {}
    result = []
    for token_id in range(vocab_size):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            key = normalizer.normalize_str(text) or text
        if key not in keys:
            keys[key] = len(keys)
        result.append(keys[key])
    return result


def next_prime(start: int, seen: set[int]) -> int:
    value = start + 1
    while (
        value in seen or value < 2 or any(value % d == 0 for d in range(2, math.isqrt(value) + 1))
    ):
        value += 1
    seen.add(value)
    return value


def table_layout(config: Any) -> dict[int, list[int]]:
    seen: set[int] = set()
    result = {}
    for layer_id in config.engram_layer_ids:
        primes = []
        for _ in range(config.engram_max_ngram_size - 1):
            current = config.engram_vocab_size - 1
            for _ in range(config.engram_n_heads):
                current = next_prime(current, seen)
                primes.append(current)
        result[layer_id] = primes
    return result


class Engram(nn.Module):
    primes: Tensor
    offsets: Tensor
    token_map: Tensor
    multipliers: Tensor
    token_map_bound: Tensor

    def __init__(self, config: Any, layer_id: int, primes: list[int]):
        super().__init__()
        self.config, self.layer_id = config, layer_id
        self.embed = nn.Embedding(sum(primes), config.engram_head_dim)
        self.wkv = nn.Linear(
            len(primes) * config.engram_head_dim, config.dim * (config.hc_mult + 1), bias=False
        )
        self.q_weight = nn.Parameter(torch.ones(config.hc_mult, config.dim))
        self.k_weight = nn.Parameter(torch.ones(config.hc_mult, config.dim))
        self.register_buffer("primes", torch.tensor(primes, dtype=torch.long))
        self.register_buffer(
            "offsets", torch.tensor([0, *np.cumsum(primes[:-1])], dtype=torch.long)
        )
        self.register_buffer("token_map", torch.arange(config.vocab_size))
        self.register_buffer(
            "multipliers", torch.ones(config.engram_max_ngram_size, dtype=torch.long)
        )
        self.register_buffer("token_map_bound", torch.tensor(False))
        self.bind_token_map(list(range(config.vocab_size)), bound=False)

    @torch.no_grad()
    def bind_token_map(self, mapping: list[int], *, bound=True) -> None:
        if len(mapping) != self.config.vocab_size or min(mapping) < 0:
            raise ValueError("Engram compressed map must cover the vocabulary")
        count = max(mapping) + 1
        bound_int = max(1, (np.iinfo(np.int64).max // count) // 2)
        values = (
            np.random.default_rng(10007 * self.layer_id).integers(
                0, bound_int, size=(self.config.engram_max_ngram_size,), dtype=np.int64
            )
            * 2
            + 1
        )
        self.token_map.copy_(torch.tensor(mapping, device=self.token_map.device))
        self.multipliers.copy_(torch.tensor(values, device=self.multipliers.device))
        self.token_map_bound.fill_(bound)

    def hashes(
        self, input_ids: Tensor, token_mask: Tensor, sample_ids: Tensor | None = None
    ) -> Tensor:
        c = self.config
        mapped = self.token_map[input_ids]
        pad_id = self.token_map[c.engram_pad_id]
        lookback, blocked = [], torch.zeros_like(token_mask)
        length = input_ids.shape[1]
        pos = torch.arange(length, device=input_ids.device)
        for shift in range(c.engram_max_ngram_size):
            source = (pos - shift).clamp_min(0)
            blocked = blocked | (pos[None] < shift) | ~token_mask[:, source]
            if sample_ids is not None:
                blocked = blocked | (sample_ids[:, source] != sample_ids)
            lookback.append(torch.where(blocked, pad_id, mapped[:, source]))
        products = torch.stack(lookback, -1) * self.multipliers
        rolling, hashes = products[..., 0], []
        for i in range(1, c.engram_max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., i])
            begin, end = (i - 1) * c.engram_n_heads, i * c.engram_n_heads
            hashes.append(rolling[..., None] % self.primes[begin:end])
        return torch.cat(hashes, -1) + self.offsets

    def forward(self, stream: Tensor, input_ids: Tensor, token_mask: Tensor) -> Tensor:
        c = self.config
        kv = self.wkv(self.embed(self.hashes(input_ids, token_mask)).flatten(-2))
        key, value = kv.split((c.hc_mult * c.dim, c.dim), -1)
        key, x = key.float().unflatten(-1, (c.hc_mult, c.dim)), stream.float()
        rms = torch.rsqrt(x.square().mean(-1) + c.norm_eps) * torch.rsqrt(
            key.square().mean(-1) + c.norm_eps
        )
        dot = (x * self.q_weight.float() * self.k_weight.float() * key).sum(-1) * rms * c.dim**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        gate = gate.masked_fill(~token_mask[..., None], 0)
        return (x + gate[..., None] * value.float().unsqueeze(-2)).to(stream.dtype)
