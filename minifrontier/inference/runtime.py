"""Shared checkpoint loading and generation for all four models."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import torch
from tokenizers import Tokenizer

from minifrontier.data import chat_tokens, sha256
from minifrontier.models.factory import build_model, configure_posttraining


@torch.no_grad()
def generate_ids(
    model,
    input_ids,
    *,
    max_new_tokens=64,
    temperature=0.8,
    top_p=0.9,
    vocab_size=None,
    eos_token_id=2,
    return_behavior=False,
    media=None,
    use_cache=True,
):
    if max_new_tokens < 1 or temperature < 0 or not 0 < top_p <= 1:
        raise ValueError("invalid generation settings")
    if return_behavior and (temperature != 1 or top_p != 1):
        raise ValueError("behavior traces currently require temperature=1 and top_p=1")
    from minifrontier.training.distributions import action_logits, forbidden_actions

    behavior, action_masks = [], []
    model.eval()
    limit = getattr(
        model.config, "max_position_embeddings", getattr(model.config, "max_seq_len", 0)
    )
    if input_ids.shape[1] + max_new_tokens > limit:
        raise ValueError(f"prompt plus response exceeds context length {limit}")
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
    all_ids = input_ids
    cache: Any = None
    if use_cache and model.__class__.__name__ == "MiniFrontier1ForCausalLM":
        from minifrontier.models.minifrontier1 import MiniFrontier1Cache

        cache = MiniFrontier1Cache()
    elif use_cache and model.__class__.__name__ == "MiniQwen4ForCausalLM":
        from minifrontier.models.miniqwen4 import MiniQwen4Cache

        cache = MiniQwen4Cache()
    elif use_cache and model.__class__.__name__ == "MiniKimiK3ForCausalLM":
        from minifrontier.models.minikimik3 import MiniKimiK3Cache

        cache = MiniKimiK3Cache()
    elif use_cache and model.__class__.__name__ == "MiniDeepSeekV4ForCausalLM":
        from minifrontier.models.minideepseekv4 import MiniDeepSeekV4Cache

        cache = MiniDeepSeekV4Cache()
    current = all_ids
    for _ in range(max_new_tokens):
        # On-policy behavior must use the caller's training precision. Forcing
        # BF16 here silently changed a FP32 policy's sampling distribution.
        precision = (
            contextlib.nullcontext()
            if return_behavior
            else torch.autocast(
                input_ids.device.type, dtype=torch.bfloat16, enabled=input_ids.is_cuda
            )
        )
        with precision:
            extras = {"media": media} if media and (cache is None or cache.length == 0) else {}
            out = (
                model(current, cache=cache, **extras)
                if cache is not None
                else model(all_ids, **extras)
            )
        logits = action_logits(out.logits[:, -1], vocab_size, forbidden_actions(model))
        if temperature == 0:
            token = logits.argmax(-1, keepdim=True)
        else:
            sorted_logits, order = (logits / temperature).sort(descending=True)
            cumulative = sorted_logits.softmax(-1).cumsum(-1)
            remove = cumulative > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            probs = sorted_logits.masked_fill(remove, float("-inf")).softmax(-1)
            token = order.gather(-1, torch.multinomial(probs, 1))
        token = torch.where(finished[:, None], eos_token_id, token)
        if return_behavior:
            logp = logits.log_softmax(-1).gather(-1, token)[:, 0]
            behavior.append(logp.masked_fill(finished, 0))
            action_masks.append(~finished)
        finished |= token[:, 0] == eos_token_id
        all_ids = torch.cat((all_ids, token), dim=1)
        current = token
        if finished.all():
            break
    if return_behavior:
        return all_ids, torch.stack(behavior, 1), torch.stack(action_masks, 1)
    return all_ids


def load_checkpoint(path, device="cpu"):
    path = Path(path).resolve()
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") == "mf1-export-v1":
        from minifrontier.inference.minifrontier1_export import load_export

        model, tokenizer, metadata = load_export(path, device)
        model.chat_template = metadata["chat_template"]
        return (
            model,
            tokenizer,
            {
                key: metadata[key]
                for key in ("model_name", "stage", "step", "phase", "chat_template")
            },
        )
    tokenizer_path = path.parent / "tokenizer.json"
    if sha256(tokenizer_path) != saved["tokenizer_sha256"]:
        raise ValueError("tokenizer does not match checkpoint")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model = build_model(saved["model_name"], saved["config"], phase=saved["phase"])
    model.load_state_dict(saved["model"], strict=True)
    configure_posttraining(model)
    run = saved.get("run_spec", {})
    model.chat_template = saved.get(
        "chat_template",
        run.get("rollout", {}).get("chat_template", run.get("chat_template", "legacy")),
    )
    return (
        model.to(device).eval(),
        tokenizer,
        dict(
            {key: saved[key] for key in ("model_name", "stage", "step", "phase")},
            chat_template=model.chat_template,
        ),
    )


def respond(
    model,
    tokenizer,
    prompt,
    *,
    chat=True,
    max_new_tokens=64,
    temperature=0.8,
    top_p=0.9,
    draft=None,
    draft_steps=3,
    mode=None,
    effort=None,
    images=None,
    max_image_features=64,
):
    if (
        chat
        and mode is None
        and effort is None
        and getattr(model, "chat_template", "legacy") == "control-v1"
    ):
        mode, effort = "direct", "low"
    if not chat and (mode is not None or effort is not None or images):
        raise ValueError("controlled/native-media generation currently uses the chat template")
    ids = (
        chat_tokens(
            [{"role": "user", "content": prompt}],
            tokenizer,
            generation_prompt=True,
            mode=mode,
            effort=effort,
        )[0]
        if chat
        else [1, *tokenizer.encode(prompt).ids]
    )
    device = next(model.parameters()).device
    media = None
    if images:
        from minifrontier.multimodal import prepare_record

        family = {
            "MiniKimiK3ForCausalLM": "minikimik3",
            "MiniQwen4ForCausalLM": "miniqwen4",
            "MiniDeepSeekV4ForCausalLM": "minideepseekv4",
        }[type(model).__name__]
        if "<|image|>" not in prompt:
            prompt = "<|image|>" * len(images) + "\n" + prompt
        native = prepare_record(
            dict(
                stage="sft",
                mode=mode,
                effort=effort,
                turns=[dict(role="user", content=prompt)],
                media=[dict(path=str(Path(path).resolve())) for path in images],
            ),
            tokenizer,
            family,
            max_features=max_image_features,
            generation_prompt=True,
            model_vocab_size=model.config.vocab_size,
        ).to(device)
        inputs, media = native.input_ids, native.extras["media"]
        ids = inputs[0].tolist()
    else:
        inputs = torch.tensor([ids], device=device)
    if draft is None:
        result = generate_ids(
            model,
            inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            vocab_size=tokenizer.get_vocab_size(),
            media=media,
        )
    else:
        from minifrontier.speculative import generate_speculative

        if temperature != 1 or top_p != 1:
            raise ValueError("the initial speculative sampler requires temperature=1 and top_p=1")
        result, _stats = generate_speculative(
            model,
            draft,
            inputs,
            max_new_tokens=max_new_tokens,
            draft_steps=draft_steps,
            vocab_size=tokenizer.get_vocab_size(),
            media=media,
        )
    response_ids = result[0, len(ids) :].tolist()
    if mode is not None or effort is not None:
        from minifrontier.chat_controls import parse_action

        parsed = parse_action(response_ids, tokenizer)
        if parsed["kind"] == "final":
            return (
                f"<think>{parsed['reasoning']}</think>\n" if parsed["reasoning"] else ""
            ) + parsed["text"]
        return tokenizer.decode(response_ids, skip_special_tokens=False)
    return tokenizer.decode(response_ids, skip_special_tokens=True)
