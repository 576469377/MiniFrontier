"""Audit MF1 rendering, full/cached inference, and continuation/chat generation."""

import argparse
import contextlib
import json
import time
from pathlib import Path

import torch

from minifrontier.data import sha256
from minifrontier.data.minifrontier1 import SPECIAL_TOKENS, safe_text, write_json
from minifrontier.inference.minifrontier1 import prepare_prompt
from minifrontier.inference.runtime import generate_ids, load_checkpoint
from minifrontier.models.minifrontier1 import MiniFrontier1Cache
from minifrontier.provenance import source_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-kda", action="store_true")
    parser.add_argument("--no-generation", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(
        6 * 1024**3 / torch.cuda.get_device_properties(0).total_memory
    )
    started = time.monotonic()
    model, tokenizer, metadata = load_checkpoint(args.checkpoint, "cuda:0")
    if args.reference_kda:
        for layer in model.layers:
            if layer.kind == "kda":
                layer.attention.backend = "reference"
    assert tokenizer.get_vocab_size() == model.config.vocab_size
    assert all(tokenizer.token_to_id(token) == i for i, token in enumerate(SPECIAL_TOKENS))
    report = dict(
        state="running",
        checkpoint_sha256=sha256(args.checkpoint),
        tokenizer_sha256=sha256(Path(args.checkpoint).parent / "tokenizer.json"),
        source=source_identity(),
        runner_sha256=sha256(__file__),
        metadata=metadata,
        kda_backend="reference" if args.reference_kda else "checkpoint_config",
        tf32=False,
        special_tokens_verified=True,
        cache_checks=[],
        generations=[],
    )
    write_json(output / "report.json", report)
    prompts = [
        ("continuation", "太阳从东边升起，"),  # noqa: RUF001 -- preserve natural Chinese punctuation
        ("continuation", "The sum of 7 and 5 is"),
        ("chat", "请用一句话介绍你自己。"),
        ("chat", "7+5="),
        ("chat", "把 hello 翻译成中文。"),
    ]
    with torch.inference_mode():
        for precision in ("fp32", "bf16"):
            ids = torch.tensor(
                [[1, 4, *safe_text(tokenizer, "这是一个检查缓存与完整前向结果的句子。")]],
                device="cuda:0",
            )
            split = max(2, ids.shape[1] // 2)
            context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if precision == "bf16"
                else contextlib.nullcontext()
            )
            with context:
                full = model(ids).logits.float()
                cache = MiniFrontier1Cache()
                chunks = [model(ids[:, :split], cache=cache).logits.float()]
                for position in range(split, ids.shape[1]):
                    chunks.append(
                        model(ids[:, position : position + 1], cache=cache).logits.float()
                    )
                streamed = torch.cat(chunks, 1)
            delta = (full - streamed).abs()
            check = dict(
                precision=precision,
                input_ids=ids[0].tolist(),
                split=split,
                max_abs=float(delta.max()),
                mean_abs=float(delta.mean()),
                greedy_token_agreement=float(
                    (full.argmax(-1) == streamed.argmax(-1)).float().mean()
                ),
                fp32_close=bool(torch.allclose(full, streamed, atol=2e-4, rtol=2e-4))
                if precision == "fp32"
                else None,
            )
            report["cache_checks"].append(check)
            write_json(output / "report.json", report)
            print(json.dumps(check), flush=True)
        for mode, prompt in prompts:
            if args.no_generation:
                break
            ids = (
                prepare_prompt(model, tokenizer, prompt)[0]
                if mode == "chat"
                else torch.tensor([[1, 4, *safe_text(tokenizer, prompt)]])
            )
            for sampling in (False, True):
                torch.manual_seed(42)
                result = generate_ids(
                    model,
                    ids.to("cuda:0"),
                    max_new_tokens=40,
                    temperature=1 if sampling else 0,
                    top_p=1,
                    vocab_size=tokenizer.get_vocab_size(),
                )
                tokens = result[0, ids.shape[1] :].tolist()
                row = dict(
                    mode=mode,
                    prompt=prompt,
                    input_ids=ids[0].tolist(),
                    rendered_prompt=tokenizer.decode(ids[0].tolist(), skip_special_tokens=False),
                    seed=42,
                    temperature=1 if sampling else 0,
                    top_p=1,
                    max_new_tokens=40,
                    output_ids=tokens,
                    raw_output=tokenizer.decode(tokens, skip_special_tokens=False),
                    text=tokenizer.decode(tokens, skip_special_tokens=True),
                    termination="eos" if 2 in tokens else "token_limit",
                    unique_token_fraction=len(set(tokens)) / max(1, len(tokens)),
                )
                report["generations"].append(row)
                write_json(output / "report.json", report)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    report.update(state="complete", elapsed_seconds=time.monotonic() - started)
    write_json(output / "report.json", report)


if __name__ == "__main__":
    main()
