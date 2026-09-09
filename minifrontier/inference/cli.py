"""Arguments for the existing `minifrontier generate` command."""

import argparse

from minifrontier.inference.runtime import load_checkpoint, respond


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float)
    p.add_argument("--top-p", type=float)
    p.add_argument("--draft", help="optional trained draft bound to this exact checkpoint")
    p.add_argument("--draft-steps", type=int, default=3)
    p.add_argument(
        "--mode",
        choices=["direct", "thinking", "tool", "non-thinking", "thinking-high", "thinking-max"],
    )
    p.add_argument("--effort", choices=["low", "high", "max"])
    p.add_argument(
        "--image", action="append", help="local image; repeat for multiple complete images"
    )
    p.add_argument("--max-image-features", type=int, default=64)
    p.add_argument("--device", default="cpu")
    p.add_argument("--completion", action="store_true")
    args = p.parse_args(argv)
    draft = None
    if args.draft:
        from minifrontier.training.drafts import load_draft

        model, draft, tokenizer, _meta = load_draft(args.draft, args.checkpoint, args.device)
    else:
        model, tokenizer, _meta = load_checkpoint(args.checkpoint, args.device)
    print(
        respond(
            model,
            tokenizer,
            args.prompt,
            chat=not args.completion,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature if args.temperature is not None else (1 if draft else 0.8),
            top_p=args.top_p if args.top_p is not None else (1 if draft else 0.9),
            draft=draft,
            draft_steps=args.draft_steps,
            mode=args.mode,
            effort=args.effort,
            images=args.image,
            max_image_features=args.max_image_features,
        )
    )


if __name__ == "__main__":
    main()
