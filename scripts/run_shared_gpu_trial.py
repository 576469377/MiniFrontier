"""Apply a per-process allocator ceiling before invoking the pinned trainer."""

import argparse
import json

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-gib", type=float, required=True)
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    total = torch.cuda.get_device_properties(0).total_memory
    fraction = args.memory_gib * 1024**3 / total
    if not 0 < fraction < 1:
        raise ValueError("allocator ceiling must be positive and below device capacity")
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    print(
        json.dumps(dict(event="allocator_limit", memory_gib=args.memory_gib, fraction=fraction)),
        flush=True,
    )
    from minifrontier.training.train import main as train

    train(args.training_args[1:] if args.training_args[:1] == ["--"] else args.training_args)


if __name__ == "__main__":
    main()
