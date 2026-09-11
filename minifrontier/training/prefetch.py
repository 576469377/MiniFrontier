"""One CPU window of ordered lookahead, with a separately committed cursor.

The producer exclusively owns its dataset/cursor. It must use its own RNG and
return an immutable cursor snapshot. Checkpoints save only consumed snapshots;
unconsumed work is discarded and regenerated on resume.
"""

import os
from concurrent.futures import ThreadPoolExecutor


def enabled():
    value = os.environ.get("MINIFRONTIER_PREFETCH_WINDOWS", "0")
    if value not in {"0", "1"}:
        raise ValueError("MINIFRONTIER_PREFETCH_WINDOWS must be 0 or 1")
    return value == "1"


class OrderedPrefetch:
    def __init__(self, produce, state):
        self.produce = produce
        self.committed = state
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="training-data")
        self.future = None

    def next(self):
        if self.future is None:
            self.future = self.pool.submit(self.produce)
        value, state = self.future.result()
        self.committed = state
        self.future = self.pool.submit(self.produce)
        return value

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


def native_window(dataset, cursor, batch_size, target_inputs, sequence_length):
    """CPU equivalent of the single-rank sample-bounded training window."""
    import torch

    from minifrontier.multimodal import TrainingBatch, collate
    from minifrontier.training.batching import microbatch_count

    rows, inputs = [], 0
    while inputs < target_inputs:
        count = microbatch_count(batch_size, target_inputs - inputs, sequence_length, 1)
        selected = [dataset[index] for index in cursor.next(count)]
        tensors = [row.input_ids if isinstance(row, TrainingBatch) else row[0] for row in selected]
        if any(x.shape[-1] > sequence_length for x in tensors):
            raise ValueError("input row exceeds the declared sequence length used for batching")
        inputs += sum(int(x.ne(0).sum()) for x in tensors)
        rows.extend(selected)
        if inputs < target_inputs and len(rows) >= 1024 * batch_size:
            raise ValueError("global input target needs more than 1024 microbatches")
    window = [
        collate(rows[start : start + batch_size], torch.device("cpu"))
        for start in range(0, len(rows), batch_size)
    ]
    return (window, inputs), cursor.state_dict()
