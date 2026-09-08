"""Data boundaries, preference objectives and interrupted-run equivalence."""

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from minifrontier.data import SPECIAL_TOKENS, chat_tokens
from minifrontier.training.posttrain import (
    dpo_loss,
    grouped_advantages,
    mopd_advantages,
    policy_loss,
)
from minifrontier.training.runtime import BatchCursor, atomic_save, restore_rng, rng_state


def test_cursor_shards_without_overlap_and_resumes_across_epochs():
    ranks = [BatchCursor(31, 42, rank=i, world_size=2, batch_size=3) for i in range(2)]
    first = [cursor.next() for cursor in ranks]
    assert not set(first[0]) & set(first[1])
    for _ in range(8):
        for cursor in ranks:
            cursor.next()
    restored = BatchCursor(31, 42, rank=1, world_size=2, batch_size=3, offset=ranks[1].offset)
    for _ in range(8):
        assert restored.next() == ranks[1].next()


def test_assistant_only_mask_preserves_role_boundaries_and_eos():
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.train_from_iterator(
        ["你好。问题答案。Hello world."],
        trainers.BpeTrainer(
            vocab_size=300,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    prompt = [{"role": "user", "content": "问题"}]
    prefix, _ = chat_tokens(prompt, tokenizer, generation_prompt=True)
    ids, labels = chat_tokens([*prompt, {"role": "assistant", "content": "答案"}], tokenizer)
    assert ids[: len(prefix)] == prefix
    assert set(labels[: len(prefix)]) == {-100}
    assert labels[-1] == 2 and all(v >= 0 for v in labels[len(prefix) :])


def test_dpo_prefers_chosen_and_masks_prompt():
    ref = torch.zeros(2, 4, 8)
    policy = ref.clone().requires_grad_()
    labels = torch.tensor([[-100, -100, 3, 2], [-100, -100, 4, 2]])
    initial, _ = dpo_loss(policy, ref, labels)
    initial.backward()
    assert torch.equal(policy.grad[:, 0], torch.zeros_like(policy.grad[:, 0]))
    with torch.no_grad():
        policy -= policy.grad
    improved, _ = dpo_loss(policy, ref, labels)
    assert improved < initial


def test_rl_rewards_are_detached_and_group_normalized():
    rewards = torch.tensor([0.0, 1.0, 1.0, 1.0])
    advantages = grouped_advantages(rewards, 2)
    torch.testing.assert_close(advantages, torch.tensor([-1.0, 1.0, 0.0, 0.0]))
    logp = torch.zeros(4, 3, requires_grad=True)
    teacher = torch.full((4, 3), 10.0, requires_grad=True)
    reward = mopd_advantages(teacher, logp, 5)
    assert not reward.requires_grad and reward.max() == 5
    policy_loss(logp, logp.detach(), reward, torch.ones_like(logp, dtype=torch.bool)).backward()
    assert logp.grad is not None and teacher.grad is None


def test_atomic_checkpoint_preserves_optimizer_cursor_and_rng(tmp_path):
    torch.manual_seed(42)
    model = torch.nn.Linear(4, 5)
    optimizer = torch.optim.AdamW(model.parameters())

    def update():
        optimizer.zero_grad()
        model(torch.randn(2, 4)).square().mean().backward()
        optimizer.step()

    update()
    path = tmp_path / "checkpoint.pt"
    atomic_save(
        dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            rng=rng_state(torch.device("cpu")),
        ),
        path,
    )
    update()
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    state = torch.load(path, weights_only=True)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"], torch.device("cpu"))
    update()
    for k, value in model.state_dict().items():
        assert torch.equal(value, expected[k])
