"""Raw-media messages through the same encoder, batching and differentiable model."""

import random
import string

import pytest
import torch
from PIL import Image
from test_miniqwen4 import tiny_config
from test_new_backbones import tiny_deepseek, tiny_kimi
from tokenizers import Tokenizer

from minifrontier.data.corpus import CorpusBuilder, train_tokenizer
from minifrontier.data.media_hash import decoded_hashes
from minifrontier.models.minideepseekv4 import MiniDeepSeekV4ForCausalLM
from minifrontier.models.minideepseekv4.vision import DeepSeekVisionConfig
from minifrontier.models.minikimik3 import MiniKimiK3ForCausalLM
from minifrontier.models.minikimik3.vision import KimiVisionConfig
from minifrontier.models.miniqwen4 import MiniQwen4ForCausalLM
from minifrontier.models.miniqwen4.vision import QwenVisionConfig
from minifrontier.multimodal import collate, prepare_record


@pytest.fixture
def tokenizer(tmp_path):
    builder = CorpusBuilder(tmp_path / "text")
    random_source = random.Random(41)
    for i in range(50):
        builder.add(
            dict(
                source="local-test",
                revision="1",
                item_id=str(i),
                group_id=str(i),
                license="apache-2.0",
                lang="en",
                task="test",
                stage="pretrain",
                text=" ".join(
                    "".join(random_source.choices(string.ascii_lowercase, k=8)) for _ in range(12)
                ),
            )
        )
    builder.finalize()
    path = tmp_path / "tokenizer.json"
    train_tokenizer(builder.root, path, 300, byte_budget=4096)
    return Tokenizer.from_file(str(path))


@pytest.mark.parametrize("family", ["minikimik3", "miniqwen4", "minideepseekv4"])
def test_raw_image_template_ple_labels_and_gradient(family, tokenizer, tmp_path):
    image = Image.new("RGB", (56, 56), (255, 0, 0))
    image.save(tmp_path / "red.png")
    record = dict(
        stage="sft",
        turns=[
            dict(role="user", content="What color? <|image|>"),
            dict(role="assistant", content="Red."),
        ],
        media=[dict(path="red.png", min_pixels=1024, **decoded_hashes(image))],
    )
    packed = prepare_record(
        record, tokenizer, family, root=tmp_path, max_features=4, model_vocab_size=512
    )
    extra = prepare_record(
        dict(
            stage="sft",
            turns=[dict(role="user", content="Hello."), dict(role="assistant", content="Hello!")],
        ),
        tokenizer,
        family,
        root=tmp_path,
        model_vocab_size=512,
    )
    batch = collate([packed, extra], "cpu")
    if family == "minikimik3":
        v = KimiVisionConfig(
            depth=1,
            hidden_size=32,
            qkv_hidden_size=48,
            num_heads=2,
            intermediate_size=64,
            output_size=32,
        )
        model = MiniKimiK3ForCausalLM(
            tiny_kimi(
                vocab_size=512, max_position_embeddings=512, vision_config=v, mtp_enabled=True
            )
        )
    elif family == "miniqwen4":
        v = QwenVisionConfig(
            depth=1, hidden_size=32, num_heads=2, intermediate_size=64, output_size=32
        )
        model = MiniQwen4ForCausalLM(
            tiny_config(
                vocab_size=512,
                hidden_size=32,
                max_position_embeddings=512,
                vision_config=v,
                mtp_enabled=True,
            )
        )
        assert torch.equal(batch.extras["ple_input_ids"], batch.input_ids)
        assert batch.extras["position_ids"].shape == (3, 2, batch.input_ids.shape[1])
    else:
        v = DeepSeekVisionConfig(
            depth=1, hidden_size=32, num_heads=2, intermediate_size=64, output_size=32
        )
        model = MiniDeepSeekV4ForCausalLM(
            tiny_deepseek(vocab_size=512, max_seq_len=512, vision_config=v, mtp_enabled=True)
        )
    result = model(
        batch.input_ids,
        attention_mask=batch.input_ids.ne(0),
        labels=batch.labels,
        return_logits=False,
        **batch.extras,
    )
    assert torch.isfinite(result.loss) and result.mtp_tokens > 0
    result.loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.vision.parameters())
    assert batch.image_count == 1 and batch.video_count == 0 and batch.image_features <= 4
    text = tokenizer.decode(
        batch.labels[0][batch.labels[0].ne(-100)].tolist(), skip_special_tokens=True
    )
    assert text == "Red."
    with pytest.raises(ValueError, match="hash differs"):
        changed = dict(record, media=[dict(record["media"][0], rgb_sha256="bad")])
        prepare_record(
            changed, tokenizer, family, root=tmp_path, max_features=4, model_vocab_size=512
        )
    # The exact training processor also supplies multimodal cache prefill.
    from minifrontier.models.minideepseekv4 import MiniDeepSeekV4Cache
    from minifrontier.models.minikimik3 import MiniKimiK3Cache
    from minifrontier.models.miniqwen4 import MiniQwen4Cache

    cache = {
        "minikimik3": MiniKimiK3Cache,
        "miniqwen4": MiniQwen4Cache,
        "minideepseekv4": MiniDeepSeekV4Cache,
    }[family]()
    model.eval()
    continuation = torch.tensor([[30, 40, 50]])
    full_ids = torch.cat((packed.input_ids, continuation), 1)
    with torch.no_grad():
        expected = model(full_ids, media=packed.extras["media"]).logits
        first = model(packed.input_ids, cache=cache, media=packed.extras["media"]).logits
        snapshot = cache.snapshot()
        second = model(continuation, cache=cache).logits
        torch.testing.assert_close(torch.cat((first, second), 1), expected, rtol=3e-5, atol=3e-6)
        cache.restore(snapshot)
        torch.testing.assert_close(second, model(continuation, cache=cache).logits, rtol=0, atol=0)
