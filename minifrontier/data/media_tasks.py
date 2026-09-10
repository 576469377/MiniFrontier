"""Pinned source task semantics, independent of text or pixel encoding."""

import hashlib

from minifrontier.data import sha256

SOURCE = "HuggingFaceM4/FineVision/allava_laion"
REVISION = "3c380a731a3429c1d04693d6ec16d7e683def84c"
# Only hashes of the upstream's 15 handcrafted caption instructions are distributed.
UPSTREAM_REVISION = "896b77f0ac48c95031cf31ce1b95ccaa380e68bf"
UPSTREAM_FILE_SHA256 = "7a3201aa1e5e60003c8ab30a09c311ce2dddc1bdc2c7e5cb9216f618691709a7"
CAPTION_QUESTIONS = frozenset(
    {
        "01384c3ba9a1e5e710baefb05db4b887606768fa1ecc1cd121edf9ca380bde72",
        "10920366a62273df063f1bb96d68334727cf564f2b37f6dd988d5ebd5583d2b0",
        "389b97546eacba078c82adaa4bb43c27ffad4bd1169430d3e419cf5f0e22bf24",
        "3ea4f39e84cd8f32ac60e11af3eea7864e6a9753dcb88fe3ab19f6b0495858bb",
        "42e4bbd005ccc52d4d9bf43333168ede24a2b16d0ca8ac85aa07118662e79d29",
        "46b68821ecf415bb3c93a887c676421f06a916d5d8b84ded6909c24fa0d5f89c",
        "4d139e7971703b270f2547a01e46a2ede87b1c29e43a71845d3af17261aa8c35",
        "54b5c86efb72cb42449d1c13edeaead7cc2fc5421caf71595ece1e19eec59f17",
        "7ceeba400267909b1e9a9b2ff980110611a604683cf13287a4c6e264d62821fe",
        "8d5c78aee8740c290cd626728c7574ed46a2f1d54edeb704bed86cc3a907165a",
        "a1803594efeb469cb68b8fa9b65499c51db46f3674d71fdcaf24384e81201c51",
        "b3564c27803deac775b4c545fca5bcfd91d5b0c16391cd23414060c0eaf206c4",
        "bdd82592df2efea54ba50254ab8b7f38fb39de7f4a9bdd98488b19d4e2ea2e03",
        "d8d49139605deb4eff7db3f9ba0ab63d50c5a2fccc27bc4cb08b8cb2874fe8f6",
        "f92b1a74f71e7f45947c4f59ac5b92420f0988d61b8505f8317d7698568cadc9",
    }
)


def allava_task(question):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("ALLaVA task classification requires its complete question")
    key = hashlib.sha256(" ".join(question.casefold().split()).encode()).hexdigest()
    return "caption" if key in CAPTION_QUESTIONS else "vqa"


def task_policy():
    return dict(
        name="allava-caption-templates-v1",
        source=SOURCE,
        revision=REVISION,
        upstream_revision=UPSTREAM_REVISION,
        upstream_file_sha256=UPSTREAM_FILE_SHA256,
        classifier_sha256=sha256(__file__),
    )


def effective_task(source, revision, task, question):
    if source != SOURCE:
        return task
    if revision != REVISION or task not in {"caption", "vqa"}:
        raise ValueError("ALLaVA task policy requires its pinned source and known domain")
    return allava_task(question)
