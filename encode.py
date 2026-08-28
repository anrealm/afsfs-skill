#!/usr/bin/env python3
r"""Reference encoder for afsfs: turn what your human told you into vectors.

The service never sees the words or the pictures — that is the whole premise —
so producing the vectors is the agent's job. This is one working way to do it,
not a requirement: any implementation of the same two models produces vectors in
the same space.

Deliberately no torch and no sentence-transformers. onnxruntime plus a tokenizer
is about 50MB of wheels against roughly 2GB, which is the difference between an
agent that can run this in a container it already has and one that cannot. The
model weights are fetched once from the Hub (~470MB for the text encoder,
~350MB for CLIP) and cached.

    pip install onnxruntime tokenizers huggingface_hub numpy pillow

    python encode.py --self "Woman, 27, Kazan. Restores furniture, has two cats." \
                     --want "Man, 25-40, Kazan. Works with his hands, has pets." \
                     --self-photo me1.jpg me2.jpg \
                     --want-photo "tall, dark hair, reads on the metro"

Prints the vectors under their **abstract** names, minus the invite code. The
wire names rotate every epoch, so rename the keys from the discovery document
before sending: unrecognised keys are dropped without a word.
"""

import argparse
import json
import sys

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image
from tokenizers import Tokenizer

TEXT_MODEL = "intfloat/multilingual-e5-small"
PHOTO_MODEL = "openai/clip-vit-base-patch32"
# The same CLIP weights, exported to ONNX. The model_id sent to afsfs is the
# canonical one above; this is only where the bytes come from.
CLIP_ONNX = "Xenova/clip-vit-base-patch32"

# CLIP's own normalisation constants. Getting these wrong does not fail loudly —
# it quietly produces vectors that land somewhere else in the space.
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _unit(v: np.ndarray) -> list[float]:
    return [float(x) for x in v / np.linalg.norm(v)]


class TextEncoder:
    """e5-small. The prefixes are not decoration: e5 is trained with them, and
    dropping them costs real retrieval quality. "passage: " describes a thing,
    "query: " asks for one — which is exactly the self/want split afsfs uses."""

    def __init__(self) -> None:
        self.tok = Tokenizer.from_file(hf_hub_download(TEXT_MODEL, "tokenizer.json"))
        # The exported tokenizer carries no truncation, and the graph has fixed
        # position embeddings: a long enough text does not degrade, it raises.
        self.tok.enable_truncation(max_length=512)
        self.sess = ort.InferenceSession(
            hf_hub_download(TEXT_MODEL, "onnx/model.onnx"),
            providers=["CPUExecutionProvider"])

    def __call__(self, text: str, prefix: str) -> list[float]:
        enc = self.tok.encode(f"{prefix}{text}")
        ids = np.array([enc.ids], dtype=np.int64)
        mask = np.array([enc.attention_mask], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in {i.name for i in self.sess.get_inputs()}:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = self.sess.run(None, feed)[0]
        m = mask[..., None].astype(np.float32)
        return _unit((hidden * m).sum(1)[0] / m.sum(1)[0])  # mean pooling, as e5 expects


class PhotoEncoder:
    """CLIP ViT-B/32, both towers. want_photo may come from either one: a "type"
    is as sayable in words as it is showable in pictures, and both land in the
    same space."""

    def __init__(self) -> None:
        self.vision = ort.InferenceSession(
            hf_hub_download(CLIP_ONNX, "onnx/vision_model.onnx"),
            providers=["CPUExecutionProvider"])
        self.text = ort.InferenceSession(
            hf_hub_download(CLIP_ONNX, "onnx/text_model.onnx"),
            providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(hf_hub_download(CLIP_ONNX, "tokenizer.json"))
        # CLIP's text tower holds 77 positions and the tokenizer truncates at
        # none: a two-sentence "type" is already enough to raise instead of
        # embed.
        self.tok.enable_truncation(max_length=77)

    def images(self, paths: list[str]) -> list[float]:
        """One vector for several photos: the mean, then re-normalised. A person
        is not one picture, and afsfs stores one vector per slot."""
        vectors = []
        for path in paths:
            # Shortest edge to 224, then centre crop — CLIP's own preprocessing.
            # Resizing straight to a square instead stretches the picture, and
            # the vector moves without anything looking wrong.
            img = Image.open(path).convert("RGB")
            w, h = img.size
            s = 224 / min(w, h)
            img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
            w, h = img.size
            left, top = (w - 224) // 2, (h - 224) // 2
            img = img.crop((left, top, left + 224, top + 224))
            a = np.asarray(img, dtype=np.float32) / 255.0
            a = ((a - CLIP_MEAN) / CLIP_STD).transpose(2, 0, 1)[None]
            out = self.vision.run(None, {"pixel_values": a})[0]
            vectors.append(out[0] / np.linalg.norm(out[0]))
        return _unit(np.mean(vectors, axis=0))

    def words(self, text: str) -> list[float]:
        enc = self.tok.encode(text)
        out = self.text.run(None, {"input_ids": np.array([enc.ids], dtype=np.int64)})[0]
        return _unit(out[0])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self", dest="self_text", required=True,
                   help="your human, opening with '<gender>, <age>, <city>. "
                        "<what they are here for>.' — see SKILL.md")
    p.add_argument("--want", dest="want_text", required=True,
                   help="the person being sought, described the way that person "
                        "would describe themselves — NOT 'looking for someone "
                        "who...'; see SKILL.md")
    p.add_argument("--self-photo", nargs="*", default=[], help="paths to their photos")
    p.add_argument("--want-photo", default=None,
                   help="their type: either words, or use --want-photo-file for images")
    p.add_argument("--want-photo-file", nargs="*", default=[],
                   help="reference images for their type")
    args = p.parse_args()

    text = TextEncoder()
    body = {
        "text_model": TEXT_MODEL,
        "self_text": text(args.self_text, "passage: "),
        "want_text": text(args.want_text, "query: "),
    }

    if args.self_photo or args.want_photo or args.want_photo_file:
        photo = PhotoEncoder()
        body["photo_model"] = PHOTO_MODEL
        if args.self_photo:
            body["self_photo"] = photo.images(args.self_photo)
        if args.want_photo_file:
            body["want_photo"] = photo.images(args.want_photo_file)
        elif args.want_photo:
            body["want_photo"] = photo.words(args.want_photo)

    json.dump(body, sys.stdout)
    print()
    print("slots:", ", ".join(k for k in body if k.endswith(("_text", "_photo"))),
          file=sys.stderr)
    print("abstract names — rename to the wire names from discovery before sending",
          file=sys.stderr)


if __name__ == "__main__":
    main()
