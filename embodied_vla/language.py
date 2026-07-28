from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

SPECIAL_TOKENS = ("<pad>", "<unk>", "<bos>", "<eos>")
WORDS = (
    "pick",
    "grasp",
    "take",
    "put",
    "place",
    "move",
    "the",
    "red",
    "green",
    "blue",
    "cube",
    "block",
    "into",
    "in",
    "left",
    "right",
    "tray",
    "bin",
)
VOCAB = {token: index for index, token in enumerate(SPECIAL_TOKENS + WORDS)}


def tokenize(text: str, *, max_length: int = 16) -> tuple[NDArray[np.int64], NDArray[np.int8]]:
    words = re.findall(r"[a-z]+", text.lower())
    token_ids = [VOCAB["<bos>"]]
    token_ids.extend(VOCAB.get(word, VOCAB["<unk>"]) for word in words)
    token_ids.append(VOCAB["<eos>"])
    token_ids = token_ids[:max_length]

    mask = np.zeros(max_length, dtype=np.int8)
    mask[: len(token_ids)] = 1
    padded = np.full(max_length, VOCAB["<pad>"], dtype=np.int64)
    padded[: len(token_ids)] = token_ids
    return padded, mask


def make_instruction(
    color: str,
    side: str,
    *,
    template_index: int,
    templates: Sequence[str] | None = None,
) -> str:
    choices = templates or (
        "pick the {color} cube and place it in the {side} tray",
        "grasp the {color} block and put it into the {side} bin",
        "move the {color} cube into the {side} tray",
    )
    return choices[template_index % len(choices)].format(color=color, side=side)
