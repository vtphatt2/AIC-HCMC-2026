"""
Torch-free CLIP BPE tokenizer for the ONNX text-encoder backend.

Vendored from open_clip's tokenizer.py (itself copied from
https://github.com/openai/CLIP, MIT License, Copyright (c) 2021 OpenAI),
with the torch dependency removed: __call__ returns a numpy int64 array
instead of a torch.LongTensor so the "onnx" backend never has to import
torch or open_clip_torch.

Needs the bundled bpe_simple_vocab_16e6.txt.gz (see onnx-models/), plus the
lightweight `ftfy` and `regex` packages — no torch, no open_clip_torch.
"""
from __future__ import annotations

import gzip
import html
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Union

import ftfy
import numpy as np
import regex as re

DEFAULT_CONTEXT_LENGTH = 72


@lru_cache()
def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text: str) -> str:
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text: str) -> str:
    return " ".join(text.split()).strip()


class SimpleTokenizer:
    """CLIP BPE tokenizer. __call__ returns a numpy int64 array, shape (batch, context_length)."""

    def __init__(self, bpe_path: str | Path, context_length: int = DEFAULT_CONTEXT_LENGTH):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1 : 49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        special_tokens = ["<start_of_text>", "<end_of_text>"]
        vocab.extend(special_tokens)
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {t: t for t in special_tokens}
        special = "|".join(special_tokens)
        self.pat = re.compile(
            special + r"""|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )
        self.sot_token_id = self.encoder["<start_of_text>"]
        self.eot_token_id = self.encoder["<end_of_text>"]
        self.context_length = context_length

    def bpe(self, token: str) -> str:
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        result = " ".join(word)
        self.cache[token] = result
        return result

    def encode(self, text: str) -> list[int]:
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def __call__(self, texts: Union[str, List[str]], context_length: Optional[int] = None) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        context_length = context_length or self.context_length

        all_tokens = [[self.sot_token_id] + self.encode(text) + [self.eot_token_id] for text in texts]
        result = np.zeros((len(all_tokens), context_length), dtype=np.int64)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                tokens = tokens[:context_length]
                tokens[-1] = self.eot_token_id
            result[i, : len(tokens)] = np.array(tokens, dtype=np.int64)

        return result


def _find_bpe_vocab() -> Path | None:
    override = os.getenv("PECORE_BPE_VOCAB_PATH")
    if override:
        return Path(override)

    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "challenge_resources" / "onnx-models" / "bpe_simple_vocab_16e6.txt.gz"
        if candidate.is_file():
            return candidate
    return None
