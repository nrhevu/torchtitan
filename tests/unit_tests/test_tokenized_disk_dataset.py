# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest

import torch
from datasets import Dataset

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.hf_datasets import DatasetConfig
from torchtitan.hf_datasets.text_datasets import DATASETS, HuggingFaceTextDataset


class DummyTokenizer(BaseTokenizer):
    def __init__(self, tokens: list[int] | None = None) -> None:
        super().__init__()
        self.tokens = tokens or [1, 2, 3]
        self.encode_calls: list[tuple[tuple, dict]] = []

    def encode(self, *args, **kwargs) -> list[int]:
        self.encode_calls.append((args, kwargs))
        return self.tokens

    def decode(self, *args, **kwargs) -> str:
        return ""

    def get_vocab_size(self) -> int:
        return 0


class NoEncodeTokenizer(DummyTokenizer):
    def encode(self, *args, **kwargs) -> list[int]:
        raise AssertionError("tokenized_disk samples should not be tokenized again")


class TestTokenizedDiskDataset(unittest.TestCase):
    def tearDown(self) -> None:
        DATASETS.pop("unit_text", None)

    def test_tokenized_processor_filters_attention_mask(self):
        processor = DATASETS["tokenized_disk"].sample_processor

        assert processor({"input_ids": [7, 8, 9]}) == [7, 8, 9]
        assert processor(
            {
                "input_ids": [10, 11, 12, 13],
                "attention_mask": [1, 0, 1, 1],
            }
        ) == [10, 12, 13]

    def test_tokenized_disk_iterator_uses_input_ids_without_tokenizer_encode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Dataset.from_dict(
                {
                    "input_ids": [[10, 11, 12, 13]],
                    "attention_mask": [[1, 0, 1, 1]],
                }
            ).save_to_disk(tmpdir)

            ds = HuggingFaceTextDataset(
                dataset_name="tokenized_disk",
                dataset_path=tmpdir,
                tokenizer=NoEncodeTokenizer(),
                seq_len=2,
                dp_rank=0,
                dp_world_size=1,
                infinite=False,
            )

            input_dict, labels = next(iter(ds))

        assert torch.equal(input_dict["input"], torch.tensor([10, 12]))
        assert torch.equal(input_dict["positions"], torch.tensor([0, 1]))
        assert torch.equal(labels, torch.tensor([12, 13]))

    def test_text_iterator_still_tokenizes_strings(self):
        DATASETS["unit_text"] = DatasetConfig(
            path="unused",
            loader=lambda path: Dataset.from_dict({"text": ["hello"]}),
            sample_processor=lambda sample: sample["text"],
        )
        tokenizer = DummyTokenizer(tokens=[21, 22, 23])
        ds = HuggingFaceTextDataset(
            dataset_name="unit_text",
            dataset_path=None,
            tokenizer=tokenizer,
            seq_len=2,
            dp_rank=0,
            dp_world_size=1,
            infinite=False,
        )

        input_dict, labels = next(iter(ds))

        assert tokenizer.encode_calls == [
            (("hello",), {"add_bos": True, "add_eos": True})
        ]
        assert torch.equal(input_dict["input"], torch.tensor([21, 22]))
        assert torch.equal(labels, torch.tensor([22, 23]))


if __name__ == "__main__":
    unittest.main()
