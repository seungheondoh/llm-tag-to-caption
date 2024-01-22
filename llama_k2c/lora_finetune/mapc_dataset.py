# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
# Reference: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
import os
import random
import torch
from datasets import load_dataset
from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset
from typing import List

PROMPT = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"

class MAPCDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train", max_words=385):
        self.partition = partition
        self.source = [
                        # wavcaps
                        'audioset_sl',
                        'bbc_sound_effects',
                        'freesound',
                        'soundbible',
                        # lp-music-caps
                        'million_song_dataset',
                        'magnatagatune',
                        'music_caps',
                    ]
        if partition == "train":
            dataset = load_dataset("seungheondoh/music-audio-pseudo-captions", split="unbalanced_sample")
            self.ann =  [item for item in dataset if "lyrics" not in item['output']]
        else:
            self.ann = load_dataset("seungheondoh/music-audio-pseudo-captions", split="balanced_test")
        self.max_words = max_words
        self.tokenizer = tokenizer

    def __len__(self):
        if self.partition == "train":
            return len(self.ann) // 10
        else:
            return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        prompt = PROMPT.format_map(ann)
        example = prompt + ann["output"]
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = 0
        example_mask = example_mask.float()
        label_mask = label_mask.float()

        return {
            "input_ids": example,
            "labels": labels,
            "attention_mask":example_mask,
        }
