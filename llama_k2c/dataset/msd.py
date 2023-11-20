import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import jsonlines
PROMPT = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"

class MSD_Dataset(Dataset):
    def __init__(self, data_path, split):
        self.data_dir = os.path.join(data_path, "msd")
        self.split = split
        msd_pretrain = load_dataset("seungheondoh/MSD-enrich")
        self.dataset = [i for i in msd_pretrain["train"] if i["tag"]]
        print(len(self.dataset))
        self.instruction_dict = {
            "singular":"write a single sentence that summarize a song with the following single attribute. Do not write lyrics information. Do not write artist name or album name.",
            "plural":"write a single sentence that summarize a song with the following attributes. Do not write lyrics information. Don't write artist name or album name.",
            }

    
    def __getitem__(self, index):
        item = self.dataset[index]
        track_id = item['track_id']
        tag_list = item['tag']
        if len(tag_list) > 1:
            inst = self.instruction_dict['plural']
        else:
            inst = self.instruction_dict['singular']
        tag_list = ", ".join(tag_list)
        user_prompt = PROMPT.format_map({"input":tag_list, "instruction": inst})
        return track_id, user_prompt, tag_list

    def __len__(self):
        return len(self.dataset)