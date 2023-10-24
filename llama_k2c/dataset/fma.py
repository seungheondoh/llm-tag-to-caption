import os
import jsonlines
from datasets import load_dataset
import pandas as pd
from torch.utils.data import Dataset
import torch
PROMPT = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"

class FMA_Dataset(Dataset):
    def __init__(self, data_path, split):
        # dataset = self.load_jsonlines(os.path.join(data_path, "fma_large","annotation.jsonl"))
        dataset = torch.load("../temporal/fma_error.pt")
        self.dataset = dataset['train'] + dataset['valid'] + dataset['test']
        self.instruction_dict = {
            "singular":"write a single sentence that summarize a song with the following single attribute. Don't write lyrics information. Don't write artist name or album name.",
            "plural":"write a single sentence that summarize a song with the following attributes. Don't write lyrics information. Don't write artist name or album name.",
            }
        print(len(self.dataset))
    def load_jsonlines(self, data_file):
        datas = []
        with jsonlines.open(data_file) as f:
            for line in f.iter():
                datas.append(line)
        return datas

        
    def __getitem__(self, index):
        item = self.dataset[index]
        _id = str(item['track_id'])
        input_tags = item['tag']
        if len(input_tags) > 1:
            inst = self.instruction_dict['plural']
        else:
            inst = self.instruction_dict['singular']
        tag_list = ", ".join(input_tags)
        user_prompt = PROMPT.format_map({"input":tag_list, "instruction": inst})
        return _id, user_prompt, tag_list
    
    def __len__(self):
        return len(self.dataset)
    