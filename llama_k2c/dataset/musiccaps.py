import os
import jsonlines
from datasets import load_dataset
import pandas as pd
from torch.utils.data import Dataset
PROMPT = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"

class MC_Dataset(Dataset):
    def __init__(self):
        musiccaps = load_dataset("seungheondoh/LP-MusicCaps-MC")
        self.dataset = musiccaps["test"]
        self.instruction_dict = {
            "singular":"write a single sentence that summarize a song with the following single attribute. Don't write artist name or album name.",
            "plural":"write a single sentence that summarize a song with the following attributes. Don't write artist name or album name.",
            }
        
    def __getitem__(self, index):
        item = self.dataset[index]
        _id = str(item['ytid'])
        input_tags = item['aspect_list']
        gt_caption = item['caption_ground_truth']
        gpt_caption = item['caption_summary']
        if len(input_tags) > 1:
            inst = self.instruction_dict['plural']
        else:
            inst = self.instruction_dict['singular']
        tag_list = ", ".join(input_tags)
        user_prompt = PROMPT.format_map({"input":tag_list, "instruction": inst})
        return _id, user_prompt, tag_list, gt_caption, gpt_caption
    
    def __len__(self):
        return len(self.dataset)
    