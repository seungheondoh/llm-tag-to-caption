from datasets import load_dataset
import pandas as pd
from torch.utils.data import Dataset
PROMPT = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"

class AudiosetMusic_Dataset(Dataset):
    def __init__(self, data_path, split):
        self.data_path = data_path
        self.split = split
        self.as_music = load_dataset("seungheondoh/audioset-music")
        self.dataset = self.as_music[split]
        self.instruction_dict = {
            "singular":"write a single sentence that summarize a song with the following single attribute. Do not write artist name or album name.",
            "plural":"write a single sentence that summarize a song with the following attributes. Do not write artist name or album name.",
            }
    
    def __getitem__(self, index):
        item = self.dataset[index]
        _id = item["ytid"]
        input_tags = item["all_tags"]
        if len(input_tags) > 1:
            inst = self.instruction_dict['plural']
        else:
            inst = self.instruction_dict['singular']
        tag_list = ", ".join(input_tags)
        user_prompt = PROMPT.format_map({"input":tag_list, "instruction": inst})
        return _id, user_prompt, tag_list

    def __len__(self):
        return len(self.dataset)