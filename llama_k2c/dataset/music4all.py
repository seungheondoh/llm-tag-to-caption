import os
import jsonlines
from datasets import load_dataset
import pandas as pd
from torch.utils.data import Dataset
PROMPT = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"

class Music4all_Dataset(Dataset):
    def __init__(self, data_path, split):
        self.id_genres = pd.read_csv(os.path.join(data_path, "music4all","id_genres.csv"),header=0, sep='	').set_index("id")
        id_information = pd.read_csv(os.path.join(data_path, "music4all","id_information.csv"),header=0, sep='	').set_index("id")
        self.id_tags = pd.read_csv(os.path.join(data_path, "music4all","id_tags.csv"),header=0, sep='	').set_index("id")

        train = self.load_jsonlines(os.path.join(data_path, "music4all", "metadata", "train.jsonl"))
        test = self.load_jsonlines(os.path.join(data_path, "music4all", "metadata", "test.jsonl"))
        already_download = [i["fname"] for i in train + test]
        self.id_information = id_information.drop(already_download, axis="index")
        
        self.instruction_dict = {
            "singular":"write a single sentence that summarize a song with the following single attribute. Don't write lyrics. Don't write artist name or album name.",
            "plural":"write a single sentence that summarize a song with the following attributes. Don't write lyrics. Don't write artist name or album name.",
            }
        
    def load_jsonlines(self, data_file):
        datas = []
        with jsonlines.open(data_file) as f:
            for line in f.iter():
                datas.append(line)
        return datas
        
    def __getitem__(self, index):
        item = self.id_information.iloc[index]
        _id = item.name
        genre = [i.strip() for i in self.id_genres.loc[_id]["genres"].split(",")]
        tags = [i.strip() for i in self.id_tags.loc[_id]["tags"].split(",")]
        input_tags = list(set(genre + tags))
        if len(input_tags) > 1:
            inst = self.instruction_dict['plural']
        else:
            inst = self.instruction_dict['singular']
        tag_list = ", ".join(input_tags)
        user_prompt = PROMPT.format_map({"input":tag_list, "instruction": inst})
        return _id, user_prompt, tag_list
    
    def __len__(self):
        return len(self.id_information)