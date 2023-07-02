import os
import openai
import argparse
import json
import argparse
import random
from dotenv import load_dotenv
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from time import sleep

def api_helper(instance):
    text = instance['text']
    split = instance['split']
    inputs = instance['inputs']
    prompt = instance['prompt']
    dataset_type = instance['dataset_type']
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "user", "content": inputs}
            ]
    )

    results = completion['choices'][0]['message']['content']
    if split == "TRAIN":
        partition = instance['partition']
        os.makedirs(f"./{dataset_type}/{prompt}/{split}/{partition}", exist_ok=True)
        with open(f"./{dataset_type}/{prompt}/{split}/{partition}/{instance['_id']}.txt", 'w') as file:
            file.write(results)
    else:
        os.makedirs(f"./{dataset_type}/{prompt}/{split}", exist_ok=True)
        with open(f"./{dataset_type}/{prompt}/{split}/{instance['_id']}.txt", 'w') as file:
            file.write(results)

    
class OpenAIGpt:
    def __init__(self, split, prompt, dataset_type, n_iter=True, partition=0):
        load_dotenv()    
        self.split = split
        self.partition = partition
        self.prompt = prompt
        self.dataset_type = dataset_type
        if self.dataset_type == "msd":
            self.annotation= json.load(open("../dataset/ecals_annotation/annotation.json", 'r'))
            self.track_split= json.load(open("../dataset/ecals_annotation/ecals_track_split.json", 'r'))
        elif self.dataset_type == "mtat":
            self.annotation = json.load(open("../dataset/mtat/codified_annotation.json", 'r'))
            self.track_split = json.load(open("../dataset/mtat/codified_track_split.json", 'r'))
        self.prompt_dict = {
            "writing": {
                "singular":"write a song description sentence including the following single attribute.",
                "plural":"write a song description sentence including the following attributes.",
                },
            "summary": {
                "singular":"write a single sentence that summarize a song with the following single attribute. Don't write artist name or album name.",
                "plural":"write a single sentence that summarize a song with the following attributes. Don't write artist name or album name.",
                },
            "paraphrase": {
                "singular":"write a song description sentence including the following single attribute. paraphraze paraphrasing is acceptable.",
                "plural":"write a song description sentence including the following attributes. paraphraze paraphrasing is acceptable.",
                },
            "prediction_attribute": {
                "singular":"write a song description sentence including the following single attribute.",
                "plural":"write the answer as a python dictionary with new_attribute and description as keys. for new_attribute, write new attributes with high co-occurrence with the following attributes. for description, write a song description sentence including the following attributes and new attributes.",
                }
            }
        if split == "TRAIN":
            if self.dataset_type == "msd":
                train_track = self.track_split['train_track'] + self.track_split['extra_track']            
                target_track = train_track[partition * 10000 : (partition + 1) * 10000]
            elif self.dataset_type == "mtat":
                target_track = self.track_split['train_track']
        elif split == "VALID":
            target_track = self.track_split['valid_track']
        else:
            target_track = self.track_split['test_track']

        if n_iter:
            self.get_already_download()
            target_track = list(set(target_track).difference(self.already_download))
        self.fl_dict = {i : self.annotation[i] for i in target_track}
        print(len(self.fl_dict))
        
    def get_already_download(self):
        if self.split == "TRAIN":
            save_path = f"./{self.dataset_type}/{self.prompt}/{self.split}/{self.partition}"
        else:
            save_path = f"./{self.dataset_type}/{self.prompt}/{self.split}"
        self.already_download = set([i.replace(".txt", "")for i in os.listdir(save_path)])
        print("already_download: ", len(self.already_download))

    def run(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        inputs = []
        
        if len(self.fl_dict) > 0:
            for _id, instance in self.fl_dict.items():
                instance['_id'] = _id
                instance['split'] = self.split
                instance['partition'] = self.partition
                instance['prompt'] = self.prompt
                instance['dataset_type'] = self.dataset_type
                if self.dataset_type == "msd":
                    tags = instance["tag"]
                elif self.dataset_type == "mtat":
                    tags = instance['extra_tag']
                text = ", ".join(tags)
                instance["text"] = text
                if len(tags) > 1:
                    instruction = self.prompt_dict[self.prompt]["plural"]
                elif len(tags) == 0:
                    # No annotation tag case
                    continue
                else:
                    instruction = self.prompt_dict[self.prompt]["singular"]
                instance["inputs"] = f'{instruction} \n {text}'
                inputs.append(instance)
            
            with ThreadPoolExecutor() as pool:
                tqdm(pool.map(api_helper, inputs))
            print("finish")
        else:
            print("already finished")
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", default="msd", type=str)
    parser.add_argument("--split", default="TRAIN", type=str)
    parser.add_argument("--prompt", default="writing", type=str)
    parser.add_argument("--n_iter", default=False, type=bool)
    parser.add_argument("--partition", default=0, type=int)
    args = parser.parse_args()

    openai_gpt = OpenAIGpt(
        split = args.split, 
        prompt = args.prompt, 
        dataset_type = args.dataset_type,
        n_iter = args.n_iter, 
        partition = args.partition
        )
    openai_gpt.run()
    # python main.py --prompt short --partition 4