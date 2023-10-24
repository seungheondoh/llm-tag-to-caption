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
    inputs = instance['inputs']
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "user", "content": inputs}
            ]
    )
    
    results = completion['choices'][0]['message']['content']

    print("query: ")
    print(inputs)
    print("-"*10)
    print("results: ")
    print(results)

    with open(f"./sample.txt", 'w') as file:
        file.write(results)

class OpenAIGpt:
    def __init__(self, prompt):
        load_dotenv()    
        self.prompt = prompt
        self.prompt_dict ={
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
            "attribute_prediction": {
                "singular":"write the answer as a python dictionary with new_attribute and description as keys. for new_attribute, write new attributes with high co-occurrence with the following single attribute. for description, write a song description sentence including the single attribute and new attribute.",
                "plural":"write the answer as a python dictionary with new_attribute and description as keys. for new_attribute, write new attributes with high co-occurrence with the following attributes. for description, write a song description sentence including the following attributes and new attributes.",
                }
            }

    def run(self, tags):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        instance = {}
        text = tags
        instance["text"] = text
        if len(tags) > 1:
            instruction = self.prompt_dict[self.prompt]["plural"]
        else:
            instruction = self.prompt_dict[self.prompt]["singular"]
        instance["inputs"] = f'{instruction} \n {text}'
        api_helper(instance)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", default="fast tempo, male singer, piano, love song, passionate", type=str)
    parser.add_argument("--prompt", default="writing", type=str)
    args = parser.parse_args()
    openai_gpt = OpenAIGpt(
        prompt = args.prompt, 
        )
    openai_gpt.run(args.tags)