# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import fire
import torch
import os
import sys
import time
from typing import List
from datasets import load_dataset
from transformers import LlamaTokenizer
from model_utils import load_model, load_peft_model, load_llama_from_config
import json
from datasets import load_dataset
from tqdm import tqdm
import re
import pandas as pd
import jsonlines
from dataset import AudiosetMusic_Dataset, Music4all_Dataset, FMA_Dataset, MSD_Dataset

def text_clearning(text):
    pattern = r"[^a-zA-Z0-9\s]"
    removed_text = re.sub(pattern, "", text)
    return removed_text

PROMPT = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"

def main(
    model_name,
    peft_model,
    dataset_name,
    dataset_split,
    device: str="auto",
    data_path: str="../../dataset",
    quantization: bool=True,
    max_new_tokens =512, #The maximum numbers of tokens to generate
    prompt_file: str=None,
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt + min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    enable_azure_content_safety: bool=False, # Enable safety check with Azure content safety api
    enable_sensitive_topics: bool=False, # Enable check for sensitive topics using AuditNLG APIs
    enable_saleforce_content_safety: bool=True, # Enable safety check woth Saleforce safety flan t5
    **kwargs
):
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    model = load_model(model_name, quantization, device)
    tokenizer = LlamaTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token_id = tokenizer.bos_token_id 
    if peft_model:
        model = load_peft_model(model, peft_model, device)
    model.eval()
    
    dataset = None
    if dataset_name == "audioset":
        dataset = AudiosetMusic_Dataset(data_path=data_path, split=dataset_split)
    elif dataset_name == "music4all":
        dataset = Music4all_Dataset(data_path= data_path, split=dataset_split)
    elif dataset_name == "fma":
        dataset = FMA_Dataset(data_path= data_path, split=dataset_split)
    elif dataset_name == "msd":
        dataset = MSD_Dataset(data_path= data_path, split=dataset_split)
    # tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    model.resize_token_embeddings(model.config.vocab_size + 1)
    print("inference size == ", len(dataset))
    dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=False,
            num_workers=24, pin_memory=True, drop_last=False
        )
    for idx, item in enumerate(tqdm(dataloader)):
        fnames, prompts, input_tags = item
        start = time.perf_counter()
        encodings = tokenizer(prompts, padding=True, truncation=True, max_length=256, return_tensors="pt").to('cuda')
        with torch.no_grad():
            outputs = model.generate(
                **encodings,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs 
            )

        e2e_inference_time = (time.perf_counter()-start)*1000
        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        ##### for inference monitar
        print(f"the inference time is {e2e_inference_time} ms")
        print(str(fnames[0]))
        print(input_tags[0])
        print(output_text[0].split("### Response:")[-1].strip())
        print("="*50)
        ##### 
        inferences = []
        for fname, input_tag, text in zip(fnames, input_tags, output_text):
            text = text.split("### Response:")[-1].strip()
            inferences.append({
                "fname": str(fname),
                "tag_list": input_tag,
                "pseudo_caption": text
            })
        os.makedirs(f"./{dataset_name}", exist_ok=True)
        with open(os.path.join(f"./{dataset_name}/{dataset_split}_{idx}.jsonl"), encoding= "utf-8",mode="w") as f: 
            for i in inferences: f.write(json.dumps(i) + "\n")

if __name__ == "__main__":
    fire.Fire(main)