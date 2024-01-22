# Tag-to-Caption Augmentation using Large Language Model

This project aims to generate captions for music using existing tags. 

> [**TTMR++: Enriching Music Descriptions with a Finetuned-LLM and Metadata for Text-to-Music Retrieval**](#)   
> SeungHeon Doh, Minhee Lee, Dasaem Jeong, Juhan Nam
> ICASSP 2024

## LLaMA-7B Finetune

1. Download pretrain LLaMA weight from [LLaMa Access](https://ai.meta.com/resources/models-and-libraries/llama-downloads/), and move `7B` weight in `models/7B`

2. (Optional) Finetune LLaMa2 Model with Quantization + LoRA
```
cd llamak2c
python lora_finetune/llama_finetuning.py --use_peft --peft_method lora --quantization --model_name ../models/7B --output_dir ../models/k2c_lora2
```

3. Inference with huggingface dataset

```
cd llama_k2c
python inference.py --model_name ../models/7B --peft_model ../models/k2c_lora --dataset_name msd --dataset_split all --
```


### License
This project is licensed under the MIT License.


### Contact
For any questions or inquiries, please contact seungheondoh@kaist.ac.kr.