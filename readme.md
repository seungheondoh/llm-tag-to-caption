# Tag-to-Caption Augmentation using Large Language Model

This project aims to generate captions for music using existing tags. 

> [**TTMR++: Enriching Music Descriptions with a Finetuned-LLM and Metadata for Text-to-Music Retrieval**](#)   
> SeungHeon Doh, Minhee Lee, Dasaem Jeong, Juhan Nam
> Submitted to ICASSP 2024

## LLaMA-7B Finetune

1. Download pretrain LLaMA weight from [LLaMa Access](https://ai.meta.com/resources/models-and-libraries/llama-downloads/), and move `7B` weight in `models/7B`

2. Inference with huggingface dataset
```
cd llama_k2c
python inference.py --model_name ../models/7B --peft_model ../models/k2c_lora --dataset_name msd --dataset_split all --
```


> [**LP-MusicCaps: LLM-Based Pseudo Music Captioning**](#)   
> SeungHeon Doh, Keunwoo Choi, Jongpil Lee, Juhan Nam   
> ISMIR 2023   

### License
This project is licensed under the MIT License.

### Acknowledgements
We would like to thank OpenAI for providing the GPT-3.5 Turbo API, which powers this project.

### Contact
For any questions or inquiries, please contact seungheondoh@kaist.ac.kr.