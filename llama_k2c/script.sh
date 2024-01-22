python inference.py --model_name ../models/7B --peft_model ../models/k2c_lora2 --dataset_name audioset --dataset_split balanced_train
python inference.py --model_name ../models/7B --peft_model ../models/k2c_lora2 --dataset_name audioset --dataset_split unbalanced_train
python inference.py --model_name ../models/7B --peft_model ../models/k2c_lora2 --dataset_name msd --dataset_split all
python inference.py --model_name ../models/7B --peft_model ../models/k2c_lora2 --dataset_name music4all --dataset_split all
python inference.py --model_name ../models/7B --peft_model ../models/k2c_lora2 --dataset_name fma --dataset_split all