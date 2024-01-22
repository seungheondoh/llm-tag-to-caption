# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
from dataclasses import dataclass
from typing import ClassVar

@dataclass
class train_config:
    model_name: str="../models_hf/7B"
    enable_fsdp: bool= False 
    run_validation: bool= True
    batch_size_training: int=12
    num_epochs: int=2
    num_workers_dataloader: int=12
    lr: float=2e-5
    weight_decay: float=0.0
    gamma: float= 0.99
    seed: int=42
    use_fp16: bool=True
    mixed_precision: bool=True
    val_batch_size: int=1
    dataset = "mapc_dataset"
    micro_batch_size: int=12
    peft_method: str = "lora" # None , llama_adapter, prefix
    use_peft: bool=True
    output_dir: str = "../models_hf/cap-llama-7B"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = True
    one_gpu: bool = False
    save_model: bool = True