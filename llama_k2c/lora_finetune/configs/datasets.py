from dataclasses import dataclass    
@dataclass
class mapc_dataset:
    dataset: str =  "mapc_dataset"
    train_split: str = "train"
    test_split: str = "val"
    input_length: int = 1024