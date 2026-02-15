import os
from datasets import load_dataset

def load_constellaration_dataset(split:str="train", cache_dir="data/.cache_dir"):

    cache_dir = os.path.expanduser(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    dataset = load_dataset("proxima-fusion/constellaration", split=split, cache_dir=cache_dir)
    return dataset
    
