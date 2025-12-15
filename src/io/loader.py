from datasets import load_dataset

def load_constellaration_dataset(split:str="train", cache_dir="data/.cache_dir"):

    dataset = load_dataset("proxima-fusion/constellaration", split=split, cache_dir=cache_dir)
    return dataset
    
