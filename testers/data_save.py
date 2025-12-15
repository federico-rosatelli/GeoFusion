import json
from src.io import loader as data_loader


def save_json(path:str="data/dataset_json.json"):
    dataset = data_loader.load_constellaration_dataset()
    data = dataset[0]
    with open(path, "w") as wrjs:
        json.dump(data,wrjs, indent=4)