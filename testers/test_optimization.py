import numpy as np
import torch
from src.io.loader import load_constellaration_dataset
from src.optimization.optimizer import optimize_stellarator
from src.ml.ensamble import StellaratorEnsemble
import json
import random

def test_optimization():
    print("Loading dataset...")
    dataset = load_constellaration_dataset()
    if not dataset:
        print("No dataset found.")
        return

    sample_config = dataset[0]
    print("Starting optimization test...")
    
    
    _ = optimize_stellarator(sample_config, max_iter=5)
    
    print("Optimization test complete.")


def getRandomData():
    dataset = load_constellaration_dataset()
    return dataset[random.randint(0, len(dataset) - 1)]


def test_models():
    print("Testing model loading and initialization...")

    # with open("testers/data_test.json", "r") as f:
    #     data = json.load(f)
    data = getRandomData()
    
    models = StellaratorEnsemble("configs/conf.yaml", "configs/model_struct.json")
    models.load_models()
    
    R_mn = np.array(data['boundary.r_cos'])
    Z_mn = np.array(data['boundary.z_sin'])

    input_vector = np.concatenate([R_mn.flatten(), Z_mn.flatten()])
    input_vector = torch.tensor(input_vector, dtype=torch.float32)
    input_vector = input_vector.unsqueeze(0)
    
    ai_input = input_vector.to("cpu").float()
    preds = models.predict(ai_input)
    print(preds)

    qi_val = preds["qi"].item()    
    iota_val = preds['iota_edge'].item()
    well_val = preds['w_mhd'].item()
    mr_val = preds['mirror_ratio'].item()

    qi_error = abs((qi_val - data['metrics.qi']) / data['metrics.qi'])
    iota_error = abs((iota_val - data['metrics.edge_rotational_transform_over_n_field_periods']) / data['metrics.edge_rotational_transform_over_n_field_periods'])
    well_error = abs((well_val - data['metrics.vacuum_well']) / data['metrics.vacuum_well'])
    mr_error = abs((mr_val - data['metrics.edge_magnetic_mirror_ratio']) / data['metrics.edge_magnetic_mirror_ratio'])

    qi_accuracy = (1 - qi_error) * 100
    iota_accuracy = (1 - iota_error) * 100
    well_accuracy = (1 - well_error) * 100
    mr_accuracy = (1 - mr_error) * 100

    print(f"QI Accuracy: {qi_accuracy}")
    print(f"IOTA Accuracy: {iota_accuracy}")
    print(f"W_MHD Accuracy: {well_accuracy}")
    print(f"Mirror Ratio Accuracy: {mr_accuracy}")

    
