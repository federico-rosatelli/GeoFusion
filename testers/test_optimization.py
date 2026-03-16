import numpy as np
import torch
from src.io.loader import load_constellaration_dataset
from src.optimization.optimizer import optimize_stellarator
from src.ml.ensamble import StellaratorEnsemble
import json
import random
import time
import matplotlib.pyplot as plt


def getDataset():
    return load_constellaration_dataset()


def getRandomData(dataset):
    return dataset[random.randint(0, len(dataset) - 1)]


def test_models():
    print("Testing model loading and initialization...")

    # with open("testers/data_test.json", "r") as f:
    #     data = json.load(f)
    dataset = getDataset()
    data = getRandomData(dataset)
    
    models = StellaratorEnsemble("configs/conf.yaml", "configs/model_struct.json")
    models.load_models()
    
    R_mn = np.array(data['boundary.r_cos'])
    Z_mn = np.array(data['boundary.z_sin'])

    input_vector = np.concatenate([R_mn.flatten(), Z_mn.flatten()])
    input_vector = torch.tensor(input_vector, dtype=torch.float32)
    input_vector = input_vector.unsqueeze(0)
    
    ai_input = input_vector.to("cpu").float()
    start_time = time.time()
    preds = models.predict(ai_input)
    #print(preds)
    print(f"Prediction Time: {time.time() - start_time}")

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


def testTime(iterations=100):

    print("Testing model time...")

    dataset = getDataset()
    
    
    models = StellaratorEnsemble("configs/conf.yaml", "configs/model_struct.json")
    models.load_models()
    
    times = []
    preds = {
        "qi": [],
        "iota_edge": [],
        "w_mhd": [],
        "mirror_ratio": []
    }

    for i in range(iterations):
        print(f"Testing iteration {i+1}/{iterations}", end="\r")
        data = getRandomData(dataset)
        if not data['metrics.qi']:
            continue
        R_mn = np.array(data['boundary.r_cos'])
        Z_mn = np.array(data['boundary.z_sin'])

        input_vector = np.concatenate([R_mn.flatten(), Z_mn.flatten()])
        input_vector = torch.tensor(input_vector, dtype=torch.float32)
        input_vector = input_vector.unsqueeze(0)
        
        ai_input = input_vector.to("cpu").float()
        start_time = time.time()
        _ = models.predict(ai_input)
        times.append(time.time() - start_time)

    print(f"Average Prediction Time: {np.mean(times)}")
    print(f"Std Dev Prediction Time: {np.std(times)}")
    
    plt.figure(figsize=(8, 5))
    plt.hist(times, bins=20, color='skyblue', edgecolor='black')
    plt.title("Prediction Times")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.axvline(np.mean(times), color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {np.mean(times):.5f}s')
    plt.legend()
    plt.tight_layout()
    plt.savefig("prediction_times.png")
    plt.close()