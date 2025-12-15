import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.io.loader import load_constellaration_dataset
from src.optimization.optimizer import optimize_stellarator

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
    
if __name__ == "__main__":
    test_optimization()
