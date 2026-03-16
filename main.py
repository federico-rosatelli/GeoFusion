import argparse
from src.io.loader import load_constellaration_dataset
from src.ml.manager import NeuralManager
from src.physics import geometry
from testers import test_optimization
from src.visualization.plotting import plot_stellarator_shape
import json


from src.visualization.plot_accuracy import (
    plot_multi_accuracy_bars, 
    plot_single_accuracy_bars, 
    plot_accuracy_comparison
    )

def main(args):

    
    if args.test:
        test_optimization.test_models()
        test_optimization.testTime(1000)


    if args.train:
        manager = NeuralManager(config_path=args.config, struct_path=args.struct, force_retrain=args.force)
        manager.train_all()
    
    if args.plot:

        with open("/home/fede/Downloads/stellarator_optim_0.json", "r") as f:
            config = json.load(f)

        conf = geometry.get_surface_coordinates(config)

        plot_stellarator_shape(conf["X"], conf["Y"], conf["Z"], "002")
        
        plot_single_accuracy_bars()
        plot_multi_accuracy_bars()
        plot_accuracy_comparison()    
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Stellarator Surrogate Models")
    
    parser.add_argument(
        "--config",
        type=str, 
        default="configs/conf.yaml",
        help="Path to the training configuration YAML file."
    )
    parser.add_argument(
        "--struct",
        type=str,
        default="configs/model_struct.json",
        help="Path to the model architecture JSON file."
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run tests."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the models."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining even if models exist (overrides config)."
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot loss history."
    )
    
    args = parser.parse_args()
    main(args)