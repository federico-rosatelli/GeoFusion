import argparse
from src.ml.manager import NeuralManager


def main(args):

    manager = NeuralManager(config_path=args.config, struct_path=args.struct, force_retrain=args.force)
    
    if args.train:
        manager.train_all()
    
    if args.plot:
        manager.plot_all()
    
    
    


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