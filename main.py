import argparse
from src.ml.manager import NeuralManager


def main(args):
    manager = NeuralManager(config_path=args.config, struct_path=args.struct, force_retrain=args.force)
    
    models = manager.getModels()

    for model in models.models:
        models.plot_loss(model)
    
    


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
        "--force",
        action="store_true",
        help="Force retraining even if models exist (overrides config)."
    )
    
    args = parser.parse_args()
    main(args)