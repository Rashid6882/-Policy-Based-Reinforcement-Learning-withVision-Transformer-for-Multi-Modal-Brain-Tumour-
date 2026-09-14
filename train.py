import argparse
from training.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, choices=["cnn", "vit"], default="cnn")
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()

    config_path = f"configs/{args.model_type}_dqn.yaml"
    train(config_path, episodes_override=args.episodes)
