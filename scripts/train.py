import argparse

from tinydet.training.trainer import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train on Modal GPU")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(train(args.config))
