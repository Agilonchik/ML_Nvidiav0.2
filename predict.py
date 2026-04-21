import argparse
import os
from src.utils.config_parser import load_config
from src.utils.logger import setup_logger
from src.inference.predictor import Predictor
from pathlib import Path

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    parser = argparse.ArgumentParser(description="U-Net Semantic Segmentation Inference")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--folder", type=str, help="Path to a folder of images")
    args = parser.parse_args()

    config = load_config()
    logger = setup_logger("PredictorLogger", Path(config["paths"]["logs_dir"]))

    if not args.image and not args.folder:
        logger.error("Please provide either --image or --folder.")
        return

    try:
        predictor = Predictor(config, logger)

        if args.image:
            logger.info(f"Running inference on: {args.image}")
            predictor.predict_single(args.image)

        if args.folder:
            logger.info(f"Running inference on folder: {args.folder}")
            predictor.predict_folder(args.folder)

    except Exception as e:
        logger.exception(f"Fatal error during inference: {e}")

if __name__ == "__main__":
    main()
