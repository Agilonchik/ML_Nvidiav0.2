import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import logging


class Evaluator:
    def __init__(self, model: tf.keras.Model, config: dict, logger: logging.Logger):
        self.model = model
        self.config = config
        self.logger = logger
        self.artifacts_dir = Path(config["paths"]["artifacts_dir"])
        self.logs_dir = Path(config["paths"]["logs_dir"])

    def evaluate(self, val_ds: tf.data.Dataset, history):
        self.logger.info("Generating training history plots...")
        self._plot_history(history)
        self.logger.info("Evaluation complete.")

    def _load_training_history(self, history) -> pd.DataFrame:
        csv_path = self.logs_dir / "training_log.csv"
        if csv_path.exists() and csv_path.stat().st_size > 0:
            self.logger.info(f"Loading accumulated training history from {csv_path}")
            return pd.read_csv(csv_path)

        self.logger.info(
            "Accumulated training log not found; using current in-memory history."
        )
        return pd.DataFrame(history.history)

    def _plot_history(self, history):
        hist_df = self._load_training_history(history)
        if hist_df.empty:
            self.logger.warning("Training history is empty; skipping history plot.")
            return

        plt.figure(figsize=(12, 5))
        x_axis = range(1, len(hist_df) + 1)

        # График потерь (Loss)
        plt.subplot(1, 2, 1)
        if "loss" in hist_df.columns:
            plt.plot(x_axis, hist_df["loss"], label="Train Loss")
        if "val_loss" in hist_df.columns:
            plt.plot(x_axis, hist_df["val_loss"], label="Val Loss")
        plt.title("Loss (Crossentropy)")
        plt.xlabel("Accumulated epoch")
        plt.legend()

        # График точности (Умный поиск имени метрики)
        plt.subplot(1, 2, 2)
        # Keras 3 часто переименовывает метрики, ищем доступную
        acc_key = (
            "sparse_categorical_accuracy"
            if "sparse_categorical_accuracy" in hist_df.columns
            else "accuracy"
        )
        val_acc_key = f"val_{acc_key}"

        if acc_key in hist_df.columns:
            plt.plot(x_axis, hist_df[acc_key], label="Train Acc")
        if val_acc_key in hist_df.columns:
            plt.plot(x_axis, hist_df[val_acc_key], label="Val Acc")

        plt.title("Pixel Accuracy")
        plt.xlabel("Accumulated epoch")
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.artifacts_dir / "training_history.png")
        plt.close()
