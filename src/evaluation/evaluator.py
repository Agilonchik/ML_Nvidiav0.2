import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
import logging

class Evaluator:
    def __init__(self, model: tf.keras.Model, config: dict, logger: logging.Logger):
        self.model = model
        self.config = config
        self.logger = logger
        self.artifacts_dir = Path(config["paths"]["artifacts_dir"])

    def evaluate(self, val_ds: tf.data.Dataset, history):
        self.logger.info("Generating training history plots...")
        self._plot_history(history)
        self.logger.info("Evaluation complete.")

    def _plot_history(self, history):
        hist_df = pd.DataFrame(history.history)
        plt.figure(figsize=(12, 5))
        
        # График потерь (Loss)
        plt.subplot(1, 2, 1)
        plt.plot(hist_df['loss'], label='Train Loss')
        if 'val_loss' in hist_df.columns:
            plt.plot(hist_df['val_loss'], label='Val Loss')
        plt.title('Loss (Crossentropy)')
        plt.legend()
        
        # График точности (Умный поиск имени метрики)
        plt.subplot(1, 2, 2)
        # Keras 3 часто переименовывает метрики, ищем доступную
        acc_key = 'sparse_categorical_accuracy' if 'sparse_categorical_accuracy' in hist_df.columns else 'accuracy'
        val_acc_key = f'val_{acc_key}'
        
        if acc_key in hist_df.columns:
            plt.plot(hist_df[acc_key], label='Train Acc')
        if val_acc_key in hist_df.columns:
            plt.plot(hist_df[val_acc_key], label='Val Acc')
            
        plt.title('Pixel Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.artifacts_dir / "training_history.png")
        plt.close()