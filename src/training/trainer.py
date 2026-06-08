import tensorflow as tf
from pathlib import Path
import logging
import math

# Импортируем наш новый модуль аналитики
from src.utils.callbacks import LossDecompositionCallback


class Trainer:
    def __init__(self, model: tf.keras.Model, config: dict, logger: logging.Logger):
        self.model = model
        self.config = config
        self.logger = logger
        self.epochs = config["training"]["epochs"]
        self.checkpoints_dir = Path(config["paths"]["checkpoints_dir"])
        self.logs_dir = Path(config["paths"]["logs_dir"])
        self.models_dir = Path(config["paths"]["models_dir"])

    def _get_callbacks(self, val_ds: tf.data.Dataset = None) -> list:
        ckpt_path = self.checkpoints_dir / "best_model.keras"
        csv_path = self.logs_dir / "training_log.csv"
        tb_path = self.logs_dir / "tensorboard"

        monitor_metric = "val_loss" if val_ds is not None else "loss"

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(ckpt_path),
                monitor=monitor_metric,
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor_metric,
                patience=self.config["training"]["early_stopping_patience"],
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor_metric,
                factor=self.config["training"]["reduce_lr_factor"],
                patience=self.config["training"]["reduce_lr_patience"],
                verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(str(csv_path), append=True),
            tf.keras.callbacks.TensorBoard(log_dir=str(tb_path), histogram_freq=1),
        ]

        # --- ДОБАВЛЯЕМ LOSS DECOMPOSITION ЕСЛИ ЕСТЬ ВАЛИДАЦИЯ ---
        if val_ds is not None:
            analyzer = LossDecompositionCallback(
                val_data=val_ds,
                loss_fn=self.model.loss,  # Наш кастомный лосс из ModelBuilder
                log_path="artifacts/loss_log.csv",
            )
            callbacks.append(analyzer)
            self.logger.info("📈 LossDecompositionCallback успешно добавлен в очередь.")

        return callbacks

    def train(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        train_count: int,
        val_count: int,
    ):
        self.logger.info("Starting U-Net training process...")

        if train_count <= 0:
            raise ValueError(
                "Training dataset is empty: no JSON files with matching images found."
            )

        batch_size = self.config["data"]["batch_size"]
        steps_per_epoch = max(1, math.ceil(train_count / batch_size))
        validation_steps = (
            max(1, math.ceil(val_count / batch_size)) if val_count > 0 else None
        )
        validation_data = val_ds.repeat() if val_count > 0 else None

        # Передаем val_ds в метод получения колбэков
        active_callbacks = self._get_callbacks(val_ds if val_count > 0 else None)

        history = self.model.fit(
            train_ds.repeat(),
            validation_data=validation_data,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=active_callbacks,  # Используем подготовленный список
        )

        final_model_path = self.models_dir / "final_model.keras"
        self.model.save(str(final_model_path))
        self.logger.info(f"Training completed. Final model saved to {final_model_path}")
        return history
