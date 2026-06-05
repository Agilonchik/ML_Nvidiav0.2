import tensorflow as tf
import pandas as pd
from pathlib import Path
import numpy as np
import sys

class LossDecompositionCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data, loss_fn, log_path="artifacts/loss_log.csv"):
        super().__init__()
        self.val_data = val_data
        # Сохраняем ссылку на функцию лосса
        self.loss_fn = loss_fn
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n[DEBUG] Колбэк анализатора инициализирован. Ждем конца эпохи...")

    def on_epoch_end(self, epoch, logs=None):
        print(f"\n--- 🛰️ ЗАПУСК АНАЛИЗА ЛОССА (Эпоха {epoch+1}) ---")
        logs = logs or {}
        
        try:
            # 1. Получаем батч данных
            # Мы используем .take(1), чтобы не сбить итератор основного обучения
            for x_batch, y_batch in self.val_data.take(1):
                x_val, y_true = x_batch, y_batch
                break
            
            # 2. Инференс
            y_pred = self.model(x_val, training=False)
            
            # 3. ВЫЗОВ ФУНКЦИИ ЛОССА (С обходом обертки Keras)
            # Пытаемся вызвать напрямую, а если нет - ищем атрибут .fn (где Keras прячет оригинал)
            target_fn = self.loss_fn
            if hasattr(self.loss_fn, 'fn'):
                target_fn = self.loss_fn.fn
            elif hasattr(self.loss_fn, 'target_fn'): # Для некоторых кастомных оберток
                target_fn = self.loss_fn.target_fn

            # Вызываем с нашим спец. аргументом
            raw_components = target_fn(y_true, y_pred, return_components=True)
            
            # 4. Преобразование в числа
            safe_components = {
                k: float(v.numpy()) if hasattr(v, "numpy") else float(v) 
                for k, v in raw_components.items()
            }
            
            # Добавляем стандартные метрики из логов обучения
            safe_components['epoch'] = epoch + 1
            safe_components['total_val_loss'] = float(logs.get('val_loss', 0))
            safe_components['masked_acc'] = float(logs.get('masked_accuracy', 0))
            
            # 5. Сохранение в append-режиме: новые эпохи дописываются к прошлым
            # запускам, а файл с историей больше не перезаписывается.
            row_df = pd.DataFrame([safe_components])
            file_exists = self.log_path.exists() and self.log_path.stat().st_size > 0
            row_df.to_csv(
                self.log_path,
                mode='a',
                header=not file_exists,
                index=False,
            )
            print(f"✅ УСПЕХ: Данные лосса добавлены в {self.log_path}")
            
        except Exception as e:
            # Выводим ошибку максимально заметно!
            print("\n" + "!"*60)
            print(f"❌ ОШИБКА В LOSS CALLBACK: {str(e)}")
            print("Тип ошибки:", type(e))
            # Если это TypeError, значит Keras всё еще блокирует аргумент return_components
            print("!"*60 + "\n")
