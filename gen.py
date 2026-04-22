import os
from pathlib import Path

def create_project():
    project_name = "image_classification_project" # Оставим имя папки прежним для совместимости
    base_path = Path(project_name)

    directories = [
        "configs", "data/train", "data/val", "models", "logs", 
        "artifacts", "checkpoints", "src/utils", "src/data", 
        "src/models", "src/training", "src/evaluation", "src/inference"
    ]

    files_content = {
        "requirements.txt": """tensorflow==2.15.0
keras==2.15.0
PyYAML==6.0.1
scikit-learn==1.4.0
matplotlib==3.8.2
pandas==2.2.0
numpy==1.26.3
opencv-python==4.9.0.80
""",

        "configs/config.yaml": """project:
  name: "unet_segmentation_v1"
  seed: 42

paths:
  data_dir: "data"
  models_dir: "models"
  logs_dir: "logs"
  artifacts_dir: "artifacts"
  checkpoints_dir: "checkpoints"

data:
  image_size: [224, 224] # Разрешение должно быть кратно 16 для U-Net
  batch_size: 16         # Уменьшен батч-сайз, сегментация ест больше памяти
  use_augmentation: false # Для генератора отключим сложные аугментации для стабильности
  cache: false

model:
  type: "unet"
  learning_rate: 0.001

training:
  epochs: 30
  early_stopping_patience: 7
  reduce_lr_patience: 3
  reduce_lr_factor: 0.5

inference:
  alpha: 0.5 # Прозрачность наложения маски (0 до 1)
""",

        "src/utils/logger.py": """import logging
import sys
from pathlib import Path

def setup_logger(name: str, log_dir: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.hasHandlers(): return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    log_file = log_dir / "project.log"
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
""",

        "src/utils/config_parser.py": """import yaml
from pathlib import Path

def load_config(config_path: str = "configs/config.yaml") -> dict:
    root_dir = Path(__file__).resolve().parents[2]
    full_config_path = root_dir / config_path
    if not full_config_path.exists(): raise FileNotFoundError(f"Config not found: {full_config_path}")
    with open(full_config_path, "r", encoding="utf-8") as f: config = yaml.safe_load(f)
    paths = config.get("paths", {})
    for key, folder in paths.items():
        if key != "data_dir":
            dir_path = root_dir / folder
            dir_path.mkdir(parents=True, exist_ok=True)
            config["paths"][key] = str(dir_path)
    config["paths"]["data_dir"] = str(root_dir / paths.get("data_dir", "data"))
    config["root_dir"] = str(root_dir)
    return config
""",

        # Генератор масок из JSON полигонов
        "src/data/dataset.py": """import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
import json
import logging

class DataPipeline:
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.data_dir = Path(config["paths"]["data_dir"])
        self.train_dir = self.data_dir / "train"
        self.val_dir = self.data_dir / "val"
        self.img_size = tuple(config["data"]["image_size"])
        self.batch_size = config["data"]["batch_size"]
        
        self.class_names = ["background"] # Индекс 0 всегда фон
        self.num_classes = 1

    def _find_image_for_json(self, json_path: Path):
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            img_path = json_path.with_suffix(ext)
            if img_path.exists(): return img_path
            img_path = json_path.with_suffix(ext.upper())
            if img_path.exists(): return img_path
        return None

    def _scan_classes(self):
        self.logger.info("Scanning for unique classes in JSON files...")
        classes = set()
        for j_path in list(self.train_dir.glob("*.json")) + list(self.val_dir.glob("*.json")):
            with open(j_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'shapes' in data: # Формат LabelMe
                    for shape in data['shapes']: classes.add(shape.get('label', ''))
        
        self.class_names.extend(sorted(list(classes)))
        self.num_classes = len(self.class_names)
        self.logger.info(f"Classes found ({self.num_classes}): {self.class_names}")
        self._save_class_names()

    def _data_generator(self, dir_path):
        json_files = list(Path(dir_path).glob("*.json"))
        
        def generator():
            for j_path in json_files:
                img_path = self._find_image_for_json(j_path)
                if not img_path: continue
                
                # Читаем картинку
                img = cv2.imread(str(img_path))
                if img is None: continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                orig_h, orig_w = img.shape[:2]
                
                # Создаем пустую маску фона (нули)
                mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                
                # Парсим полигоны и рисуем маску
                with open(j_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'shapes' in data:
                        for shape in data['shapes']:
                            label = shape.get('label', '')
                            if label in self.class_names:
                                class_idx = self.class_names.index(label)
                                points = np.array(shape.get('points', []), dtype=np.int32)
                                if len(points) >= 3:
                                    cv2.fillPoly(mask, [points], class_idx)
                
                # Ресайз
                img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)
                # Маску ресайзим с INTER_NEAREST, чтобы индексы классов не смешались в дроби!
                mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
                
                img = img.astype(np.float32)
                mask = np.expand_dims(mask, axis=-1).astype(np.int32)
                
                yield img, mask
        
        return generator

    def create_datasets(self):
        self._scan_classes()

        output_signature = (
            tf.TensorSpec(shape=(self.img_size[1], self.img_size[0], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(self.img_size[1], self.img_size[0], 1), dtype=tf.int32)
        )

        train_ds = tf.data.Dataset.from_generator(
            self._data_generator(self.train_dir), output_signature=output_signature
        )
        val_ds = tf.data.Dataset.from_generator(
            self._data_generator(self.val_dir), output_signature=output_signature
        )

        train_ds = train_ds.shuffle(100).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds, self.num_classes

    def _save_class_names(self):
        artifacts_dir = Path(self.config["paths"]["artifacts_dir"])
        mapping_path = artifacts_dir / "label_map.json"
        mapping = {i: name for i, name in enumerate(self.class_names)}
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=4)
        self.logger.info(f"Class mapping saved to {mapping_path}")
""",

        # Чистая архитектура U-Net
        "src/models/model_builder.py": """import tensorflow as tf
from tensorflow.keras import layers, models
import logging

class ModelBuilder:
    def __init__(self, config: dict, num_classes: int, logger: logging.Logger):
        self.config = config
        self.num_classes = num_classes
        self.logger = logger
        self.img_size = tuple(config["data"]["image_size"]) + (3,)

    def build(self) -> tf.keras.Model:
        self.logger.info("Building U-Net Semantic Segmentation model...")
        
        inputs = layers.Input(shape=self.img_size)
        x = layers.Rescaling(1./255)(inputs)

        # --- Encoder (Downsampling) ---
        c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(x)
        c1 = layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
        p1 = layers.MaxPooling2D(2)(c1)

        c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
        c2 = layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
        p2 = layers.MaxPooling2D(2)(c2)

        c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
        c3 = layers.Conv2D(128, 3, activation='relu', padding='same')(c3)
        p3 = layers.MaxPooling2D(2)(c3)

        # --- Bottleneck ---
        b = layers.Conv2D(256, 3, activation='relu', padding='same')(p3)
        b = layers.Conv2D(256, 3, activation='relu', padding='same')(b)

        # --- Decoder (Upsampling & Skip Connections) ---
        u1 = layers.UpSampling2D(2)(b)
        u1 = layers.Concatenate()([u1, c3])
        c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(u1)
        c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(c4)

        u2 = layers.UpSampling2D(2)(c4)
        u2 = layers.Concatenate()([u2, c2])
        c5 = layers.Conv2D(64, 3, activation='relu', padding='same')(u2)
        c5 = layers.Conv2D(64, 3, activation='relu', padding='same')(c5)

        u3 = layers.UpSampling2D(2)(c5)
        u3 = layers.Concatenate()([u3, c1])
        c6 = layers.Conv2D(32, 3, activation='relu', padding='same')(u3)
        c6 = layers.Conv2D(32, 3, activation='relu', padding='same')(c6)

        # --- Output ---
        # Для многоклассовой сегментации используем softmax на каждый пиксель
        outputs = layers.Conv2D(self.num_classes, 1, activation='softmax')(c6)

        model = models.Model(inputs, outputs)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config["model"]["learning_rate"])
        
        # sparse_categorical_crossentropy позволяет кормить маску в виде int индексов (0, 1, 2...)
        model.compile(
            optimizer=optimizer, 
            loss='sparse_categorical_crossentropy', 
            metrics=['sparse_categorical_accuracy']
        )
        
        self.logger.info("U-Net model compiled successfully.")
        return model
""",

        "src/training/trainer.py": """import tensorflow as tf
from pathlib import Path
import logging

class Trainer:
    def __init__(self, model: tf.keras.Model, config: dict, logger: logging.Logger):
        self.model = model
        self.config = config
        self.logger = logger
        self.epochs = config["training"]["epochs"]
        self.checkpoints_dir = Path(config["paths"]["checkpoints_dir"])
        self.logs_dir = Path(config["paths"]["logs_dir"])
        self.models_dir = Path(config["paths"]["models_dir"])

    def _get_callbacks(self) -> list:
        ckpt_path = self.checkpoints_dir / "best_model.keras"
        csv_path = self.logs_dir / "training_log.csv"
        tb_path = self.logs_dir / "tensorboard"

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(ckpt_path), monitor='val_loss', # Сохраняем по лоссу валидации
                save_best_only=True, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=self.config["training"]["early_stopping_patience"], 
                restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=self.config["training"]["reduce_lr_factor"], 
                patience=self.config["training"]["reduce_lr_patience"], verbose=1
            ),
            tf.keras.callbacks.CSVLogger(str(csv_path)),
            tf.keras.callbacks.TensorBoard(log_dir=str(tb_path), histogram_freq=1)
        ]
        return callbacks

    def train(self, train_ds: tf.data.Dataset, val_ds: tf.data.Dataset):
        self.logger.info("Starting U-Net training process...")
        history = self.model.fit(
            train_ds, validation_data=val_ds,
            epochs=self.epochs, callbacks=self._get_callbacks()
        )
        final_model_path = self.models_dir / "final_model.keras"
        self.model.save(str(final_model_path))
        self.logger.info(f"Training completed. Final model saved to {final_model_path}")
        return history
""",

        # Оценка обучения
        "src/evaluation/evaluator.py": """import numpy as np
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
        
        plt.subplot(1, 2, 1)
        plt.plot(hist_df['loss'], label='Train Loss')
        plt.plot(hist_df['val_loss'], label='Val Loss')
        plt.title('Loss (Sparse Categ. Crossentropy)')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(hist_df['sparse_categorical_accuracy'], label='Train Acc')
        plt.plot(hist_df['val_sparse_categorical_accuracy'], label='Val Acc')
        plt.title('Pixel Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.artifacts_dir / "training_history.png")
        plt.close()
""",

        # Инференс масок с наложением (Overlay)
        "src/inference/predictor.py": """import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
import json
import logging
import matplotlib.pyplot as plt

class Predictor:
    def __init__(self, config: dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.img_size = tuple(config["data"]["image_size"])
        self.alpha = config["inference"].get("alpha", 0.5)
        
        models_dir = Path(config["paths"]["models_dir"])
        self.artifacts_dir = Path(config["paths"]["artifacts_dir"])
        
        map_path = self.artifacts_dir / "label_map.json"
        if not map_path.exists(): raise FileNotFoundError("label_map.json not found.")
        
        with open(map_path, "r", encoding="utf-8") as f:
            self.idx_to_class = json.load(f)
            self.num_classes = len(self.idx_to_class)
            
        # Генерация ярких случайных цветов для классов (фон = черный)
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(self.num_classes, 3), dtype=np.uint8)
        self.colors[0] = [0, 0, 0] # Background
            
        model_path = models_dir / "final_model.keras"
        if not model_path.exists(): raise FileNotFoundError("Model not found.")
        
        self.logger.info("Loading U-Net model...")
        self.model = tf.keras.models.load_model(str(model_path))
        self.logger.info("Model loaded.")

    def predict_single(self, img_path: str, save_output=True):
        path = Path(img_path)
        if not path.exists():
            self.logger.error(f"Image not found: {path}")
            return None
            
        try:
            # Читаем через OpenCV (оставляем оригинал для красивого наложения)
            orig_img = cv2.imread(str(path))
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            orig_shape = orig_img.shape[:2]
            
            # Подготовка для модели
            img_resized = cv2.resize(orig_img, self.img_size)
            img_tensor = np.expand_dims(img_resized, axis=0).astype(np.float32)
            
            # Инференс маски: предсказываем класс для каждого пикселя
            pred_probs = self.model.predict(img_tensor, verbose=0)[0] # Shape: (H, W, Classes)
            pred_mask = np.argmax(pred_probs, axis=-1).astype(np.uint8) # Shape: (H, W)
            
            # Возвращаем размер маски к оригинальному разрешению изображения
            pred_mask = cv2.resize(pred_mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
            
            if save_output:
                # Создаем цветную маску
                color_mask = np.zeros_like(orig_img)
                for class_idx in range(1, self.num_classes): # Пропускаем фон
                    color_mask[pred_mask == class_idx] = self.colors[class_idx]
                
                # Наложение маски на оригинал с прозрачностью alpha
                overlay = cv2.addWeighted(orig_img, 1.0, color_mask, self.alpha, 0)
                
                out_path = self.artifacts_dir / f"pred_{path.name}"
                overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_path), overlay_bgr)
                self.logger.info(f"Saved segmentation overlay to {out_path}")
            
            return pred_mask
        except Exception as e:
            self.logger.error(f"Error predicting {path.name}: {e}")
            return None

    def predict_folder(self, folder_path: str):
        folder = Path(folder_path)
        if not folder.is_dir(): return
        
        images = [p for p in folder.iterdir() if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'}]
        self.logger.info(f"Found {len(images)} images for segmentation.")
        
        # Сохраним легенду цветов
        self._save_legend()
        
        for img_path in images:
            self.predict_single(str(img_path))

    def _save_legend(self):
        plt.figure(figsize=(4, self.num_classes * 0.5))
        for i in range(1, self.num_classes):
            plt.plot([0], [0], color=self.colors[i]/255.0, label=self.idx_to_class[str(i)], linewidth=10)
        plt.legend(loc='center', fontsize=12)
        plt.axis('off')
        plt.savefig(self.artifacts_dir / "color_legend.png", bbox_inches='tight')
        plt.close()
""",

        "train.py": """import os
import tensorflow as tf
from src.utils.config_parser import load_config
from src.utils.logger import setup_logger
from src.data.dataset import DataPipeline
from src.models.model_builder import ModelBuilder
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator
from pathlib import Path

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    config = load_config()
    logger = setup_logger("TrainerLogger", Path(config["paths"]["logs_dir"]))
    tf.random.set_seed(config["project"]["seed"])

    try:
        pipeline = DataPipeline(config, logger)
        train_ds, val_ds, num_classes = pipeline.create_datasets()

        builder = ModelBuilder(config, num_classes, logger)
        model = builder.build()
        model.summary(print_fn=logger.info)

        trainer = Trainer(model, config, logger)
        history = trainer.train(train_ds, val_ds)

        evaluator = Evaluator(model, config, logger)
        evaluator.evaluate(val_ds, history)

        logger.info("Segmentation training finished successfully.")

    except Exception as e:
        logger.exception(f"Fatal error during training: {e}")

if __name__ == "__main__":
    main()
""",

        "predict.py": """import argparse
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
""",
        "src/__init__.py": "", "src/utils/__init__.py": "", "src/data/__init__.py": "",
        "src/models/__init__.py": "", "src/training/__init__.py": "",
        "src/evaluation/__init__.py": "", "src/inference/__init__.py": ""
    }

    base_path.mkdir(exist_ok=True)
    print(f"[*] Папка: {project_name}/")

    for d in directories:
        (base_path / d).mkdir(parents=True, exist_ok=True)

    for file_path, content in files_content.items():
        with open(base_path / file_path, "w", encoding="utf-8") as f:
            f.write(content)

    print("\\n✅ Проект успешно переписан под U-Net Семантическую Сегментацию!")

if __name__ == "__main__":
    create_project()