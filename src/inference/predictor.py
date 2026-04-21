import tensorflow as tf
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
            
            # Инференс маски
            pred_probs = self.model.predict(img_tensor, verbose=0)[0] 
            
            # СГЛАЖИВАНИЕ: растягиваем дробные вероятности (INTER_CUBIC дает самые плавные края)
            pred_probs_resized = cv2.resize(pred_probs, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_CUBIC)
            
            # И только теперь выбираем класс-победитель для каждого пикселя
            pred_mask = np.argmax(pred_probs_resized, axis=-1).astype(np.uint8)
            
            # --- Морфологическая очистка маски ---
            # 1. Убираем "соль" (мелкие случайные пиксели)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
            
            # 2. Замазываем "дырки" внутри объектов
            pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)
            
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
