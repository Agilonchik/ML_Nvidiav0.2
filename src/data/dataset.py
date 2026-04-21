import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
import json
import logging
import base64

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
        
        # Если в конфиге уже жестко заданы классы, используем их
        if 'classes' in self.config['data']:
            self.class_names = self.config['data']['classes']
        else:
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
                
                # Парсим полигоны и маски
                with open(j_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'shapes' in data:
                        for shape in data['shapes']:
                            label = shape.get('label', '')
                            if label in self.class_names:
                                class_idx = self.class_names.index(label)
                                pts = shape.get('points', [])
                                sType = shape.get('shape_type', '')

                                try:
                                    # 1. Обработка Base64 масок
                                    if sType == 'mask' and 'mask' in shape:
                                        m_bytes = base64.b64decode(shape['mask'])
                                        m_img = cv2.imdecode(np.frombuffer(m_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
                                        
                                        if m_img is not None and len(pts) >= 2:
                                            if len(m_img.shape) > 2:
                                                m_img = np.max(m_img, axis=2)
                                                
                                            x = int(min(pts[0][0], pts[1][0]))
                                            y = int(min(pts[0][1], pts[1][1]))
                                            hm, wm = m_img.shape[:2]
                                            
                                            ye, xe = min(y + hm, orig_h), min(x + wm, orig_w)
                                            x_start, y_start = max(0, x), max(0, y)
                                            
                                            if (ye - y_start) > 0 and (xe - x_start) > 0:
                                                roi = mask[y_start:ye, x_start:xe]
                                                m_crop = m_img[y_start-y : ye-y, x_start-x : xe-x]
                                                # БЕРЕМ ВСЁ, ЧТО БОЛЬШЕ 0
                                                roi[m_crop > 0] = class_idx
                                                
                                    # 2. Обработка Прямоугольников
                                    elif sType == 'rectangle' or len(pts) == 2:
                                        x1 = max(0, int(min(pts[0][0], pts[1][0])))
                                        y1 = max(0, int(min(pts[0][1], pts[1][1])))
                                        x2 = min(orig_w, int(max(pts[0][0], pts[1][0])))
                                        y2 = min(orig_h, int(max(pts[0][1], pts[1][1])))
                                        cv2.rectangle(mask, (x1, y1), (x2, y2), class_idx, -1)
                                        
                                    # 3. Обработка Полигонов
                                    elif sType == 'polygon' or len(pts) >= 3:
                                        temp = np.zeros_like(mask)
                                        cv2.fillPoly(temp, [np.array(pts, np.int32)], class_idx)
                                        mask = np.where(temp > 0, class_idx, mask)

                                except Exception as e:
                                    self.logger.warning(f"Error drawing shape {label} in {img_path.name}: {e}")
                
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

        # ==========================================
        # 📊 ПОДСЧЕТ И ЛОГИРОВАНИЕ ФАЙЛОВ
        # ==========================================
        train_jsons = list(self.train_dir.glob("*.json"))
        val_jsons = list(self.val_dir.glob("*.json"))
        
        # Считаем только те JSON, у которых есть реальная картинка
        train_valid = sum(1 for j in train_jsons if self._find_image_for_json(j))
        val_valid = sum(1 for j in val_jsons if self._find_image_for_json(j))
        
        self.logger.info("=" * 50)
        self.logger.info(f"📂 ДАННЫЕ ДЛЯ ОБУЧЕНИЯ (TRAIN): {train_valid} файлов")
        self.logger.info(f"📂 ДАННЫЕ ДЛЯ ВАЛИДАЦИИ (VAL):  {val_valid} файлов")
        self.logger.info("=" * 50)
        # ==========================================

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

        return train_ds, val_ds, self.num_classes, train_valid, val_valid

    def _save_class_names(self):
        artifacts_dir = Path(self.config["paths"]["artifacts_dir"])
        mapping_path = artifacts_dir / "label_map.json"
        mapping = {i: name for i, name in enumerate(self.class_names)}
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=4)
        self.logger.info(f"Class mapping saved to {mapping_path}")