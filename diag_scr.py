import os
import random
import json
import base64
import yaml
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

# ==========================================
# 🚀 1. ИНИЦИАЛИЗАЦИЯ И СЕТАП ЖЕЛЕЗА
# ==========================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def setup_hardware():
    """Memory Growth и Mixed Precision"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ Memory Growth включен.")
        except: pass
    
    # Включаем Mixed Precision, чтобы диагност читал модель корректно
    mixed_precision.set_global_policy('mixed_float16')
    print("⚡ Mixed Precision включен.")

# ==========================================
# 🧬 2. ВСПОМОГАТЕЛЬНАЯ МАТЕМАТИКА (dataset.py)
# ==========================================
def decode_mask(b64_str):
    try:
        mask_bytes = base64.b64decode(b64_str)
        m_img = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        if m_img is not None and len(m_img.shape) > 2: m_img = np.max(m_img, axis=2)
        return m_img
    except: return None

def mask_to_rgb(mask, colors):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx in range(1, len(colors)): rgb[mask == idx] = colors[idx]
    return rgb

# ==========================================
# 🛠 3. ГЛАВНЫЙ ЦИКЛ ДИАГНОСТИКИ
# ==========================================
def main():
    setup_hardware()
    
    # Загрузка конфига
    with open("configs/config.yaml", "r", encoding="utf-8") as f: config = yaml.safe_load(f)
    VAL_DIR = Path("data/val")
    MODEL_PATH = Path("models/final_model.keras")
    OUTPUT_PATH = Path("artifacts/diagnostics.png")
    IMG_SIZE = tuple(config["data"]["image_size"])
    CLASS_NAMES = config['data']['classes']

    if not MODEL_PATH.exists():
        print(f"❌ Ошибка: Модель не найдена по пути {MODEL_PATH}")
        return

    # Генерация цветов (фон=черный, остальные - яркие)
    np.random.seed(42)
    COLORS = np.random.randint(50, 255, size=(len(CLASS_NAMES), 3), dtype=np.uint8)
    COLORS[0] = (0, 0, 0)
    
    # Загрузка модели (БЕЗ компиляции, нам только предсказания)
    print("🌀 Загрузка модели...")
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    
    # Сбор и выбор файлов
    json_files = list(VAL_DIR.glob("*.json"))
    # Игнорируем аугментации, если они есть в val
    json_files = [f for f in json_files if "_aug_" not in f.name]
    
    if len(json_files) < 5:
        print(f"❌ Ошибка: В data/val меньше 5 оригинальных файлов ({len(json_files)})")
        return
    
    selected_jsons = random.sample(json_files, 5)
    
    grid_rows = []
    
    # Стандартизируем размер для вывода на панель (например, 320x320)
    DISPLAY_SIZE = (320, 320)
    
    print("\n🔬 Начинаю анализ 5 образцов...")
    for j_path in selected_jsons:
        # Ищем картинку
        img_path = None
        for ext in ['.jpg','.jpeg','.png','.bmp']:
            if j_path.with_suffix(ext).exists():
                img_path = j_path.with_suffix(ext)
                break
        if not img_path: continue
        
        # --- 1. ОРИГИНАЛ ---
        img_bgr = cv2.imread(str(img_path))
        orig_h, orig_w = img_bgr.shape[:2]
        
        # --- 2. ЭТАЛОН (GT) ---
        mask_gt = np.zeros((orig_h, orig_w), dtype=np.uint8)
        with open(j_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'shapes' in data:
                for shape in data['shapes']:
                    label = shape.get('label', '')
                    if label in CLASS_NAMES:
                        idx = CLASS_NAMES.index(label)
                        pts = shape.get('points', [])
                        sType = shape.get('shape_type', '')
                        try:
                            if sType == 'mask' and 'mask' in shape:
                                m_img = decode_mask(shape['mask'])
                                if m_img is not None and len(pts) >= 2:
                                    x, y = int(min(pts[0][0], pts[1][0])), int(min(pts[0][1], pts[1][1]))
                                    hm, wm = m_img.shape[:2]
                                    ye, xe = min(y + hm, orig_h), min(x + wm, orig_w)
                                    roi, m_crop = mask_gt[max(0, y):ye, max(0, x):xe], m_img[max(0, y)-y:ye-y, max(0, x)-x:xe-x]
                                    roi[m_crop > 0] = idx
                            elif sType == 'rectangle' or len(pts) == 2:
                                cv2.rectangle(mask_gt, (int(pts[0][0]), int(pts[0][1])), (int(pts[1][0]), int(pts[1][1])), idx, -1)
                            elif sType == 'polygon' or len(pts) >= 3:
                                cv2.fillPoly(mask_gt, [np.array(pts, np.int32)], idx)
                        except: pass
        
        # --- 3. ПРЕДСКАЗАНИЕ (PRED) ---
        # Подготовка тензора
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inp = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_LINEAR)
        inp = np.expand_dims(inp.astype(np.float32), axis=0)
        
        # Инференс и postprocess
        pred_probs = model.predict(inp, verbose=0)[0]
        pred_mask = np.argmax(pred_probs, axis=-1).astype(np.uint8)
        
        # Обязательно возвращаем маску к оригинальному размеру для честного сравнения!
        pred_mask = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        
        # --- 4. ОЦЕНКА ОШИБОК ПОД МИКРОСКОПОМ (СТРОГАЯ ПРОВЕРКА) ---
        # Где реально есть объекты (не фон)
        gt_objects_mask = (mask_gt > 0)
        total_gt_pixels = np.sum(gt_objects_mask)
        
        # Где сеть угадала ИМЕННО ПРАВИЛЬНЫЙ КЛАСС (И это не фон)
        correct_pixels_mask = (pred_mask == mask_gt) & gt_objects_mask
        correct_pixels = np.sum(correct_pixels_mask)
        
        # Считаем процент именно ВЕРНЫХ совпадений
        if total_gt_pixels > 0:
            true_accuracy = (correct_pixels / total_gt_pixels) * 100
            
            if true_accuracy > 50:
                match = f"✅ Точное попадание: {true_accuracy:.1f}%"
            elif true_accuracy > 10:
                match = f"⚠️ Частичное совпадение: {true_accuracy:.1f}%"
            else:
                match = f"❌ Промах (Не тот класс): {true_accuracy:.1f}%"
        else:
            # Если на картинке вообще нет объектов (только фон)
            if np.sum(pred_mask > 0) > 0:
                match = "❌ Галлюцинация (Нарисовала объект на фоне)"
            else:
                match = "✅ Идеально чистый фон"

        # --- 5. КОМПОНОВКА ПАНЕЛИ ---
        p1 = cv2.resize(img_bgr, DISPLAY_SIZE)
        
        # Красим маски в BGR для сохранения в файл
        m_gt_rgb = mask_to_rgb(mask_gt, COLORS)
        m_gt_bgr = cv2.cvtColor(m_gt_rgb, cv2.COLOR_RGB2BGR)
        p2 = cv2.resize(m_gt_bgr, DISPLAY_SIZE, interpolation=cv2.INTER_NEAREST)
        
        m_pred_rgb = mask_to_rgb(pred_mask, COLORS)
        m_pred_bgr = cv2.cvtColor(m_pred_rgb, cv2.COLOR_RGB2BGR)
        p3 = cv2.resize(m_pred_bgr, DISPLAY_SIZE, interpolation=cv2.INTER_NEAREST)
        
        # Добавляем текстовый отчет на панель (с фоном для читаемости)
        cv2.rectangle(p3, (0, DISPLAY_SIZE[1]-25), (DISPLAY_SIZE[0], DISPLAY_SIZE[1]), (0,0,0), -1)
        cv2.putText(p3, match, (10, DISPLAY_SIZE[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        row = np.hstack([p1, p2, p3])
        grid_rows.append(row)
        print(f"   Обработан файл {j_path.name}: {match}")

    # Финальная склейка и сохранение
    print("\n✅ Компоновка диагностической панели...")
    final_grid = np.vstack(grid_rows)
    # Добавляем заголовки
    header = np.zeros((50, final_grid.shape[1], 3), dtype=np.uint8)
    w_p = DISPLAY_SIZE[0]
    cv2.putText(header, "Original Image", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(header, "Your Annotation (GT)", (w_p + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(header, "Network Prediction (Pred)", (w_p*2 + 10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    final_output = np.vstack([header, final_grid])
    cv2.imwrite(str(OUTPUT_PATH), final_output)
    print(f"🎉 Готово! Строгая диагностика сохранена в: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()