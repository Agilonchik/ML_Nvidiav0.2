import os
import sys
import random
import re
import json
import base64
import yaml
import colorsys
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from PIL import Image, ImageDraw, ImageFont

# Базовая директория проекта. Так скрипт можно запускать из любой папки.
BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# ТЕСТОВЫЙ РЕЖИМ
# ============================================================
# Логика выбора файлов:
# 1) Если файлы переданы через параметры запуска, анализ идет строго в этом порядке:
#    python "Вставленный код_тестовый_порядок.py" 001.jpg 005.jpg 002.jpg
#
# 2) Если параметры запуска НЕ переданы, скрипт работает как раньше:
#    случайно выбирает TEST_SAMPLE_LIMIT файлов из data/val.
#
# Можно писать имя картинки или имя JSON-разметки:
# 001.jpg, 001.png, 001.json
TEST_SAMPLE_LIMIT = 5

# Шрифты с поддержкой кириллицы. Первый найденный будет использован для подписей.
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/calibri.ttf",
]

FONT_PATH = next((p for p in FONT_CANDIDATES if Path(p).exists()), None)


def _bgr_to_rgb_color(color):
    """Перевод цвета OpenCV BGR в PIL RGB."""
    return (int(color[2]), int(color[1]), int(color[0]))


def _get_font(font_size):
    """Возвращает TTF-шрифт с кириллицей или стандартный шрифт PIL."""
    if FONT_PATH:
        return ImageFont.truetype(FONT_PATH, font_size)
    return ImageFont.load_default()


def _strip_status_icons(text):
    """
    Убирает emoji-иконки из текста на изображении.
    Некоторые TTF-шрифты в WSL не содержат ✅ ⚠️ ❌, из-за этого могут быть квадраты.
    В терминале исходный текст с emoji остается без изменений.
    """
    return (
        str(text)
        .replace("✅", "")
        .replace("⚠️", "")
        .replace("⚠", "")
        .replace("❌", "")
        .strip()
    )


def put_text_ru(
    img,
    text,
    position,
    font_size=28,
    color=(255, 255, 255),
    background_box=None,
    background_color=(0, 0, 0),
    max_width=None,
):
    """
    Выводит русский текст на OpenCV-изображение через Pillow.

    img: изображение OpenCV в формате BGR
    text: строка на русском
    position: координаты верхнего левого угла текста (x, y)
    font_size: размер шрифта
    color: цвет текста в формате BGR
    background_box: прямоугольник фона (x1, y1, x2, y2) или None
    background_color: цвет фона в формате BGR
    max_width: максимальная ширина текста; если текст не помещается, шрифт уменьшается
    """
    text = str(text)

    # OpenCV BGR -> PIL RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    draw = ImageDraw.Draw(pil_img)
    x, y = position

    # Подбор размера шрифта под заданную ширину
    current_size = font_size
    font = _get_font(current_size)

    if max_width is not None:
        while current_size > 8:
            bbox = draw.textbbox((x, y), text, font=font)
            text_width = bbox[2] - bbox[0]
            if text_width <= max_width:
                break
            current_size -= 1
            font = _get_font(current_size)

    if background_box is not None:
        draw.rectangle(background_box, fill=_bgr_to_rgb_color(background_color))

    draw.text((x, y), text, font=font, fill=_bgr_to_rgb_color(color))

    # PIL RGB -> OpenCV BGR
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

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


def _build_class_colors(class_names):
    """
    Возвращает стабильный список цветов BGR под каждый класс:
    - индекс 0 (фон) всегда черный
    - первые 15 классов получают фиксированные различимые цвета
    - для остальных генерируются дополнительные различимые цвета
    """
    # Палитра в RGB (визуально различимые цвета), далее переводим в BGR для OpenCV
    reserved_rgb = [
        (0, 0, 0),         # background
        (230, 25, 75),     # red
        (60, 180, 75),     # green
        (255, 225, 25),    # yellow
        (0, 130, 200),     # blue
        (245, 130, 48),    # orange
        (145, 30, 180),    # purple
        (70, 240, 240),    # cyan
        (240, 50, 230),    # magenta
        (210, 245, 60),    # lime
        (250, 190, 190),   # pink
        (0, 128, 128),     # teal
        (230, 190, 255),   # lavender
        (170, 110, 40),    # brown
        (255, 250, 200),   # beige
    ]
    colors_bgr = [(rbg[2], rbg[1], rbg[0]) for rbg in reserved_rgb]

    # Для классов сверх 15 создаем цвета "рандомайзером" с фиксированным seed
    # (чтобы цвет у класса не менялся между запусками).
    if len(class_names) > len(colors_bgr):
        rng = random.Random(42)
        used = set(colors_bgr)
        while len(colors_bgr) < len(class_names):
            # Генерируем насыщенные и яркие цвета через HSV
            h = rng.random()
            s = rng.uniform(0.6, 0.95)
            v = rng.uniform(0.75, 1.0)
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            bgr = (int(b * 255), int(g * 255), int(r * 255))
            if bgr not in used:
                colors_bgr.append(bgr)
                used.add(bgr)

    return np.array(colors_bgr, dtype=np.uint8)


def _save_color_key(class_names, colors_bgr, output_path):
    """Сохраняет ключ соответствия класс -> цвет в отдельный JSON-файл."""
    key_payload = []
    for idx, class_name in enumerate(class_names):
        b, g, r = [int(c) for c in colors_bgr[idx]]
        key_payload.append(
            {
                "class_index": idx,
                "class_name": class_name,
                "color_bgr": [b, g, r],
                "color_rgb": [r, g, b],
                "color_hex_rgb": f"#{r:02X}{g:02X}{b:02X}",
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(key_payload, f, ensure_ascii=False, indent=2)

# ==========================================
# 🛠 3. ГЛАВНЫЙ ЦИКЛ ДИАГНОСТИКИ
# ==========================================
def main():
    setup_hardware()
    
    # Загрузка конфига
    CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    VAL_DIR = BASE_DIR / "data" / "val"
    MODEL_PATH = BASE_DIR / "models" / "final_model.keras"
    OUTPUT_PATH = BASE_DIR / "artifacts" / "diagnostics.png"
    COLOR_KEY_PATH = BASE_DIR / "artifacts" / "diagnostics_color_key.json"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    IMG_SIZE = tuple(config["data"]["image_size"])
    CLASS_NAMES = config['data']['classes']

    if not MODEL_PATH.exists():
        print(f"❌ Ошибка: Модель не найдена по пути {MODEL_PATH}")
        return

    # Стабильная палитра: 15 зарезервированных цветов + генерация для остальных
    COLORS = _build_class_colors(CLASS_NAMES)
    _save_color_key(CLASS_NAMES, COLORS, COLOR_KEY_PATH)
    print(f"🎨 Ключ цветов сохранен в: {COLOR_KEY_PATH}")
    
    # Загрузка модели (БЕЗ компиляции, нам только предсказания)
    print("🌀 Загрузка модели...")
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    
    # Сбор файлов для тестового анализа.
    # 1) Если переданы аргументы командной строки, используется их порядок.
    # 2) Если аргументы не переданы, выполняется случайная выборка TEST_SAMPLE_LIMIT файлов.
    requested_order = sys.argv[1:]

    try:
        selected_jsons = resolve_jsons_in_order(
            VAL_DIR,
            requested_items=requested_order,
            sample_limit=TEST_SAMPLE_LIMIT,
        )
    except FileNotFoundError as e:
        print(f"❌ Ошибка: {e}")
        return

    if not selected_jsons:
        print("❌ Ошибка: список файлов для анализа пуст.")
        return

    if requested_order:
        print("\n📌 Режим: порядок из параметров запуска.")
    else:
        print(f"\n📌 Режим: случайная выборка {len(selected_jsons)} файлов.")

    print("📌 Порядок анализа:")
    for n, j_path in enumerate(selected_jsons, start=1):
        print(f"   {n}. {j_path.name}")

    grid_rows = []
    
    # Стандартизируем размер для вывода на панель (например, 320x320)
    DISPLAY_SIZE = (320, 320)
    
    print(f"\n🔬 Начинаю анализ {len(selected_jsons)} образцов...")
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
        
        # Добавляем текстовый отчет на панель через Pillow, чтобы русский текст не превращался в "????"
        match_for_image = _strip_status_icons(match)
        p3 = put_text_ru(
            p3,
            match_for_image,
            (8, DISPLAY_SIZE[1] - 25),
            font_size=16,
            color=(255, 255, 255),
            background_box=(0, DISPLAY_SIZE[1] - 32, DISPLAY_SIZE[0], DISPLAY_SIZE[1]),
            background_color=(0, 0, 0),
            max_width=DISPLAY_SIZE[0] - 16,
        )

        row = np.hstack([p1, p2, p3])
        grid_rows.append(row)
        print(f"   Обработан файл {j_path.name}: {match}")

    # Финальная склейка и сохранение
    print("\n✅ Компоновка диагностической панели...")
    final_grid = np.vstack(grid_rows)
    # Добавляем заголовки через Pillow, чтобы кириллица корректно отображалась на изображении
    header = np.zeros((60, final_grid.shape[1], 3), dtype=np.uint8)
    w_p = DISPLAY_SIZE[0]

    header = put_text_ru(
        header,
        "Исходное изображение",
        (10, 15),
        font_size=24,
        color=(255, 255, 255),
        max_width=w_p - 20,
    )

    header = put_text_ru(
        header,
        "Экспертная разметка",
        (w_p + 10, 15),
        font_size=24,
        color=(255, 255, 255),
        max_width=w_p - 20,
    )

    header = put_text_ru(
        header,
        "Прогнозирование разметки",
        (w_p * 2 + 10, 15),
        font_size=24,
        color=(255, 255, 255),
        max_width=w_p - 20,
    )
    
    final_output = np.vstack([header, final_grid])
    cv2.imwrite(str(OUTPUT_PATH), final_output)
    print(f"🎉 Готово! Строгая диагностика сохранена в: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
