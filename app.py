import streamlit as st
import cv2
import json
import numpy as np
import os
import base64
import yaml
from pathlib import Path
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import load_model

# ==========================================
# 1. КОНФИГУРАЦИЯ И ПУТИ
# ==========================================
st.set_page_config(page_title="U-Net HIL Annotation Tool", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"


@st.cache_data
def load_project_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if not CONFIG_PATH.exists():
    st.error(f"❌ Конфиг не найден: {CONFIG_PATH}")
    st.stop()

config = load_project_config(CONFIG_PATH)
data_cfg = config.get("data", {})

if "classes" in data_cfg:
    CLASS_NAMES = data_cfg["classes"]
elif "classes" in config:
    CLASS_NAMES = config["classes"]
else:
    CLASS_NAMES = ["Background"]

IMG_SIZE = tuple(data_cfg.get("image_size", (512, 512)))
OVERLAY_ALPHA = config.get("inference", {}).get("alpha", 0.5)
MODEL_PATH = BASE_DIR / "models" / "final_model.keras"

np.random.seed(42)
_colors_raw = np.random.randint(50, 255, size=(len(CLASS_NAMES), 3), dtype=np.uint8)
COLORS = [tuple(map(int, c)) for c in _colors_raw]
COLORS[0] = (0, 0, 0)

PATHS = {
    "train": Path("data/train"),
    "val": Path("data/val"),
    "unlabelled": Path("data/unlabelled"),
    "train_pl": Path("data/train_pl"),
}
for p in PATHS.values():
    p.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================


# 💡 Добавляем show_spinner, чтобы ты видел, когда грузятся новые веса
@st.cache_resource(show_spinner="🔄 Обнаружена новая модель! Загружаем веса...")
def get_model(path_str, file_mod_time):
    if not os.path.exists(path_str):
        return None
    try:
        # Принудительно ставим политику для инференса, чтобы не было пустых тензоров
        mixed_precision.set_global_policy("mixed_float16")
        return load_model(path_str, compile=False)
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None


def preprocess_image(img_path, target_size):
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img_rgb.shape[:2]
    img_res = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)
    return np.expand_dims(img_res.astype(np.float32), axis=0), (h_orig, w_orig)


def postprocess_prediction(pred, orig_shape):
    # Растягиваем вероятности плавно (INTER_CUBIC)
    pred_resized = cv2.resize(
        pred[0], (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_CUBIC
    )
    mask = np.argmax(pred_resized, axis=-1).astype(np.uint8)

    # 🪄 Морфологическое сглаживание (убираем пыль и дырки)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def mask_to_rgb(mask, colors):
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx in range(1, len(colors)):
        rgb[mask == idx] = colors[idx]
    return rgb


def apply_overlay(image, mask_rgb, alpha=0.5):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    res = cv2.addWeighted(image_bgr, 1.0, mask_bgr, alpha, 0)
    return cv2.cvtColor(res, cv2.COLOR_BGR2RGB)


def save_labelme_json(img_path, mask, class_names, output_dir):
    img_name = img_path.name
    output_json = output_dir / img_path.with_suffix(".json").name
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    shapes = []

    for idx in range(1, len(class_names)):
        class_mask = (mask == idx).astype(np.uint8) * 255
        if np.sum(class_mask) == 0:
            continue
        contours, _ = cv2.findContours(
            class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, wb, hb = cv2.boundingRect(c)
            local_mask = class_mask[y : y + hb, x : x + wb]
            _, encoded = cv2.imencode(".png", local_mask)
            shapes.append(
                {
                    "label": class_names[idx],
                    "points": [[float(x), float(y)], [float(x + wb), float(y + hb)]],
                    "shape_type": "mask",
                    "mask": base64.b64encode(encoded).decode("utf-8"),
                }
            )

    json_data = {
        "version": "5.11.3",
        "shapes": shapes,
        "imagePath": img_name,
        "imageHeight": h,
        "imageWidth": w,
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    import shutil

    shutil.copy2(img_path, output_dir / img_name)
    return output_json


# ==========================================
# 3. ИНТЕРФЕЙС
# ==========================================

st.sidebar.title("💎 U-Net HIL Control")
app_mode = st.sidebar.selectbox(
    "Режим", ["Просмотр разметки (HIL)", "Авто-разметка новых данных"]
)

# --- МАГИЯ АВТО-ОБНОВЛЕНИЯ КЭША ---
if MODEL_PATH.exists():
    mod_time = os.path.getmtime(MODEL_PATH)
else:
    mod_time = 0

model = get_model(str(MODEL_PATH), mod_time)
# ----------------------------------

if app_mode == "Просмотр разметки (HIL)":
    st.header("🔍 Режим Human-in-the-Loop: Сравнение")

    ds_type = st.radio("Папка:", ["train", "val"], horizontal=True)
    folder = PATHS[ds_type]
    json_files = list(folder.glob("*.json"))

    if not json_files:
        st.info("В папке нет JSON файлов.")
    else:
        selected_json = st.selectbox("Файл:", [f.name for f in json_files])
        json_path = folder / selected_json

        pred_mask = None
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            if json_path.with_suffix(ext).exists():
                img_path = json_path.with_suffix(ext)
                break

        if img_path:
            img_bgr = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h_orig, w_orig = img_rgb.shape[:2]

            mask_gt = np.zeros((h_orig, w_orig), dtype=np.uint8)

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            cls_lower = [str(c).lower().strip() for c in CLASS_NAMES]
            debug_logs = []

            if "shapes" in data:
                for sh in data["shapes"]:
                    lbl_orig = sh.get("label", "")
                    lbl = str(lbl_orig).lower().strip()

                    if lbl in cls_lower:
                        idx = cls_lower.index(lbl)
                        pts = sh.get("points", [])
                        sType = sh.get("shape_type", "")

                        try:
                            pixels_drawn = 0  # Вводим счетчик реальных пикселей!

                            if sType == "mask" and "mask" in sh:
                                m_bytes = base64.b64decode(sh["mask"])
                                # Читаем маску В ЛЮБОМ формате (даже если она цветная или с прозрачностью)
                                m_img = cv2.imdecode(
                                    np.frombuffer(m_bytes, np.uint8),
                                    cv2.IMREAD_UNCHANGED,
                                )

                                if m_img is not None and len(pts) >= 2:
                                    # Если в маске несколько каналов (RGB/RGBA), сплющиваем в один
                                    if len(m_img.shape) > 2:
                                        m_img = np.max(m_img, axis=2)

                                    x = int(min(pts[0][0], pts[1][0]))
                                    y = int(min(pts[0][1], pts[1][1]))
                                    hm, wm = m_img.shape[:2]

                                    ye, xe = min(y + hm, h_orig), min(x + wm, w_orig)
                                    x_start, y_start = max(0, x), max(0, y)

                                    if (ye - y_start) > 0 and (xe - x_start) > 0:
                                        roi = mask_gt[y_start:ye, x_start:xe]
                                        m_crop = m_img[
                                            y_start - y : ye - y, x_start - x : xe - x
                                        ]

                                        # БЕРЕМ ВСЁ, ЧТО БОЛЬШЕ 0 (а не 127)!
                                        valid_pixels = m_crop > 0
                                        roi[valid_pixels] = idx
                                        pixels_drawn = np.sum(valid_pixels)

                            elif sType == "rectangle" or len(pts) == 2:
                                x1 = max(0, int(min(pts[0][0], pts[1][0])))
                                y1 = max(0, int(min(pts[0][1], pts[1][1])))
                                x2 = min(w_orig, int(max(pts[0][0], pts[1][0])))
                                y2 = min(h_orig, int(max(pts[0][1], pts[1][1])))
                                pixels_drawn = (x2 - x1) * (y2 - y1)
                                if pixels_drawn > 0:
                                    cv2.rectangle(mask_gt, (x1, y1), (x2, y2), idx, -1)

                            elif sType == "polygon" or len(pts) >= 3:
                                temp = np.zeros_like(mask_gt)
                                cv2.fillPoly(temp, [np.array(pts, np.int32)], idx)
                                mask_gt = np.where(temp > 0, idx, mask_gt)
                                pixels_drawn = np.sum(temp > 0)

                            # Честный отчет в дебаг-панель
                            if pixels_drawn > 0:
                                debug_logs.append(
                                    f"✅ Нарисовано: {lbl_orig} ({sType}) -> {pixels_drawn} пикселей"
                                )
                            else:
                                debug_logs.append(
                                    f"⚠️ Пусто: {lbl_orig} найден, но внутри 0 пикселей!"
                                )

                        except Exception as e:
                            debug_logs.append(
                                f"❌ Ошибка кода при отрисовке {lbl_orig}: {e}"
                            )
                    else:
                        debug_logs.append(f"❌ Пропущено: {lbl_orig} (Нет в конфиге)")

            with st.expander("🛠 Дебаг-панель", expanded=False):
                for log in debug_logs:
                    st.write(log)

            # --- Предсказание нейросети ---
            if model:
                inp, shp = preprocess_image(img_path, IMG_SIZE)
                if inp is not None:
                    res = model.predict(inp, verbose=0)
                    pred_mask = postprocess_prediction(res, shp)

            # --- Отрисовка Общего вида ---
            st.subheader("📊 Общий вид")
            c1, c2, c3 = st.columns(3)
            c1.image(img_rgb, caption="Оригинал", use_container_width=True)

            m_gt_rgb = mask_to_rgb(mask_gt, COLORS)
            c2.image(
                apply_overlay(img_rgb, m_gt_rgb, OVERLAY_ALPHA),
                caption="Разметка человека",
                use_container_width=True,
            )

            if pred_mask is not None:
                m_pred_rgb = mask_to_rgb(pred_mask, COLORS)
                c3.image(
                    apply_overlay(img_rgb, m_pred_rgb, OVERLAY_ALPHA),
                    caption="Нейросеть",
                    use_container_width=True,
                )

            # --- Детальный разбор ---
            st.markdown("---")
            st.subheader("🧩 Детально по НАЙДЕННЫМ классам")

            drawn_any = False
            for i in range(1, len(CLASS_NAMES)):
                in_gt = np.any(mask_gt == i)
                in_pred = pred_mask is not None and np.any(pred_mask == i)

                # Показываем только если класс ЕСТЬ в разметке или предсказании
                if in_gt or in_pred:
                    drawn_any = True
                    st.markdown(f"**Класс: {CLASS_NAMES[i]}**")
                    left_col, right_col = st.columns(2)

                    if in_gt:
                        single_gt = np.zeros_like(mask_gt)
                        single_gt[mask_gt == i] = i
                        single_rgb_gt = mask_to_rgb(single_gt, COLORS)
                        # Накладываем НА ОРИГИНАЛЬНОЕ ФОТО
                        left_col.image(
                            apply_overlay(img_rgb, single_rgb_gt, OVERLAY_ALPHA),
                            caption=f"Эталон: {CLASS_NAMES[i]}",
                            use_container_width=True,
                        )
                    else:
                        left_col.info("Отсутствует в ручной разметке")

                    if in_pred:
                        single_pred = np.zeros_like(mask_gt)
                        single_pred[pred_mask == i] = i
                        single_rgb_pred = mask_to_rgb(single_pred, COLORS)
                        # Накладываем НА ОРИГИНАЛЬНОЕ ФОТО
                        right_col.image(
                            apply_overlay(img_rgb, single_rgb_pred, OVERLAY_ALPHA),
                            caption=f"Нейросеть: {CLASS_NAMES[i]}",
                            use_container_width=True,
                        )
                    else:
                        right_col.info("Нейросеть не обнаружила")

            if not drawn_any:
                st.warning(
                    "На этом фото нет ни одного объекта из списка классов (только фон)."
                )

elif app_mode == "Авто-разметка новых данных":
    # (Оставляем вкладку авто-разметки как была)
    st.header("🔮 Автоматическая разметка")
    # ...
    st.info("Переключитесь на вкладку HIL, чтобы проверить разметку.")
