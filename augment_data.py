import cv2
import json
import numpy as np
import base64
from pathlib import Path
import random
import uuid
import sys
import os

# ==========================================
# КОНФИГУРАЦИЯ
# ==========================================
TARGET_SIZE = (512, 512)

# ==========================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ (ДЛЯ МАСОК)
# ==========================================
def decode_mask(b64_str):
    try:
        mask_bytes = base64.b64decode(b64_str)
        mask_img = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        if mask_img is not None and len(mask_img.shape) > 2:
            mask_img = np.max(mask_img, axis=2)
        return mask_img
    except: return None

def encode_mask(mask_img):
    if len(mask_img.shape) > 2: mask_img = np.max(mask_img, axis=2)
    _, encoded = cv2.imencode('.png', mask_img)
    return base64.b64encode(encoded).decode('utf-8')

def get_full_mask(shape, h, w):
    full_m = np.zeros((h, w), dtype=np.uint8)
    if shape.get('shape_type') == 'mask' and 'mask' in shape:
        m = decode_mask(shape['mask'])
        if m is not None:
            pts = shape['points']
            mx = int(min(pts[0][0], pts[1][0]))
            my = int(min(pts[0][1], pts[1][1]))
            mh, mw = m.shape[:2]
            ye, xe = min(my + mh, h), min(mx + mw, w)
            my_start, mx_start = max(0, my), max(0, mx)
            if (ye - my_start) > 0 and (xe - mx_start) > 0:
                full_m[my_start:ye, mx_start:xe] = m[my_start-my:ye-my, mx_start-mx:xe-mx]
    return full_m

def extract_local_mask(full_m):
    contours, _ = cv2.findContours((full_m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, None
    c = max(contours, key=cv2.contourArea)
    x, y, w_bbox, h_bbox = cv2.boundingRect(c)
    local_m = full_m[y:y+h_bbox, x:x+w_bbox]
    pts = [[float(x), float(y)], [float(x+w_bbox), float(y+h_bbox)]]
    return local_m, pts

def save_augmentation(orig_img_path, new_img, new_data, suffix):
    unique_id = uuid.uuid4().hex[:4]
    new_img_name = f"{orig_img_path.stem}_{suffix}_{unique_id}{orig_img_path.suffix}"
    # Сохраняем в ТУ ЖЕ папку, где лежал оригинал (train или val)
    new_img_path = orig_img_path.parent / new_img_name
    new_data['imagePath'] = new_img_name
    new_data['imageData'] = None
    cv2.imwrite(str(new_img_path), new_img)
    with open(new_img_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
        json.dump(new_data, f, indent=2)

# ==========================================
# МОДУЛИ АУГМЕНТАЦИИ
# ==========================================
def augment_flip(img, data):
    h, w = img.shape[:2]
    new_img = cv2.flip(img, 1)
    new_data = json.loads(json.dumps(data))
    valid_shapes = []
    for s in new_data.get('shapes', []):
        if s.get('shape_type') == 'mask' and 'mask' in s:
            full_m = get_full_mask(s, h, w)
            flipped_full_m = cv2.flip(full_m, 1)
            local_m, new_pts = extract_local_mask(flipped_full_m)
            if local_m is not None:
                s['mask'] = encode_mask(local_m)
                s['points'] = new_pts
                valid_shapes.append(s)
        else:
            s['points'] = [[w - p[0], p[1]] for p in s['points']]
            valid_shapes.append(s)
    new_data['shapes'] = valid_shapes
    return new_img, new_data, "aug_flip"

def augment_crop(img, data):
    h, w = img.shape[:2]
    all_pts = [p for s in data.get('shapes', []) for p in s['points']]
    if not all_pts: return None, None, None
    tp = random.choice(all_pts)
    cw, ch = int(w * random.uniform(0.4, 0.7)), int(h * random.uniform(0.4, 0.7))
    x1, y1 = int(max(0, min(tp[0]-cw//2, w-cw))), int(max(0, min(tp[1]-ch//2, h-ch)))
    x2, y2 = x1 + cw, y1 + ch
    new_img = cv2.resize(img[y1:y2, x1:x2], TARGET_SIZE)
    new_data = json.loads(json.dumps(data))
    valid_shapes = []
    for s in new_data.get('shapes', []):
        if s.get('shape_type') == 'mask' and 'mask' in s:
            full_m = get_full_mask(s, h, w)
            cropped_m = full_m[y1:y2, x1:x2]
            cm = cv2.resize(cropped_m, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
            local_m, new_pts = extract_local_mask(cm)
            if local_m is not None:
                s['mask'] = encode_mask(local_m)
                s['points'] = new_pts
                valid_shapes.append(s)
        else:
            np_pts = (np.array(s['points']) - [x1, y1]) * [TARGET_SIZE[0]/cw, TARGET_SIZE[1]/ch]
            if np.any((np_pts[:, 0] >= 0) & (np_pts[:, 0] <= TARGET_SIZE[0]) & 
                      (np_pts[:, 1] >= 0) & (np_pts[:, 1] <= TARGET_SIZE[1])):
                s['points'] = np_pts.tolist()
                valid_shapes.append(s)
    if not valid_shapes: return None, None, None
    new_data['shapes'] = valid_shapes
    return new_img, new_data, "aug_crop"

def augment_color(img, data):
    new_img = cv2.convertScaleAbs(img, alpha=random.uniform(0.8, 1.2), beta=random.randint(-30, 30))
    return new_img, json.loads(json.dumps(data)), "aug_color"

def augment_noise_blur(img, data):
    new_img = img.copy()
    # С вероятностью 50% делаем либо размытие, либо шум
    if random.random() > 0.5:
        # Гауссовское размытие (эффект скорости / расфокуса)
        k = random.choice([3, 5, 7])
        new_img = cv2.GaussianBlur(new_img, (k, k), 0)
    else:
        # Гауссовский шум (эффект дешевой камеры / ночи)
        noise = np.random.normal(0, 15, new_img.shape).astype(np.uint8)
        new_img = cv2.add(new_img, noise)
        
    # Данные масок не меняются, геометрия осталась прежней!
    return new_img, json.loads(json.dumps(data)), "aug_noise"

def augment_cutout(img, data):
    new_img = img.copy()
    h, w = new_img.shape[:2]
    
    # Добавляем от 1 до 4 черных "заплаток"
    for _ in range(random.randint(1, 4)):
        # Размер заплатки от 5% до 15% экрана
        box_h = int(h * random.uniform(0.05, 0.15))
        box_w = int(w * random.uniform(0.05, 0.15))
        
        x = random.randint(0, w - box_w)
        y = random.randint(0, h - box_h)
        
        # Закрашиваем черным
        new_img[y:y+box_h, x:x+box_w] = (0, 0, 0)
        
    # Данные масок остаются! Сеть будет вынуждена догадываться о скрытом объекте
    return new_img, json.loads(json.dumps(data)), "aug_cutout"

# ==========================================
# ГЛАВНЫЙ ЗАПУСК (КАСКАДНАЯ ЛОГИКА)
# ==========================================
def main():
    print("\n" + "="*50, flush=True)
    print("🛠 СКРИПТ АУГМЕНТАЦИИ [TRAIN & VAL]", flush=True)
    print("="*50 + "\n", flush=True)

    # 1. ВЫБОР ПАПКИ
    print("📁 Выберите папки для обработки:")
    print("  1. Только data/train (Для обучения)")
    print("  2. Только data/val   (Для валидации)")
    print("  3. Обе папки         (train + val)")
    
    if len(sys.argv) > 2: # Если передано аргументами, например: python script.py 3 1
        folder_choice = sys.argv[2]
    else:
        print("\n⌨️ Ваш выбор папок (по умолчанию 1): ", end='', flush=True)
        folder_choice = sys.stdin.readline().strip()

    target_dirs = []
    if folder_choice == '2':
        target_dirs.append(Path("data/val"))
    elif folder_choice == '3':
        target_dirs.append(Path("data/train"))
        target_dirs.append(Path("data/val"))
    else:
        target_dirs.append(Path("data/train"))

    for d in target_dirs:
        if not d.exists():
            print(f"❌ Ошибка: Папка {d} не найдена!", flush=True)
            return

    # 2. ВЫБОР МЕТОДА
    print("\n🛠 Выберите методы (номера через запятую, например: 1,2,4):")
    print("  1. 🪞 Горизонтальное отражение (Flip)")
    print("  2. ☀️ Яркость/Контраст (Color)")
    print("  3. ✂️ Умная случайная обрезка (Smart Crop - ОСТОРОЖНО!)")
    print("  4. 🌫️ Шум и Размытие (Blur & Noise - ТОП ДЛЯ ФОРМ)")
    print("  5. ⬛ Перекрытия (Cutout - ПРОТИВ ПЕРЕОБУЧЕНИЯ)")
    print("  9. 🗑  УДАЛИТЬ ВСЕ АУГМЕНТАЦИИ из выбранных папок")
    print("  0. Выход")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print("\n⌨️ Ваш выбор методов: ", end='', flush=True)
        choice = sys.stdin.readline().strip()

    if not choice or '0' in choice:
        print("❌ Выход.", flush=True)
        return

    # 3. ОЧИСТКА
    if '9' in choice:
        del_count = 0
        for d in target_dirs:
            for f in d.glob("*_aug_*"):
                os.remove(f)
                del_count += 1
        print(f"🧹 Очистка завершена! Удалено файлов: {del_count}", flush=True)
        return

    mods = []
    if '1' in choice: mods.append((augment_flip, "aug_flip", "Отражение"))
    if '2' in choice: mods.append((augment_color, "aug_color", "Цвет"))
    if '3' in choice: mods.append((augment_crop, "aug_crop", "Обрезка"))
    if '4' in choice: mods.append((augment_noise_blur, "aug_noise", "Шум/Размытие"))
    if '5' in choice: mods.append((augment_cutout, "aug_cutout", "Перекрытия"))

    total_generated = 0

    # 4. ВЫПОЛНЕНИЕ ДЛЯ КАЖДОЙ ПАПКИ
    for current_dir in target_dirs:
        print("\n" + "="*50, flush=True)
        print(f"🚀 ЗАПУСК В ПАПКЕ: {current_dir}", flush=True)
        print("="*50, flush=True)

        for m_func, sfx_name, ru_name in mods:
            print(f"\n▶️ ЭТАП: {ru_name}...", flush=True)
            
            current_files = list(current_dir.glob("*.json"))
            files_to_process = [f for f in current_files if sfx_name not in f.name]
            
            print(f"   Файлов в очереди на обработку: {len(files_to_process)}", flush=True)
            
            step_count = 0
            for j_path in files_to_process:
                img_path = next((j_path.with_suffix(ext) for ext in ['.jpg','.png','.jpeg'] if j_path.with_suffix(ext).exists()), None)
                if not img_path: continue
                
                img = cv2.imread(str(img_path))
                with open(j_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                ni, nd, sfx = m_func(img, data)
                if ni is not None:
                    save_augmentation(img_path, ni, nd, sfx)
                    step_count += 1
                    total_generated += 1
                    
            print(f"   ✅ Создано файлов на этом этапе: {step_count}", flush=True)

    print("\n" + "="*50, flush=True)
    print(f"🎉 СЕССИЯ ЗАВЕРШЕНА! Всего сгенерировано новых файлов: {total_generated}", flush=True)
    print("="*50 + "\n", flush=True)

if __name__ == "__main__":
    main()