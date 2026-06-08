import random
import shutil
from pathlib import Path

def move_train_to_val(percentage=0.10):
    train_dir = Path("data/train")
    val_dir = Path("data/val")

    if not train_dir.exists() or not val_dir.exists():
        print("❌ Ошибка: Папки data/train или data/val не найдены!")
        return

    # Ищем только оригинальные JSON файлы (без пометок _aug_)
    json_files = [f for f in train_dir.glob("*.json") if "_aug_" not in f.name]
    total_files = len(json_files)

    if total_files == 0:
        print("❌ Ошибка: В папке data/train нет оригинальных JSON файлов!")
        return

    # Считаем нужное количество (10%)
    num_to_move = int(total_files * percentage)
    if num_to_move == 0:
        num_to_move = 1 # Переместим хотя бы один файл

    print("="*50)
    print(f"📊 Всего чистых файлов в train: {total_files}")
    print(f"🚀 Планируется переместить в val: {num_to_move} ({(percentage*100):.0f}%)")
    print("="*50)

    # Выбираем случайные файлы
    files_to_move = random.sample(json_files, num_to_move)

    moved_count = 0
    for j_path in files_to_move:
        # Ищем парную картинку
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            if j_path.with_suffix(ext).exists():
                img_path = j_path.with_suffix(ext)
                break
            elif j_path.with_suffix(ext.upper()).exists():
                img_path = j_path.with_suffix(ext.upper())
                break

        if img_path:
            # Перемещаем JSON
            shutil.move(str(j_path), str(val_dir / j_path.name))
            # Перемещаем Картинку
            shutil.move(str(img_path), str(val_dir / img_path.name))
            moved_count += 1
        else:
            print(f"⚠️ Внимание: Для {j_path.name} не найдена картинка, пропускаем.")

    print(f"\n✅ Успешно перемещено {moved_count} пар (Картинка + JSON) в папку data/val!")

if __name__ == "__main__":
    move_train_to_val(0.10) # 0.10 означает 10%