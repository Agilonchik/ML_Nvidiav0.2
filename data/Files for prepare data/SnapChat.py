#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
select_photos.py

Интерактивный отбор PNG‑фотографий.
При запуске программа запрашивает у пользователя:
    • путь к папке с исходными изображениями;
    • путь к папке экспорта (будет создана, если её нет).

После этого последовательно открываются все найденные *.png‑файлы
(включая подпапки).  В окне изображения пользователь нажимает:
    – Y   → файл перемещается в папку экспорта;
    – N   → файл удаляется из исходного каталога;
    – ESC или Q → досрочно завершить работу скрипта.
"""

import logging
from pathlib import Path

import cv2
from tqdm import tqdm


# ------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------
def ask_directory(prompt: str, must_exist: bool = False) -> Path:
    """
    Запрашивает у пользователя путь к директории.

    :param prompt: Текст приглашения.
    :param must_exist: Если True – проверяем, что каталог существует,
                       иначе будем запрашивать снова.
    :return: pathlib.Path (абсолютный и нормализованный).
    """
    while True:
        raw = input(prompt).strip()
        if not raw:
            print("Путь не может быть пустым. Попробуйте ещё раз.")
            continue

        p = Path(raw).expanduser().resolve()

        if must_exist and not p.is_dir():
            print(f"Указанный путь «{p}» не существует или не является папкой.")
            continue

        return p


def show_image_and_wait(img_path: Path) -> int:
    """
    Открывает изображение в окне OpenCV и ждёт нажатия клавиши.

    :param img_path: Полный путь к файлу изображения.
    :return: Код клавиши, полученный от cv2.waitKey().
    """
    # Читаем файл (может быть повреждён)
    img = cv2.imread(str(img_path))
    if img is None:
        return -1  # Специальный код – ошибка чтения

    # При необходимости уменьшаем изображение, чтобы оно помещалось на экран
    screen_h, screen_w = 800, 1200   # примерные ограничения; можно изменить
    h, w = img.shape[:2]
    scale = min(screen_w / w, screen_h / h, 1.0)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    cv2.imshow("Photo – press Y (keep) / N (delete) / ESC/Q (quit)", img)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()
    return key


def process_file(src_path: Path, dst_root: Path, dry_run: bool = False) -> str:
    """
    Обрабатывает один файл в зависимости от нажатой клавиши.

    :param src_path: Путь к исходному файлу.
    :param dst_root: Каталог экспорта.
    :param dry_run: Если True – не меняем файловую систему, только считаем.
    :return: Статус обработки – 'kept', 'deleted', 'skipped' или 'error'.
    """
    key = show_image_and_wait(src_path)

    # Ошибка чтения изображения
    if key == -1:
        logging.error(f"Не удалось открыть изображение {src_path}")
        return "error"

    # ESC (27) или Q/q – досрочное завершение
    if key in (27, ord('q'), ord('Q')):
        raise KeyboardInterrupt

    # Y / y – оставить (переместить)
    if key in (ord('y'), ord('Y')):
        if not dry_run:
            # Если в целевой папке уже есть файл с таким именем,
            # добавляем суффикс _1, _2 и т.д.
            dst_path = dst_root / src_path.name
            counter = 1
            while dst_path.exists():
                dst_path = dst_root / f"{src_path.stem}_{counter}{src_path.suffix}"
                counter += 1
            try:
                src_path.rename(dst_path)
            except Exception as exc:
                logging.error(f"Не удалось переместить {src_path} → {dst_path}: {exc}")
                return "error"
        return "kept"

    # N / n – удалить
    if key in (ord('n'), ord('N')):
        if not dry_run:
            try:
                src_path.unlink()
            except Exception as exc:
                logging.error(f"Не удалось удалить файл {src_path}: {exc}")
                return "error"
        return "deleted"

    # Любая другая клавиша – считаем, что пользователь пропустил файл
    return "skipped"


# ------------------------------------------------------------
# Основная часть программы
# ------------------------------------------------------------
def main() -> None:
    # ------------------------------
    # Запрос путей у пользователя
    # ------------------------------
    print("=== Photo‑selector 0.2 ===")
    src_root = ask_directory(
        "Введите путь к папке с исходными PNG‑фото: ", must_exist=True
    )
    dst_root = ask_directory(
        "Введите путь к папке экспорта (будет создана, если её нет): "
    )

    # Если целевая папка не существует – создаём её сразу
    dst_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------
    # Поиск всех PNG‑файлов (рекурсивно)
    # ------------------------------
    png_files = list(src_root.rglob("*.png"))
    if not png_files:
        logging.warning(f"В каталоге {src_root} не найдено файлов *.png.")
        return

    logging.info(
        f"Найдено {len(png_files)} PNG‑файлов (включая подпапки) в {src_root}."
    )

    # ------------------------------
    # Обработка файлов
    # ------------------------------
    kept = deleted = skipped = errors = 0
    try:
        for file_path in tqdm(png_files, desc="Обрабатываем", unit="file"):
            try:
                status = process_file(file_path, dst_root, dry_run=False)
                if status == "kept":
                    kept += 1
                elif status == "deleted":
                    deleted += 1
                elif status == "skipped":
                    skipped += 1
                else:  # error
                    errors += 1
            except KeyboardInterrupt:
                logging.info("Прерывание пользователем (ESC/Q). Завершаем работу.")
                break
    finally:
        cv2.destroyAllWindows()

    # ------------------------------
    # Вывод статистики
    # ------------------------------
    print("\n=== Статистика ===")
    print(f"Отобрано (перемещено) : {kept}")
    print(f"Отклонено (удалено)   : {deleted}")
    print(f"Пропущено            : {skipped}")
    if errors:
        print(f"Ошибки при чтении/обработке: {errors}")

    remaining = len(list(src_root.rglob("*.png")))
    print(f"\nОсталось файлов в исходной папке: {remaining}")
    print("Готово.")


if __name__ == "__main__":
    # Настраиваем простой логгер (по умолчанию INFO)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
