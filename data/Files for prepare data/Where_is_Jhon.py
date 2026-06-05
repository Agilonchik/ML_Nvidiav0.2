#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
move_pairs.py

Ищет в заданной директории файлы с одинаковым именем (без расширения),
но разными расширениями, и перемещает их все в указанную пользователем подпапку.

Требования:
    Python ≥ 3.6
    pip install tqdm   # для красивого прогресс‑бара (необязательно)
"""

import importlib
import logging
import shutil
from collections import defaultdict
from pathlib import Path

_tqdm_spec = importlib.util.find_spec("tqdm")
if _tqdm_spec is None:

    def tqdm(iterable, **kwargs):
        return iterable

else:
    tqdm = importlib.import_module("tqdm").tqdm


# ----------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------
def ask_path(prompt: str, must_exist: bool = False) -> Path:
    """Запрашивает у пользователя путь к файлу/каталогу."""
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


def safe_move(src: Path, dst_dir: Path) -> Path:
    """
    Перемещает файл src в каталог dst_dir.
    При конфликте имён (файл с тем же именем уже есть) добавляет суффикс _1, _2 … .
    Возвращает путь к новому файлу.
    """
    if not dst_dir.is_dir():
        dst_dir.mkdir(parents=True, exist_ok=True)

    candidate = dst_dir / src.name
    counter = 1
    while candidate.exists():
        candidate = dst_dir / f"{src.stem}_{counter}{src.suffix}"
        counter += 1

    shutil.move(str(src), str(candidate))
    return candidate


def find_pairs(root: Path, exclude_dir: Path) -> dict[Path, list[Path]]:
    """
    Возвращает словарь: stem → список файлов с этим stem,
    где в списке минимум два файла с разными расширениями.
    Файлы из `exclude_dir` (и её подпапок) игнорируются.
    """
    groups = defaultdict(list)

    for p in root.rglob("*.*"):  # только файлы, у которых есть расширение
        if not p.is_file():
            continue
        # Не учитываем файлы, которые уже находятся в целевой папке
        try:
            p.relative_to(exclude_dir)
            continue  # файл внутри exclude_dir → пропускаем
        except ValueError:
            pass

        groups[p.stem].append(p)

    # Оставляем только те стемы, у которых ≥2 разных расширения
    result = {stem: files for stem, files in groups.items() if len(files) >= 2}
    return result


# ----------------------------------------------------------------------
# Основная логика программы
# ----------------------------------------------------------------------
def main() -> None:
    print("=== Переместить файлы‑пары ===")
    src_dir = ask_path("Введите путь к исходной папке: ", must_exist=True)

    # Целевая подпапка может быть внутри, а может и вне исходного каталога.
    dst_dir = ask_path("Введите путь к целевой подпапке (будет создана, если её нет): ")
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Находим группы одинаковых имён
    print("\nПоиск файлов…")
    pairs = find_pairs(src_dir, exclude_dir=dst_dir)
    total_groups = len(pairs)
    total_files = sum(len(v) for v in pairs.values())

    if total_groups == 0:
        print("Не найдено ни одной группы одинаковых имён с разными расширениями.")
        return

    print(f"Найдено {total_groups} групп, всего {total_files} файлов.\n")

    # Перемещаем файлы
    moved = 0
    for stem, files in tqdm(pairs.items(), desc="Перемещение", unit="group"):
        for f in files:
            new_path = safe_move(f, dst_dir)
            logging.debug(f"Перемещён {f} → {new_path}")
            moved += 1

    # Итоги
    print("\n=== Результат ===")
    print(f"Групп перемещено : {total_groups}")
    print(f"Файлов перемещено: {moved}")
    print(f"Все файлы‑пары теперь находятся в «{dst_dir}».")


if __name__ == "__main__":
    # Уровень логов INFO – переключить на DEBUG, если хотите увидеть каждый файл.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        main()
    except KeyboardInterrupt:  # Ctrl+C в терминале
        print("\nПрервано пользователем.")
