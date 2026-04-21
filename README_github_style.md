# U-Net Segmentation Tool

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Keras](https://img.shields.io/badge/Keras-2.15-red)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-ff4b4b)
![Platform](https://img.shields.io/badge/Platform-Windows%2011%20%2B%20WSL-success)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

Production-oriented проект для **сегментации изображений** на **TensorFlow / Keras** с поддержкой:

- обучения модели по разметке **LabelMe JSON**;
- визуального контроля качества через **Streamlit**;
- диагностики результатов;
- аугментации датасета;
- инференса по одному изображению или по папке.

---

## Содержание

- [О проекте](#о-проекте)
- [Основные возможности](#основные-возможности)
- [Структура проекта](#структура-проекта)
- [Требования](#требования)
- [Установка](#установка)
- [Быстрый старт](#быстрый-старт)
- [Формат данных](#формат-данных)
- [Конфигурация](#конфигурация)
- [Основные сценарии работы](#основные-сценарии-работы)
- [Команды запуска](#команды-запуска)
- [Streamlit-интерфейс](#streamlit-интерфейс)
- [Диагностика и артефакты](#диагностика-и-артефакты)
- [Типичные ошибки](#типичные-ошибки)
- [Рекомендации по эксплуатации](#рекомендации-по-эксплуатации)
- [Раздел для разработчиков](#раздел-для-разработчиков)

---

## О проекте

Проект предназначен для обучения и использования нейросети сегментации изображений по пользовательской разметке.

Система строится вокруг следующего workflow:

1. подготовка изображений и LabelMe JSON;
2. настройка списка классов;
3. обучение модели;
4. визуальная проверка результата;
5. диагностика качества на валидации;
6. запуск инференса на новых данных.

---

## Основные возможности

### Обучение
- обучение модели сегментации на TensorFlow / Keras;
- сохранение лучшего чекпоинта и финальной модели;
- поддержка продолжения обучения по сохранённым весам;
- mixed precision и настройка GPU.

### Работа с данными
- чтение разметки **LabelMe JSON**;
- поддержка `mask`, `polygon`, `rectangle`;
- автоматический сбор списка классов;
- перенос части train-данных в val;
- аугментация датасета.

### Визуализация
- сравнение оригинала, ручной разметки и предсказания модели;
- просмотр по классам в режиме HIL;
- диагностические панели на валидационной выборке;
- сохранение цветных overlay-предсказаний.

### Инженерные возможности
- модульная структура проекта;
- конфигурация через `config.yaml`;
- логи обучения;
- сохранение артефактов в отдельные директории.

---

## Структура проекта

```text
.
├── app.py
├── train.py
├── predict.py
├── diag_scr.py
├── augment_data.py
├── find_classes.py
├── move_to_val.py
├── check_gpu.py
├── loss_analyzer.py
├── Boot.py
├── configs/
│   └── config.yaml
├── data/
│   ├── train/
│   ├── val/
│   ├── unlabelled/
│   └── train_pl/
├── models/
├── checkpoints/
├── logs/
├── artifacts/
└── src/
    ├── data/
    ├── models/
    ├── training/
    ├── evaluation/
    ├── inference/
    └── utils/
```

### Назначение ключевых файлов

| Файл | Назначение |
|---|---|
| `app.py` | Streamlit-интерфейс для визуального контроля |
| `train.py` | Обучение модели |
| `predict.py` | Инференс по картинке или папке |
| `diag_scr.py` | Диагностика на валидационных данных |
| `augment_data.py` | Аугментация изображений и масок |
| `find_classes.py` | Поиск всех классов в JSON-разметке |
| `move_to_val.py` | Перенос части train в val |
| `check_gpu.py` | Проверка GPU |
| `loss_analyzer.py` | Построение графика анализа loss |

---

## Требования

### Платформа
- Windows 11
- рекомендуется запуск через **WSL**

### Python
- Python **3.10+**

### Основные зависимости
- TensorFlow 2.15
- Keras 2.15
- OpenCV
- NumPy
- Pandas
- Matplotlib
- PyYAML
- Streamlit

---

## Установка

### 1. Создать окружение

```bash
python -m venv unet-env
```

### 2. Активировать окружение

#### Linux / WSL
```bash
source unet-env/bin/activate
```

#### Windows PowerShell
```powershell
unet-env\Scripts\Activate.ps1
```

### 3. Установить зависимости

#### Универсальный вариант
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### Для Linux / WSL
```bash
pip install -r requirements_linux.txt
```

---

## Быстрый старт

### Шаг 1. Подготовить данные
Разместите изображения и соответствующие JSON-файлы в папках:

```text
data/train/
data/val/
```

### Шаг 2. Собрать список классов
```bash
python find_classes.py
```

Скопируйте выведенный список в `configs/config.yaml`.

### Шаг 3. Проверить GPU
```bash
python check_gpu.py
```

### Шаг 4. Обучить модель
```bash
python train.py
```

### Шаг 5. Запустить интерфейс
```bash
streamlit run app.py
```

### Шаг 6. Построить диагностику
```bash
python diag_scr.py
```

### Шаг 7. Запустить инференс
```bash
python predict.py --image path/to/image.png
```

---

## Формат данных

### Структура датасета

Для каждого изображения должен существовать JSON с тем же именем:

```text
data/train/
  sample_001.png
  sample_001.json
  sample_002.jpg
  sample_002.json
```

Поддерживаемые форматы изображений:

- `.png`
- `.jpg`
- `.jpeg`
- `.bmp`

### Формат разметки

Используется формат, совместимый с **LabelMe**.

Пример:

```json
{
  "version": "5.11.3",
  "flags": {},
  "shapes": [
    {
      "label": "BS_CL_intermediate",
      "points": [[75.0, 263.0], [717.0, 430.0]],
      "shape_type": "mask",
      "mask": "..."
    }
  ],
  "imagePath": "178352548.png",
  "imageHeight": 599,
  "imageWidth": 800
}
```

### Поддерживаемые типы фигур
- `mask`
- `polygon`
- `rectangle`

### Важно
- названия классов в JSON должны совпадать с классами в конфиге;
- `Background` должен быть первым классом;
- рядом с каждым JSON должен лежать файл изображения.

---

## Конфигурация

Основной конфиг:

```text
configs/config.yaml
```

Пример:

```yaml
project:
  name: "unet_segmentation_v1"
  seed: 42

paths:
  data_dir: "data"
  models_dir: "models"
  logs_dir: "logs"
  artifacts_dir: "artifacts"
  checkpoints_dir: "checkpoints"

data:
  image_size: [512, 512]
  batch_size: 16
  classes:
    - Background
    - BS_CL_es
    - BS_CL_intermediate
  use_augmentation: false
  cache: false

model:
  type: "unet"
  learning_rate: 0.00005

training:
  epochs: 30
  early_stopping_patience: 12
  reduce_lr_patience: 6
  reduce_lr_factor: 0.5

inference:
  alpha: 0.5
```

### Ключевые параметры

| Параметр | Описание |
|---|---|
| `image_size` | Размер изображения на вход сети |
| `batch_size` | Размер батча |
| `classes` | Список классов |
| `learning_rate` | Скорость обучения |
| `epochs` | Количество эпох |
| `alpha` | Прозрачность overlay-маски |

---

## Основные сценарии работы

## 1. Обучение модели

Базовый запуск:

```bash
python train.py
```

Запуск с переопределением числа эпох:

```bash
python train.py 50
```

Запуск fine-tuning режима:

```bash
python train.py 30 --unfreeze
```

Что делает `train.py`:
- загружает конфиг;
- настраивает GPU и mixed precision;
- строит датасеты;
- создает модель;
- при наличии `models/final_model.keras` загружает веса;
- обучает модель;
- сохраняет артефакты.

---

## 2. Аугментация данных

Интерактивный запуск:

```bash
python augment_data.py
```

Доступные методы:

1. Горизонтальное отражение
2. Яркость / контраст
3. Умная случайная обрезка
4. Шум / размытие
5. Cutout / перекрытия
9. Удалить все аугментации

Аугментированные файлы создаются рядом с исходными и получают суффикс вида:

```text
sample_001_aug_flip_ab12.png
sample_001_aug_flip_ab12.json
```

---

## 3. Разделение train / val

```bash
python move_to_val.py
```

По умолчанию скрипт переносит 10% оригинальных данных из `train` в `val`.

---

## 4. Инференс

### По одному изображению
```bash
python predict.py --image path/to/image.png
```

### По папке
```bash
python predict.py --folder path/to/folder
```

Результаты сохраняются в `artifacts/`.

---

## 5. Диагностика

```bash
python diag_scr.py
```

Скрипт:
- берет случайные образцы из `data/val`;
- строит панель сравнения;
- сохраняет итог в:

```text
artifacts/diagnostics.png
```

---

## Команды запуска

```bash
# Проверка GPU
python check_gpu.py

# Поиск классов
python find_classes.py

# Перенос части train в val
python move_to_val.py

# Аугментация
python augment_data.py

# Обучение
python train.py
python train.py 50
python train.py 30 --unfreeze

# Инференс
python predict.py --image path/to/image.png
python predict.py --folder path/to/folder

# Диагностика
python diag_scr.py

# Streamlit UI
streamlit run app.py

# Анализ loss
python loss_analyzer.py
```

---

## Streamlit-интерфейс

Запуск:

```bash
streamlit run app.py
```

### Доступные режимы

#### 1. Просмотр разметки (HIL)
Позволяет:
- выбрать папку `train` или `val`;
- выбрать JSON-файл;
- увидеть:
  - оригинальное изображение;
  - ручную разметку;
  - предсказание сети;
- открыть debug-панель;
- просмотреть сравнение по классам.

#### 2. Авто-разметка новых данных
В текущей версии раздел отображается, но пока не реализован полностью.

---

## Диагностика и артефакты

После работы проекта полезные файлы появляются в следующих директориях:

### `models/`
- `final_model.keras`

### `checkpoints/`
- `best_model.keras`

### `logs/`
- `training_log.csv`
- TensorBoard-логи

### `artifacts/`
- `training_history.png`
- `diagnostics.png`
- `pred_*.png`
- `label_map.json`
- `loss_log.csv`
- `loss_analysis_plot.png`

---

## Типичные ошибки

### `Config not found`
Проверьте наличие файла:

```text
configs/config.yaml
```

### `Model not found`
Проверьте, существует ли:

```text
models/final_model.keras
```

### Streamlit не показывает файлы
Проверьте, что в `data/train` или `data/val` есть JSON-файлы.

### Для JSON не находится изображение
Проверьте, что имена совпадают:

```text
sample_01.json
sample_01.png
```

### Класс пропущен
Проверьте:
- название `label` в JSON;
- наличие класса в `config.yaml`;
- лишние пробелы и различия в регистре.

### TensorFlow не видит GPU
Возможные причины:
- не установлен корректный GPU-стек;
- нет CUDA-поддержки;
- TensorFlow установлен в CPU-конфигурации;
- WSL/GPU passthrough настроен неправильно.

---

## Рекомендации по эксплуатации

- держите `Background` первым классом;
- храните оригинальные и аугментированные данные вместе, но следите за суффиксами `_aug_`;
- перед обучением проверьте корректность всех `label`;
- не меняйте архитектуру модели между перезапусками дообучения без необходимости;
- после каждой значимой сессии обучения сохраняйте артефакты и проверяйте `diagnostics.png`;
- отдельно контролируйте, как датапайплайн интерпретирует `mask`-разметку LabelMe.

---

## Раздел для разработчиков

### Архитектурные блоки

- `src/data/` — подготовка датасета;
- `src/models/` — сборка модели;
- `src/training/` — обучение и callbacks;
- `src/evaluation/` — графики и оценка;
- `src/inference/` — предсказание и сохранение результатов;
- `src/utils/` — логгер, конфиг-парсер, вспомогательные callbacks.

### Что важно учитывать

1. `train.py` ожидает совместимую архитектуру модели при загрузке старых весов.
2. `app.py` и `diag_scr.py` используют собственную логику чтения масок из JSON.
3. `LossDecompositionCallback` работает только если текущая функция потерь поддерживает соответствующий интерфейс.
4. `Boot.py` — это набор команд, а не полноценный исполняемый Python-модуль.

### Рекомендуемые улучшения

- довести до конца режим авто-разметки в `app.py`;
- унифицировать код декодирования LabelMe-масок в одном модуле;
- добавить unit-тесты на чтение JSON и сборку масок;
- добавить экспорт метрик IoU / Dice по классам;
- оформить `Boot.py` как shell-скрипт или Makefile.

---

## Примечание

Этот README описывает текущую структуру и поведение проекта по предоставленному коду. Если архитектура модели, формат датасета или пайплайн будут изменены, документацию тоже нужно обновить.

## Порядок запуска в с любого диска

В порядке ввести в терминад VSC следующие команды
cd $HOME
wsl
Буквой f обозначен диск с которого производится попытка запустить
  sudo mkdir -p /mnt/f
    Пароль Parallelogramm
  sudo mount -t drvfs F: /mnt/f
Проверка пути:
ls /mnt/f
ls "/mnt/f/Programms"
find /mnt/f -maxdepth 4 -type f -name train.py 2>/dev/null
Если путь нашелся то:
cd "/mnt/f/Programms/G_Intel_4"
source ~/unet-env/bin/activate
python train.py