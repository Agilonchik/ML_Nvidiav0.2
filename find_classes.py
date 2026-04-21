import json
from pathlib import Path

def find_all_classes():
    unique_classes = set()
    
    # Ищем во всех папках
    for split in ['train', 'val']:
        folder = Path(f"data/{split}")
        if folder.exists():
            for j_path in folder.glob("*.json"):
                with open(j_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'shapes' in data:
                        for shape in data['shapes']:
                            label = shape.get('label', '')
                            # Добавляем только если это не пустая строка
                            if label: 
                                unique_classes.add(label)
    
    # Сортируем по алфавиту и добавляем Background первым
    final_list = ["Background"] + sorted(list(unique_classes))
    
    print("\n" + "="*50)
    print("✅ ГОТОВО! Скопируй строчку ниже и вставь в configs/config.yaml:")
    print("="*50 + "\n")
    # Красиво печатаем в формате YAML
    print(f"classes: {final_list}\n")

if __name__ == "__main__":
    find_all_classes()