import os
from pathlib import Path

def pack_project_for_ai(output_filename="project_code_for_ai.txt"):
    # 1. Папки, куда скрипту лезть НЕ НАДО (чтобы не зависнуть на гигабайтах данных)
    IGNORE_DIRS = {'.venv', 'unet-env', '__pycache__', '.git', 'data', 'models', 'logs', 'checkpoints', 'artifacts'}
    
    # 2. Расширения файлов, которые нам нужны (только текст/код)
    ALLOWED_EXTENSIONS = {'.py', '.yaml', '.txt', '.md', '.json'}

    base_dir = Path('.')
    
    print("Начинаю сборку проекта...")
    
    with open(output_filename, 'w', encoding='utf-8') as outfile:
        # Пишем заголовок
        outfile.write("ОПИСАНИЕ ПРОЕКТА ДЛЯ ИИ\n")
        outfile.write("Структура и исходный код\n\n")
        
        for root, dirs, files in os.walk(base_dir):
            # Исключаем ненужные папки "на лету"
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for file in files:
                file_path = Path(root) / file
                
                # Игнорируем сам скрипт сборки
                if file_path.name == "pack_for_ai.py":
                    continue
                    
                if file_path.suffix in ALLOWED_EXTENSIONS:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            content = infile.read()
                            
                        # Красивый разделитель для каждого файла
                        outfile.write(f"\n{'='*60}\n")
                        outfile.write(f"ФАЙЛ: {file_path.relative_to(base_dir)}\n")
                        outfile.write(f"{'='*60}\n\n")
                        outfile.write(content)
                        outfile.write("\n")
                        
                        print(f"Добавлен: {file_path.relative_to(base_dir)}")
                    except Exception as e:
                        print(f"⚠️ Пропущен файл {file_path} (ошибка чтения: {e})")

    print(f"\n✅ Успех! Весь код аккуратно упакован в файл: {output_filename}")

if __name__ == "__main__":
    pack_project_for_ai()