import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_loss_decomposition(csv_path="artifacts/loss_log.csv"):
    if not Path(csv_path).exists():
        print(f"❌ Файл {csv_path} не найден. Сначала заверши хотя бы одну эпоху обучения!")
        return

    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(10, 6))
    
    # Список компонентов
    cols = ['focal', 'dice', 'entropy', 'area']
    available_cols = [c for c in cols if c in df.columns]
    
    # Используем стандартный stackplot от matplotlib
    plt.stackplot(df['epoch'], 
                  [df[c] for c in available_cols], 
                  labels=available_cols, 
                  alpha=0.7)
    
    # Добавляем общую линию val_loss для наглядности
    if 'val_loss' in df.columns:
        plt.plot(df['epoch'], df['val_loss'], color='black', linestyle='--', label='Total Val Loss')

    plt.title("Loss Decomposition: Анализ ошибок", fontsize=14)
    plt.xlabel("Эпоха")
    plt.ylabel("Значение штрафа")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    output_img = "artifacts/loss_analysis_plot.png"
    plt.savefig(output_img)
    print(f"📈 График построен и сохранен в {output_img}")
    plt.show()

if __name__ == "__main__":
    plot_loss_decomposition()