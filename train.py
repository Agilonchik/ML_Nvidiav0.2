import os
import argparse
import tensorflow as tf
from tensorflow.keras import mixed_precision
from src.utils.config_parser import load_config
from src.utils.logger import setup_logger
from src.data.dataset import DataPipeline
from src.models.model_builder import ModelBuilder  # Исправлен импорт с большой буквы
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator
from pathlib import Path

def setup_gpu_and_precision(logger):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"✅ Успех: Memory Growth включен для {len(gpus)} GPU.")
        except RuntimeError as e:
            logger.error(f"⚠️ Ошибка при настройке GPU: {e}")
    else:
        logger.warning("⚠️ GPU не найден, обучение будет идти на CPU.")

    # Включаем смешанную точность для ускорения
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    logger.info(f"⚡ Успех: Mixed Precision включен.")

def main():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    config = load_config()
    
    # --- НАСТРОЙКА АРГУМЕНТОВ КОМАНДНОЙ СТРОКИ ---
    parser = argparse.ArgumentParser(description="Запуск обучения U-Net")
    parser.add_argument("epochs", nargs="?", type=int, default=None, 
                        help="Количество эпох (если не указано, берется из config.yaml)")
    
    # Параметр --unfreeze. Если он есть в команде, значение True, если нет — False.
    parser.add_argument("--unfreeze", action="store_true", 
                        help="Разморозить предобученный энкодер (MobileNet) для Fine-tuning")
    
    args = parser.parse_args()

    logger = setup_logger("TrainerLogger", Path(config["paths"]["logs_dir"]))
    
    # Обработка количества эпох
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
        logger.info(f"🔄 Количество эпох переопределено -> {args.epochs}")
    else:
        logger.info(f"📄 Используются эпохи из конфига -> {config['training']['epochs']}")

    # Логируем режим работы энкодера
    unfreeze_mode = args.unfreeze
    mode_text = "🔥 FINE-TUNING (Энкодер разморожен)" if unfreeze_mode else "❄️ WARM-UP (Энкодер заморожен)"
    logger.info(f"🚀 РЕЖИМ ОБУЧЕНИЯ: {mode_text}")

    tf.random.set_seed(config["project"]["seed"])

    try:
        setup_gpu_and_precision(logger)

        pipeline = DataPipeline(config, logger)
        train_ds, val_ds, num_classes, train_count, val_count = pipeline.create_datasets()

        # 1. Строим модель, передавая статус разморозки энкодера
        builder = ModelBuilder(config, num_classes, logger)
        model = builder.build(trainable_encoder=unfreeze_mode)

        # --- 🚀 МАГИЯ ДООБУЧЕНИЯ (ЗАГРУЗКА ВЕСОВ) ---
        final_model_path = Path(config["paths"]["models_dir"]) / "final_model.keras"
        if final_model_path.exists():
            logger.info(f"🔄 НАЙДЕНА МОДЕЛЬ! Загружаем веса из {final_model_path}...")
            # Загружаем веса в новую архитектуру
            model.load_weights(str(final_model_path))
            logger.info("✅ Веса успешно загружены. Продолжаем обучение!")
        else:
            logger.info("🌟 Старая модель не найдена. Начинаем с нуля (или с весов ImageNet).")

        trainer = Trainer(model, config, logger)
        history = trainer.train(train_ds, val_ds, train_count, val_count)

        evaluator = Evaluator(model, config, logger)
        evaluator.evaluate(val_ds, history)

        logger.info("Segmentation training finished successfully.")

    except Exception as e:
        logger.exception(f"Fatal error during training: {e}")

if __name__ == "__main__":
    main()