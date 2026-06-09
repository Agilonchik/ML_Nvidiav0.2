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
    gpus = tf.config.list_physical_devices("GPU")
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
    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_global_policy(policy)
    logger.info("⚡ Успех: Mixed Precision включен.")


def load_decoder_weights_only(model, weights_path: Path, logger) -> None:
    source_model = tf.keras.models.load_model(str(weights_path), compile=False)
    encoder_layer_names = getattr(model, "encoder_layer_names", set())
    loaded_layers = 0
    skipped_layers = 0

    for layer in model.layers:
        if layer.name in encoder_layer_names or not layer.weights:
            continue

        try:
            source_layer = source_model.get_layer(layer.name)
        except ValueError:
            skipped_layers += 1
            continue

        source_weights = source_layer.get_weights()
        target_weights = layer.get_weights()
        if not source_weights or [w.shape for w in source_weights] != [
            w.shape for w in target_weights
        ]:
            skipped_layers += 1
            continue

        layer.set_weights(source_weights)
        loaded_layers += 1

    logger.info(
        "✅ Загружены веса декодера/головы: "
        f"{loaded_layers} слоев. Пропущено: {skipped_layers}."
    )


def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    config = load_config()

    # --- НАСТРОЙКА АРГУМЕНТОВ КОМАНДНОЙ СТРОКИ ---
    parser = argparse.ArgumentParser(description="Запуск обучения U-Net")
    parser.add_argument(
        "epochs",
        nargs="?",
        type=int,
        default=None,
        help="Количество эпох (если не указано, берется из config.yaml)",
    )

    # Параметр --unfreeze. Если он есть в команде, значение True, если нет — False.
    parser.add_argument(
        "--unfreeze",
        action="store_true",
        help="Разморозить энкодер (MobileNet) для Fine-tuning",
    )
    parser.add_argument(
        "--encoder-weights",
        choices=["imagenet", "none"],
        default=None,
        help=(
            "Начальные веса энкодера: imagenet — предобученные, "
            "none — случайная инициализация без предобученного энкодера"
        ),
    )
    parser.add_argument(
        "--fresh-start",
        action="store_true",
        help="Не загружать models/final_model.keras перед стартом обучения",
    )
    parser.add_argument(
        "--load-decoder-only",
        action="store_true",
        help=(
            "Загрузить из models/final_model.keras только веса декодера/головы, "
            "оставив текущие веса энкодера"
        ),
    )

    args = parser.parse_args()
    if args.fresh_start and args.load_decoder_only:
        parser.error("--fresh-start и --load-decoder-only нельзя использовать вместе")

    logger = setup_logger("TrainerLogger", Path(config["paths"]["logs_dir"]))

    # Обработка количества эпох
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
        logger.info(f"🔄 Количество эпох переопределено -> {args.epochs}")
    else:
        logger.info(
            f"📄 Используются эпохи из конфига -> {config['training']['epochs']}"
        )

    # Логируем режим работы энкодера
    encoder_weights = args.encoder_weights or config["model"].get(
        "encoder_weights", "imagenet"
    )
    if encoder_weights == "none":
        encoder_weights = None

    # Если энкодер стартует со случайных весов, его нельзя оставлять замороженным.
    trainable_encoder = args.unfreeze or encoder_weights is None
    mode_text = (
        "🌱 SCRATCH (энкодер без предобученных весов, разморожен)"
        if encoder_weights is None
        else (
            "🔥 FINE-TUNING (энкодер ImageNet разморожен)"
            if trainable_encoder
            else "❄️ WARM-UP (энкодер ImageNet заморожен)"
        )
    )
    logger.info(f"🚀 РЕЖИМ ОБУЧЕНИЯ: {mode_text}")

    tf.random.set_seed(config["project"]["seed"])

    try:
        setup_gpu_and_precision(logger)

        pipeline = DataPipeline(config, logger)
        train_ds, val_ds, num_classes, train_count, val_count = (
            pipeline.create_datasets()
        )

        # 1. Строим модель, передавая статус энкодера и источник его весов
        builder = ModelBuilder(config, num_classes, logger)
        model = builder.build(
            trainable_encoder=trainable_encoder, encoder_weights=encoder_weights
        )

        # --- 🚀 ДООБУЧЕНИЕ (ОПЦИОНАЛЬНАЯ ЗАГРУЗКА ВЕСОВ) ---
        final_model_path = Path(config["paths"]["models_dir"]) / "final_model.keras"
        if args.fresh_start:
            logger.info("🌟 Fresh-start включен: старые веса не загружаем.")
        elif args.load_decoder_only:
            if not final_model_path.exists():
                raise FileNotFoundError(
                    f"Для --load-decoder-only нужна модель: {final_model_path}"
                )
            logger.info(
                "🔄 Загружаем только декодер/голову из "
                f"{final_model_path}; энкодер оставляем текущим."
            )
            load_decoder_weights_only(model, final_model_path, logger)
        elif final_model_path.exists():
            logger.info(f"🔄 НАЙДЕНА МОДЕЛЬ! Загружаем веса из {final_model_path}...")
            # Загружаем веса в новую архитектуру
            model.load_weights(str(final_model_path))
            logger.info("✅ Веса успешно загружены. Продолжаем обучение!")
        else:
            logger.info(
                "🌟 Старая модель не найдена. Начинаем с нуля (или с весов ImageNet)."
            )

        trainer = Trainer(model, config, logger)
        history = trainer.train(train_ds, val_ds, train_count, val_count)

        evaluator = Evaluator(model, config, logger)
        evaluator.evaluate(val_ds, history)

        logger.info("Segmentation training finished successfully.")

    except Exception as e:
        logger.exception(f"Fatal error during training: {e}")


if __name__ == "__main__":
    main()
