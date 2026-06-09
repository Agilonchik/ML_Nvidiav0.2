import tensorflow as tf
from tensorflow.keras import layers, models
import logging


# ==========================================
# 🎯 ГИБРИДНАЯ ФУНКЦИЯ ПОТЕРЬ
# ==========================================
def spatial_aware_loss(
    num_classes,
    class_weights=None,
    gamma=2.0,
    focal_weight=5.0,
    dice_weight=6.0,
    area_weight=2.0,
    entropy_weight=0.05,
    tv_weight=5e-6,
):

    if class_weights is None:
        class_weights = tf.ones([num_classes], dtype=tf.float32)
    else:
        class_weights = tf.constant(class_weights, dtype=tf.float32)

    def loss_fn(y_true, y_pred, return_components=False):
        y_true_idx = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
        y_true_oh = tf.one_hot(y_true_idx, depth=num_classes)
        y_pred = tf.clip_by_value(
            y_pred, tf.keras.backend.epsilon(), 1.0 - tf.keras.backend.epsilon()
        )

        # А. WEIGHTED FOCAL LOSS
        ce = -y_true_oh * tf.math.log(y_pred)
        focal_mod = tf.pow(1.0 - y_pred, gamma)
        l_focal = (
            tf.reduce_mean(tf.reduce_sum(ce * focal_mod * class_weights, axis=-1))
            * focal_weight
        )

        # Б. EQUITABLE CLASS-WISE DICE LOSS
        intersection = tf.reduce_sum(y_true_oh * y_pred, axis=[1, 2])
        union = tf.reduce_sum(y_true_oh, axis=[1, 2]) + tf.reduce_sum(
            y_pred, axis=[1, 2]
        )
        dice_scores = (2.0 * intersection + 1e-5) / (union + 1e-5)
        l_dice = (1.0 - tf.reduce_mean(dice_scores)) * dice_weight

        # В. ЭНТРОПИЙНЫЙ ШТРАФ
        entropy = -tf.reduce_sum(y_pred * tf.math.log(y_pred), axis=-1)
        l_entropy = tf.reduce_mean(entropy) * entropy_weight

        # Г. ШТРАФ ПО ПЛОЩАДИ
        pred_area_ratio = tf.reduce_mean(y_pred, axis=[1, 2])
        true_area_ratio = tf.reduce_mean(y_true_oh, axis=[1, 2])
        area_diff = tf.maximum(0.0, pred_area_ratio - true_area_ratio)
        l_area = tf.reduce_mean(tf.square(area_diff)) * area_weight

        # Д. TOTAL VARIATION (Гладкость)
        l_tv = tf.reduce_mean(tf.image.total_variation(y_pred)) * tv_weight

        if return_components:
            return {
                "focal": l_focal,
                "dice": l_dice,
                "entropy": l_entropy,
                "area": l_area,
                "tv": l_tv,
            }

        return l_focal + l_dice + l_entropy + l_area + l_tv

    return loss_fn


# ==========================================
# 📊 МЕТРИКИ
# ==========================================
class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self, num_classes, name="mean_iou", **kwargs):
        super().__init__(num_classes=num_classes, name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
        y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
        return super().update_state(y_true, y_pred, sample_weight)


def masked_accuracy(y_true, y_pred):
    y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
    y_pred_classes = tf.cast(tf.argmax(y_pred, axis=-1), tf.int32)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    correct = tf.cast(tf.equal(y_true, y_pred_classes), tf.float32)
    return tf.math.divide_no_nan(tf.reduce_sum(correct * mask), tf.reduce_sum(mask))


# ==========================================
# 🏗️ СБОРКА МОДЕЛИ U-NET (с MobileNetV2)
# ==========================================
class ModelBuilder:
    def __init__(self, config: dict, num_classes: int, logger: logging.Logger):
        self.config = config
        self.num_classes = num_classes
        self.logger = logger
        self.img_size = tuple(config["data"]["image_size"]) + (3,)

    def build(
        self, trainable_encoder=False, encoder_weights="imagenet"
    ) -> tf.keras.Model:
        if encoder_weights in (None, "none", "None", "NONE"):
            encoder_weights = None

        self.logger.info(
            "🚀 Building U-Net with MobileNetV2. "
            f"Encoder Trainable: {trainable_encoder}. "
            f"Encoder weights: {encoder_weights or 'random'}"
        )

        inputs = layers.Input(shape=self.img_size)

        # 1. Спец-препроцессинг для MobileNetV2 (от -1 до 1)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

        # 2. Подключаем энкодер: ImageNet-веса или случайная инициализация
        encoder = tf.keras.applications.MobileNetV2(
            input_tensor=x, weights=encoder_weights, include_top=False, alpha=1.0
        )

        # Устанавливаем статус заморозки (True/False)
        encoder.trainable = trainable_encoder
        encoder_layer_names = {layer.name for layer in encoder.layers}

        # 3. Достаем слои для Skip Connections
        s1 = inputs  # 512x512
        s2 = encoder.get_layer("expanded_conv_project_BN").output  # 256x256
        s3 = encoder.get_layer("block_2_add").output  # 128x128
        s4 = encoder.get_layer("block_5_add").output  # 64x64
        s5 = encoder.get_layer("block_12_add").output  # 32x32
        b = encoder.get_layer("out_relu").output  # 16x16 (Bottleneck)

        # 4. НАШ ДЕКОДЕР
        def upsample_block(x, skip, filters):
            x = layers.UpSampling2D(2)(x)
            x = layers.Concatenate()([x, skip])
            x = layers.Conv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.Conv2D(filters, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)
            return layers.Activation("relu")(x)

        u1 = upsample_block(b, s5, 256)  # 32x32
        u2 = upsample_block(u1, s4, 128)  # 64x64
        u3 = upsample_block(u2, s3, 64)  # 128x128
        u4 = upsample_block(u3, s2, 32)  # 256x256
        u5 = upsample_block(u4, s1, 16)  # 512x512

        # 5. Выходной слой
        outputs = layers.Conv2D(
            self.num_classes, 1, activation="softmax", dtype="float32"
        )(u5)

        model = models.Model(inputs, outputs)
        model.encoder_layer_names = encoder_layer_names

        # --- ⚖️ ДИНАМИЧЕСКИЙ LEARNING RATE ---
        if trainable_encoder and encoder_weights == "imagenet":
            lr = 1e-5  # Медленно, чтобы не стереть память MobileNet
            self.logger.info("🐢 Используется низкий LR (1e-5) для Fine-Tuning")
        elif trainable_encoder:
            lr = self.config["model"].get("learning_rate", 5e-5)
            self.logger.info(f"🌱 Используется LR ({lr}) для обучения энкодера с нуля")
        else:
            lr = 1e-3  # Быстро, чтобы обучить пустой декодер
            self.logger.info("⚡ Используется высокий LR (1e-3) для Warm-Up")

        # --- ⚖️ НАШИ ЗОЛОТЫЕ ВЕСА КЛАССОВ ---
        weights = [1.5] + [1.0] * (self.num_classes - 1)
        for idx in [6, 11]:  # Проверь индексы "маляров"
            if self.num_classes > idx:
                weights[idx] = 0.5
        for idx in [1, 2, 9, 10]:  # Близнецы
            if self.num_classes > idx:
                weights[idx] = 4.0

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=spatial_aware_loss(
                self.num_classes,
                class_weights=weights,
                focal_weight=5.0,
                dice_weight=6.0,
                area_weight=2.0,
                entropy_weight=0.05,
                tv_weight=5e-6,
            ),
            metrics=[masked_accuracy, UpdatedMeanIoU(self.num_classes)],
        )

        return model
