import os
import tensorflow as tf
import keras
import config

IMG_SIZE = (96, 96)
NUM_CLASSES = 2
INPUT_SHAPE = IMG_SIZE + (3,)


def create_dataset(folder_path, is_training=True):
    """
    Usa a estrutura de pastas (0 e 1) para carregar imagens e labels automaticamente.
    """
    if not os.path.exists(folder_path):
        raise ValueError(f"Pasta não encontrada: {folder_path}")

    print(f"Carregando dataset de: {folder_path}")

    ds = keras.utils.image_dataset_from_directory(
        folder_path,
        labels="inferred",
        label_mode="binary",
        class_names=["0", "1"],
        color_mode="rgb",
        batch_size=config.TRAIN_CONFIG["batch"],
        image_size=IMG_SIZE,
        shuffle=is_training,
        seed=123,
    )

    ds = ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def create_model():
    inputs = keras.Input(shape=INPUT_SHAPE)

    x = keras.layers.Rescaling(1.0 / 127.5, offset=-1)(inputs)

    base_model = keras.applications.MobileNetV2(
        input_shape=INPUT_SHAPE, include_top=False, weights="imagenet", alpha=0.35
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(config.TRAIN_CONFIG["dropout_rate"])(x)

    outputs = keras.layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)

    focal_loss = keras.losses.BinaryFocalCrossentropy(
        alpha=0.25, gamma=2, from_logits=False
    )
    model.compile(
        optimizer="adam", loss=focal_loss, metrics=["accuracy", "precision", "recall"]
    )
    return model


def convert_to_tflite(model, dataset_generator, filename, quantize_int8):
    print(f"Convertendo para {filename}")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize_int8:
        print("Quantização INT8")

        def representative_data_gen():
            for images, _ in dataset_generator.take(50):
                for i in range(images.shape[0]):
                    yield [tf.expand_dims(images[i], axis=0)]

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    with open(filename, "wb") as f:
        f.write(tflite_model)
    print(f"Modelo salvo com sucesso: {filename}")


def run():
    # Setup GPU
    try:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"--> GPU ATIVA: {len(gpus)}")
    except Exception as e:
        print(f"Erro GPU: {e}")

    train_ds = create_dataset(config.TRAIN_PATH, is_training=True)
    test_ds = create_dataset(config.TEST_PATH, is_training=False)
    val_ds = create_dataset(config.VAL_PATH, is_training=False)

    model = create_model()
    model.summary()
    model.fit(train_ds, validation_data=val_ds, epochs=config.TRAIN_CONFIG["epochs"])
    results = model.evaluate(test_ds, return_dict=True)

    print("=" * 30)
    print(f"Loss (Focal): {results['loss']:.4f}")
    print(f"Accuracy:     {results['accuracy']:.2%}")
    print(f"Precision:    {results['precision']:.2%}")
    print(f"Recall:       {results['recall']:.2%}")

    p = results["precision"]
    r = results["recall"]
    if (p + r) > 0:
        f1 = 2 * (p * r) / (p + r)
        print(f"F1-Score:     {f1:.2%}")
    print("=" * 30 + "\n")

    print("--- Exportando TFLite ---")
    convert_to_tflite(model, train_ds, "model_fire_a35_int8.tflite", quantize_int8=True)

    with open("labels.txt", "w") as f:
        f.write("0\n1\n")

    print("Arquivo gerado quantizado")


if __name__ == "__main__":
    run()
