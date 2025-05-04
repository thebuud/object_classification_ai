from datasets import load_dataset as hf_load_dataset
import tensorflow as tf
import numpy as np

IMG_SIZE = (224, 224)


def load_data():
    dataset = hf_load_dataset("zh-plus/tiny-imagenet", streaming=True)
    train_data = dataset["train"]
    val_data = dataset["valid"]

    def preprocess(example):
        # Convert image to RGB and resize (on-the-fly)
        img = example["image"].convert("RGB").resize(IMG_SIZE)
        img = tf.convert_to_tensor(np.array(img), dtype=tf.float32) / 255.0
        label = tf.convert_to_tensor(example["label"])
        return img, label

    def to_tf_dataset(data, augment=False):
        def generator():
            for example in data:
                try:
                    yield preprocess(example)
                except Exception as e:
                    print(f"Skipped: {e}")

        ds = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tf.TensorSpec(shape=(224, 224, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int64),
            ),
        )

        if augment:
            data_augmentation = tf.keras.Sequential(
                [
                    tf.keras.layers.RandomFlip("horizontal"),
                    tf.keras.layers.RandomRotation(0.1),
                    tf.keras.layers.RandomZoom(0.1),
                ]
            )
            ds = ds.map(
                lambda x, y: (data_augmentation(x), y),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        return ds.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

    train_ds = to_tf_dataset(train_data, augment=True)
    val_ds = to_tf_dataset(val_data, augment=False)

    return train_ds, val_ds
