from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam


def resnet_model():
    base_model = ResNet50(
        weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3))
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    predictions = Dense(200, activation="softmax")(x)  # Tiny ImageNet has 200 classes
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=Adam(), loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model
