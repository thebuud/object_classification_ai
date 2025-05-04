from utils import load_data
from models import resnet_model
from evaluate import evaluate_model
import tensorflow as tf


# pause callback
class PauseAfterEpoch(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        input(
            f"\n Epoch {epoch + 1} complete. Press Enter to begin the next epoch...\n"
        )


# Load dataset from Hugging Face (Tiny ImageNet)
X_train, X_val = load_data()

# Load model
model = resnet_model()

# Train
model.fit(X_train, validation_data=X_val, epochs=10, callbacks=[PauseAfterEpoch()])

# Evaluate
evaluate_model(model, X_val)
print("✅ Training complete!")
