import pickle
from typing import Tuple, List
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from IPython.display import clear_output
from birdbox.augmentation import RandomBrightness, RandomSaturation
from birdbox.database import Bird
from birdbox.training import build_datasets, equalize_query_count, training_bird_query
from birdbox.tools import module_dir


def keras_example_classifier(image_size: Tuple[float, float], class_count: int):
    input_shape = image_size + (3,)

    data_augmentation = keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ])

    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if class_count == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = class_count

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)


COLOR_ADDITION = 5


def default_augmentation(seed=None):
    return keras.models.Sequential([
            preprocessing.RandomRotation(factor=0.03, seed=seed),
            preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1, seed=seed),
            preprocessing.RandomFlip(mode="horizontal", seed=seed),
            preprocessing.RandomContrast(factor=0.1, seed=seed),
            RandomSaturation(lower=0.8, upper=1.2, seed=seed),
            RandomBrightness(50, seed=seed),
        ], name="augmentation")


def efficient_net(net_id, image_size, class_count, augmentation_seed=None):
    kwargs = {"weights": None, "classes": class_count, "input_shape": image_size + (3,)}
    nets = {
        0: tf.keras.applications.EfficientNetB0,
        1: tf.keras.applications.EfficientNetB1,
        2: tf.keras.applications.EfficientNetB1,
        3: tf.keras.applications.EfficientNetB3,
        4: tf.keras.applications.EfficientNetB4,
        5: tf.keras.applications.EfficientNetB5,
        6: tf.keras.applications.EfficientNetB6,
        7: tf.keras.applications.EfficientNetB7,
        }
    base_net = nets[net_id](**kwargs)

    inputs = layers.Input(shape=image_size + (3,))
    augmentation = default_augmentation(seed=augmentation_seed)
    x = augmentation(inputs)
    outputs = base_net(x)
    return tf.keras.Model(inputs, outputs)


class CustomCallback(keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        learning_rate = self.model.optimizer.learning_rate.numpy()

        if "learning_rate" in self.model.history.history:
            self.model.history.history["learning_rate"] += [learning_rate]
        else:
            self.model.history.history["learning_rate"] = [learning_rate]


class LearningRate(tf.keras.callbacks.Callback):

    def __init__(self, learning_rate_function):
        super().__init__()
        self.learning_rate_function = learning_rate_function

    def on_train_begin(self, logs={}):
        self.model.history.history["learning_rate"] = []

    def on_epoch_end(self, batch, logs={}):
        self.model.history.history["learning_rate"].append(self.learning_rate_function(len(self.model.history.history["learning_rate"]) + 1))


class Classifier:

    TRAINING_DIR = Path(r"images\training")

    def __init__(self, birds: List[str], image_size: Tuple[int] = (180, 180), net_class: int = 0,
                 augmentation_seed: int = 0):

        self.image_size = image_size
        self.net_class = net_class
        self.augmentation_seed = augmentation_seed
        self.birds = [Bird.find(bird_term).nickname() for bird_term in birds]
        self.category_index = {i+1: {"id": i+1, "name": bird_name} for i, bird_name in enumerate(sorted(self.birds))}
        bird_ids = sorted([Bird.find(bird_name).id for bird_name in self.birds])
        model_suffix = "-".join(str(id_) for id_ in bird_ids)
        self.name = f"EfficientNetB{self.net_class}_{self.image_size[0]}x{self.image_size[1]}_{model_suffix}"

        self.train_ds = None
        self.val_ds = None
        self.queries = None
        self.model = None
        self.history = None

    def build_model(self):
        self.model = efficient_net(self.net_class, image_size=self.image_size, class_count=len(self.birds),
                                   augmentation_seed=self.augmentation_seed)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                           loss="categorical_crossentropy", metrics=["accuracy"])
        self.history = None

    def prepare_training_data(self, batch_size: int = 32, validation_split: float = 0.2, limit=None):
        queries = {bird: training_bird_query(bird, min_size=self.image_size, crop=True, limit=limit)
                   for bird in self.birds}
        self.queries = equalize_query_count(queries)
        self.train_ds, self.val_ds = build_datasets(self.queries, image_size=self.image_size,
                                        validation_split=validation_split, batch_size=batch_size, crop=True)

    def train(self, epochs=1, learning_rate_function=lambda epoch: 1e-3):
        if not self.train_ds:
            raise Exception("Prepare training data first.")

        lrate = tf.keras.callbacks.LearningRateScheduler(learning_rate_function)

        history = self.model.fit(self.train_ds, epochs=epochs, validation_data=self.val_ds, callbacks=[lrate, LearningRate(learning_rate_function)])
        if self.history:
            for key in self.history:
                self.history[key] += history.history[key]
        else:
            self.history = history.history

    def save(self):
        self.model.save(self.path())
        with open(self._history_path(), 'wb') as file:
            pickle.dump(self.history, file)

    def path(self):
        return module_dir() / "../models" / self.name

    def _history_path(self):
        return self.path() / "history.pi"

    def validation_accuracy(self):
        return self.history["val_accuracy"][-1]

    @classmethod
    def load(cls, name: str):

        parts = name.split("_")
        net_class = int(parts[0][-1])
        image_size = tuple([int(size_string) for size_string in parts[1].split("x")])
        bird_ids = [int(id_string) for id_string in parts[2].split("-")]
        birds = [Bird.get(Bird.id == id_).name for id_ in bird_ids]

        obj = cls(birds, image_size=image_size, net_class=net_class)

        obj.model = tf.keras.models.load_model(obj.path())

        with open(obj._history_path(), "rb") as file:
            obj.history = pickle.load(file)

        clear_output()
        return obj




