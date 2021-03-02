from birdbox.classifiers import Classifier
from birdbox.database import Bird


class Analysis:

    def __init__(self, birds, image_size=(120, 120), batch_size=32, net_class=0, validation_split=0.2, epochs=1,
                 learning_rate_function=lambda x: 1e-3, class_limit=None):
        self.birds = [Bird.find(bird).name for bird in birds]
        self.image_size = image_size
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.epochs = epochs
        self.learning_rate_function = learning_rate_function
        self.class_limit = class_limit
        self.model = Classifier(self.birds, image_size=image_size, net_class=net_class)

    def run(self):
        self.model.build_model()
        self.model.prepare_training_data(validation_split=self.validation_split, batch_size=self.batch_size,
                                         limit=self.class_limit)
        self.model.train(epochs=self.epochs, learning_rate_function=self.learning_rate_function)
        return {"accuracy": self.model.history["accuracy"][-1], "val_accuracy": self.model.history["val_accuracy"][-1]}


