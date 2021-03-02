from PIL import Image as PIL_Image
import matplotlib.pyplot as plt
from IPython.display import clear_output
from abc import ABC
from progressbar import progressbar
from birdbox.database import Image, Bird, Url, SearchHit, SearchTerm


class InvalidInputError(Exception):
    pass


class Session(ABC):

    def query(self):
        return Image.select().where(
            Image.ignore == 0,
            (Image.is_duplicate == 0) | (Image.is_duplicate.is_null()),
            Image.has_bad_quality == 0,
        )

    def process(self, image_entry, user_input):
        if user_input == "i":
            image_entry.ignore = 1
        elif user_input == "d":
            image_entry.is_duplicate = 1
        elif user_input == "q":
            image_entry.has_bad_quality = 1
        else:
            raise InvalidInputError()

    def start(self):
        last_id = None
        for image_entry in progressbar(self.query()):
            print(image_entry)
            image = PIL_Image.open(image_entry.file())
            plt.figure(figsize=(13, 13))
            plt.imshow(image)
            plt.show()
            print(f"{last_id=}")
            last_id = image_entry.id
            user_input = input()
            clear_output()
            if user_input == "":
                print("Aborted.")
                return
            self.process(image_entry, user_input)
            image_entry.save()
        print("End of query.")


class BirdSession(Session):

    def __init__(self, bird_name):
        self.bird_entry = Bird.find(bird_name)

    def query(self):
        super_query = super(BirdSession, self).query()
        return (super_query
                .where(
                    Image.is_bird.is_null(),
                    (Image.is_natural.is_null() | Image.is_natural == 1),
                    Image.bird.is_null())
                .join(Url)
                .join(SearchHit)
                .join(SearchTerm)
                .where(SearchTerm.bird == self.bird_entry)
                .order_by(Image.width.desc())
                )

    def process(self, image_entry, user_input):

        try:
            super(BirdSession, self).process(image_entry, user_input)
        except InvalidInputError:
            pass
        else:
            return

        # Check for other birds
        if len(user_input) > 5:
            bird_entry = Bird.find(user_input)
            image_entry.is_bird = 1
            image_entry.has_multiple_birds = 0
            image_entry.bird = bird_entry
            image_entry.is_natural = 1
            print(f"Labeled as \"{image_entry.name}\".")
            return

        if user_input not in ["1", "u", "b", "n", "m", "q"]:
            raise InvalidInputError()

        image_entry.is_natural = 0 if user_input == "u" else 1
        if user_input == "n":
            image_entry.is_bird = 0
            image_entry.has_multiple_birds = 0
        elif user_input == "m":
            image_entry.is_bird = 1
            image_entry.has_multiple_birds = 1
        elif user_input != "u":
            image_entry.is_bird = 1
            image_entry.has_multiple_birds = 0

        if user_input == "1":
            image_entry.bird = self.bird_entry


class FakeSession(Session):

    def query(self):
        super_query = super(FakeSession, self).query()
        return (super_query
                .where(
                    Image.is_natural == 0,
                    Image.is_drawing.is_null(),
                    Image.is_decoy.is_null()
                ))

    def process(self, image_entry, user_input):

        try:
            super(FakeSession, self).process(image_entry, user_input)
        except InvalidInputError:
            pass
        else:
            return

        if user_input not in ["1", "2", "y"]:
            raise InvalidInputError()
        image_entry.is_drawing = 1 if user_input == "1" else 0
        image_entry.is_decoy = 1 if user_input == "y" else 0
