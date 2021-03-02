import os
from pathlib import Path
import shutil
import math
from tensorflow.keras.preprocessing import image_dataset_from_directory
from peewee import JOIN
import numpy as np
import PIL
from birdbox.database import Bird, Image, Crop, query_dict_to_hashable
from birdbox.tools import crop_to_square_crop

TRAINING_DIR = Path(r"images\training")


def equalize_query_count(queries):
    limiting_class = min(queries, key=lambda x: queries[x].count())
    images_per_class = queries[limiting_class].count()
    print(f"Images per class: {images_per_class}\nLimiting class: {limiting_class}")
    return {class_: query.limit(images_per_class) for class_, query in queries.items()}


def build_datasets(image_queries: dict, image_size=(180, 180), batch_size=32, seed=1, validation_split=0.1,
                   label_mode="categorical", crop=False):


    _copy_images(TRAINING_DIR, image_queries, crop=crop)
    kwargs = {"seed": seed, "image_size": image_size, "batch_size": batch_size,
              "validation_split": validation_split, "label_mode": label_mode}
    train_ds = image_dataset_from_directory(TRAINING_DIR, subset="training", **kwargs)
    val_ds = image_dataset_from_directory(TRAINING_DIR, subset="validation", **kwargs)
    train_ds = train_ds.cache()
    val_ds = val_ds.cache()
    return train_ds, val_ds


def _copy_images(folder, image_queries, crop=False):

    # Avoid copying/croping the same dataset again.
    arg_hash = str(hash((query_dict_to_hashable(image_queries), crop)))
    hash_file = folder / "hash.txt"
    is_equal = False
    if hash_file.exists():
        with open(hash_file) as f:
            saved_arg_hash = f.readline().strip()
        is_equal = saved_arg_hash == arg_hash
    if not is_equal:
        with open(hash_file, "w") as f:
            f.write(arg_hash)
    if is_equal:
        return

    _clear_folder()
    for class_name, image_query in image_queries.items():
        class_folder = folder / class_name
        class_folder.mkdir(parents=True)
        for image_entry in image_query:
            destination = class_folder / image_entry.file().name

            if crop:
                crop_entry = Crop.get(Crop.source == image_entry)
                image_np = np.array(image_entry.load())

                x1, y1, x2, y2 = crop_to_square_crop(
                    (crop_entry.x1, crop_entry.y1, crop_entry.x2, crop_entry.y2),
                    (image_entry.width, image_entry.height))

                crop_image_np = image_np[y1:y2, x1:x2]

                PIL.Image.fromarray(crop_image_np).save(destination)

            else:
                shutil.copy(image_entry.file(), destination)


def _clear_folder():
    if TRAINING_DIR.exists():
        for folder in os.listdir(TRAINING_DIR):
            if folder == "hash.txt":
                continue
            for file in os.listdir(TRAINING_DIR / folder):
                Path(TRAINING_DIR / folder / file).unlink()
            os.rmdir(TRAINING_DIR / folder)


def training_bird_query(bird_name, min_size=None, crop=False, limit=None):
    bird_entry = Bird.find(bird_name)
    query = (Image.select()
            .where(
                Image.ignore == 0,
                (Image.is_duplicate == 0) | (Image.is_duplicate.is_null()),
                Image.bird == bird_entry,
                Image.is_natural == 1,
                (Image.has_multiple_birds.is_null()) | (Image.has_multiple_birds == 0))
            )

    if min_size:
        query = query.where(Image.width >= min_size[0], Image.height >= min_size[1])

    if crop:
        query = query.join(Crop, JOIN.LEFT_OUTER).where(
            (Crop.id.is_null(False) & Crop.ignore == 0)
        )
        if min_size:
            query = query.where(Crop.x2 - Crop.x1 >= min_size[0], Crop.y2 - Crop.y1 >= min_size[1])
    if limit:
        query = query.limit(limit)
    return query


def class_ids_from_directory(folder):

    for (dirpath, dirnames, filenames) in os.walk(folder):
        class_names = dirnames
        break
    return {class_name: id_ for id_, class_name in enumerate(class_names)}


def exponential_decay(start=1e-3, end=1e-3, epochs=100):
    if end > start:
        raise ValueError("End value must not be greater than the start value.")

    k = math.log(end/start) / epochs

    def learning_rate(epoch):
        return start * math.exp(k*epoch)

    return learning_rate
