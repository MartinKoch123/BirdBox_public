from pathlib import Path
from datetime import datetime
import shutil
import matplotlib.pyplot as plt
from math import inf
from peewee import Model, SqliteDatabase, CharField, IntegerField, FloatField, IntegrityError, ForeignKeyField
from peewee import DateTimeField
from PIL import Image as PIL_Image
from typing import Tuple
import os
from birdbox.tools import module_dir


DATABASE_FILE = Path(r"..\images\images.db")
DATABASE_FOLDER = Path(r"..\images\database")

database = SqliteDatabase(DATABASE_FILE)





class BaseModel(Model):
    class Meta:
        database = database

    def __str__(self):
        return ", ".join([f"{key}={value:.2f}"
                          if isinstance(value, float)
                          else f"{key}={value}"
                          for key, value in self.__data__.items()])

    @classmethod
    def peak(cls, n=inf, stride=1, start=0):
        peak(cls.select(), n, stride, start)


class Bird(BaseModel):
    name = CharField(unique=True)
    short_name = CharField(unique=True)

    def nickname(self):
        return self.short_name if self.short_name else self.name

    @staticmethod
    def find(partial_name):
        query = Bird.select().where(Bird.name.contains(partial_name))
        if query.count() > 1:
            raise Exception(f"Found multiple birds matching \"{partial_name}\".")
        if query.count() == 0:
            raise Exception(f"Bird matching \"{partial_name}\" not found.")
        return next(iter(query))


class Image(BaseModel):
    name = CharField(unique=True)
    is_duplicate = IntegerField(null=True)
    detection_score = FloatField(null=True)
    detection_score_sum = FloatField(null=True)
    bird = ForeignKeyField(Bird, null=True)
    is_bird = IntegerField(null=True)
    ignore = IntegerField(default=0)
    has_multiple_birds = IntegerField(null=True)
    width = IntegerField(null=True)
    height = IntegerField(null=True)
    has_bad_quality = IntegerField(default=0)

    def file(self):
        return module_dir() / DATABASE_FOLDER / f"{self.name}.jpg"

    def load(self):
        return PIL_Image.open(self.file())

    @classmethod
    def reset_labels(cls, id_):
        if isinstance(id_, str):
            id_ = Image.get(Image.name == id_).id
        cls.update({cls.bird: None, cls.is_bird: None, cls.has_multiple_birds: None}).where(Image.id == id_).execute()
        crop_query = Crop.select().where(Crop.source == id_)
        for crop_entry in crop_query:
            Crop.delete().where(Crop.id == crop_entry.id).execute()


class Url(BaseModel):
    url = CharField(unique=True)
    is_broken = IntegerField(default=0)
    image = ForeignKeyField(Image)


class SearchTerm(BaseModel):
    term = CharField(unique=True)
    bird = ForeignKeyField(Bird)
    source = IntegerField()


class Search(BaseModel):
    search_term = ForeignKeyField(SearchTerm)
    hit_count = IntegerField(default=-1)
    error_count = IntegerField(default=-1)
    time = DateTimeField()


class SearchHit(BaseModel):
    search_term = ForeignKeyField(SearchTerm)
    url = ForeignKeyField(Url)

    def save(self, *args, **kwargs):
        if SearchHit.select().where(SearchHit.url == self.url, SearchHit.search_term == self.search_term).exists():
            raise IntegrityError("Entry already exists.")
        super(SearchHit, self).save(*args, **kwargs)


class Crop(BaseModel):
    source = ForeignKeyField(Image)
    x1 = IntegerField()
    y1 = IntegerField()
    x2 = IntegerField()
    y2 = IntegerField()
    ignore = IntegerField(default=0)


def backup():
    date_string = str(datetime.now()).replace(":", "-").replace(".", "-")
    backup_file = module_dir() / DATABASE_FILE.parent / "backups" / f"{DATABASE_FILE.stem} {date_string}{DATABASE_FILE.suffix}"
    backup_file.parent.mkdir(exist_ok=True)
    shutil.copy(module_dir() / DATABASE_FILE, backup_file)


def peak(query, n=inf, stride=1, start=0):
    for i, row in enumerate(query):
        if i < start or (i - start) % stride != 0:
            continue
        print(row)
        if (i - start + stride) // stride >= n:
            break


def show(id_, figsize: Tuple[float, float] = (13, 13)):
    image_entry = Image.get(Image.id == id_)
    plt.figure(figsize=figsize)
    plt.imshow(image_entry.load())
    plt.axis("off")
    plt.show()


def query_dict_to_hashable(queries):
    return tuple((key, str(queries[key])) for key in sorted(queries))
