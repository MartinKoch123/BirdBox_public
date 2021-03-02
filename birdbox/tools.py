from __future__ import annotations
from csv import reader
from pathlib import Path
import os
import matplotlib.pyplot as plt
from progressbar import progressbar
import time
import warnings
import cv2
import ipywidgets as widgets
from IPython.display import display

from typing import Union, Tuple

Number = Union[int, float]
TwoNumbers = Tuple[Number, Number]
FourNumberTuple = Tuple[Number, Number, Number, Number]


def module_dir():
    return Path(os.path.realpath(__file__)).parent


class RectIterator:
    def __init__(self, rect):
        self.rect = rect

    def __next__(self):
        yield self.rect.left
        yield self.rect.top
        yield self.rect.right
        yield self.rect.bottom
        raise StopIteration()


class Rect:
    def __init__(self, left: Number, top: Number, right: Number, bottom: Number):
        self.left, self.top, self.right, self.bottom = left, top, right, bottom

    def tuple(self) -> FourNumberTuple:
        return self.left, self.top, self.right, self.bottom

    def area(self) -> Number:
        return (self.bottom - self.top) * (self.right - self.left)

    def size(self) -> TwoNumbers:
        return self.width(), self.height()

    def width(self) -> Number:
        return self.right - self.left

    def height(self) -> Number:
        return self.bottom - self.top

    def center(self) -> TwoNumbers:
        return self.horizontal_center(), self.vertical_center()

    def vertical_center(self) -> Number:
        return (self.top + self.bottom) / 2

    def horizontal_center(self) -> Number:
        return (self.left + self.right) / 2

    def left_top_width_height(self) -> FourNumberTuple:
        return self.left, self.top, self.right - self.left, self.bottom - self.top

    def scale_size(self, factor) -> Rect:
        center_x, center_y = self.center()
        width, height = self.width(), self.height()
        return Rect(
            center_x - factor * width / 2,
            center_y - factor * height / 2,
            center_x + factor * width / 2,
            center_y + factor * height / 2
        )

    def int(self):
        return Rect(*map(int, self))

    def __str__(self) -> str:
        return f"Rect(left={self.left}, top={self.top}, right={self.right}, bottom={self.bottom}"

    def __repr__(self) -> str:
        return str(self)

    def __iter__(self):
        return iter(self.tuple())

    def __round__(self, n=None):
        return Rect(*map(lambda x: round(x, n), self))

    def __eq__(self, other):
        return all(c1 == c2 for c1, c2 in zip(self, other))

    def __ne__(self, other):
        return not self == other

    def __mul__(self, other):
        try:
            vertical_factor = other[1]
            horizontal_factor = other[0]
        except TypeError:
            horizontal_factor = other
            vertical_factor = other

        return Rect(
            self.left * horizontal_factor,
            self.top * vertical_factor,
            self.right * horizontal_factor,
            self.bottom * vertical_factor
        )

    def __truediv__(self, other):
        try:
            vertical_factor = other[1]
            horizontal_factor = other[0]
        except TypeError:
            horizontal_factor = other
            vertical_factor = other

        return Rect(
            self.left / horizontal_factor,
            self.top / vertical_factor,
            self.right / horizontal_factor,
            self.bottom / vertical_factor
        )

    @staticmethod
    def intersection(r1: Rect, r2: Rect) -> Number:
        top = max(r1.top, r2.top)
        left = max(r1.left, r2.left)
        bottom = min(r1.bottom, r2.bottom)
        right = min(r1.right, r2.right)
        height = max(bottom - top, 0)
        width = max(right - left, 0)
        return height * width

    @staticmethod
    def union(r1: Rect, r2: Rect) -> Number:
        return r1.area() + r2.area() - Rect.intersection(r1, r2)

    @staticmethod
    def intersection_over_union(r1: Rect, r2: Rect) -> float:
        return Rect.intersection(r1, r2) / Rect.union(r1, r2)

    @classmethod
    def from_left_top_width_height(cls, left: Number, top: Number, width: Number, height: Number) -> Rect:
        return cls(left, top, left + width, top + height)


def read_csv(file):
    with open(file, 'r', encoding='utf-8') as read_obj:
        csv_reader = reader(read_obj)
        list_of_rows = list(csv_reader)
    return list_of_rows


def image_identifiers_from_directory(folder):
    return (Path(file).stem for file in os.listdir(folder))


class StopFile:
    def __init__(self, file="stop.txt", iterable=None, check_after=1):
        self.file = file
        self.check_after = check_after
        self.check_counter = 0
        self.iterable = iterable
        open(file, "w").close()

    def stop_is_pending(self):
        self.check_counter += 1
        if self.check_counter < self.check_after:
            return False
        else:
            self.check_counter = 0
        try:
            with open(self.file) as f:
                content = f.read()
        except (PermissionError, FileNotFoundError):
            return False
        stop = content.strip().lower()
        if stop == "stop":
            open(self.file, "w").close()
        return stop

    def abort_if_stop_is_pending(self):
        if self.stop_is_pending():
            raise Exception("User abort")

    def __iter__(self):
        self.iterator = iter(self.iterable)
        return self

    def __next__(self):
        self.abort_if_stop_is_pending()
        return next(self.iterator)


def fancy_loop(iterable):
    return StopFile(iterable=progressbar(iterable))


def show_images(query, n=1, columns=4, offset=0):
    plt.figure(figsize=(20, 16))
    for i, image_entry in enumerate(query.offset(offset).limit(n)):
        plt.subplot((n // columns) + 1, columns, i+1)
        plt.title(f"{image_entry.id}")
        plt.imshow(image_entry.load())
        plt.axis("off")
    plt.show()


def insane(errors=(Exception,), wait_periods=()):
    def inner_function(f):
        def innermost_function(*args, **kwargs):
            for wait_period in wait_periods:
                try:
                    return f(*args, **kwargs)
                except errors as e:
                    pass
                time.sleep(wait_period)
            return f(*args, **kwargs)
        return innermost_function
    return inner_function


def crop_to_square_crop(crop_box: Rect, image_size: TwoNumbers) -> Rect:
    full_width, full_height = image_size
    center_x, center_y = crop_box.center()

    size = min(max(crop_box.size()), full_width, full_height)
    half_size = size / 2

    if full_width - center_x < half_size:
        center_x = full_width - half_size
    elif center_x < half_size:
        center_x = half_size
    if full_height - center_y < half_size:
        center_y = full_height - half_size
    elif center_y < half_size:
        center_y = half_size

    x1 = center_x - half_size
    y1 = center_y - half_size
    return Rect.from_left_top_width_height(x1, y1, size, size).int()


def process_video(file: Path, processing_function, skip: int = 0, start: int = 0,
                  end: int = None, output_path: Union[Path, str] = None):

    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(str(file))
    output_path.parent.mkdir(exist_ok=True)

    cv2.namedWindow(file.stem, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(file.stem, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.waitKey(100)
    cv2.setWindowProperty(file.stem, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cap = cv2.VideoCapture(str(file))

    out = None
    if output_path:
        frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
        out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    last_time, current_time = 0, 0
    counter = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            counter += 1
            if counter % (skip+1) != 0:
                continue
            if counter < start or (end is not None and counter > end):
                continue

            processing_function(frame)

            current_time = time.time()
            fps = 1 / (current_time - last_time)
            last_time = current_time

            cv2_putText(frame, f"FPS: {fps:.0f}", (10, 20))
            cv2.imshow(file.stem, frame)

            if out:
                out.write(frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()


def detection_box_to_crop_box(relative_box, image_size, margin=0.1):
    warnings.warn("Deprecated", DeprecationWarning, stacklevel=2)
    det_x1, det_y1, det_x2, det_y2 = relative_box
    width = image_size[0] * (det_x2 - det_x1)
    height = image_size[1] * (det_y2 - det_y1)

    return Rect(
        max([int(det_x1 * image_size[0] - margin * width), 0]),
        max([int(det_y1 * image_size[1] - margin * height), 0]),
        min([int(det_x2 * image_size[0] + margin * width), image_size[0] - 1]),
        min([int(det_y2 * image_size[1] + margin * height), image_size[1] - 1]),
    )


def crop_box_to_square_crop_box(box, image_size):
    warnings.warn("Deprecated, use 'crop_to_square_crop' instead.", DeprecationWarning)
    full_width, full_height = image_size
    x1, y1, x2, y2 = box
    center_x, center_y = (x2 + x1) / 2, (y2 + y1) / 2

    size = min(max([x2 - x1, y2 - y1]), full_width, full_height)

    if full_width - center_x < size / 2:
        center_x = full_width - size / 2
    elif center_x < size / 2:
        center_x = size / 2
    if full_height - center_y < size / 2:
        center_y = full_height - size / 2
    elif center_y < size / 2:
        center_y = size / 2

    x1, x2 = int(center_x - size / 2), int(center_x + size / 2)
    y1, y2 = int(center_y - size / 2), int(center_y + size / 2)

    return Rect(x1, y1, x2, y2)


def cv2_putText(image, text, org, font=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA):
    cv2.putText(image, text, org, font, fontScale, color, thickness, lineType)


def video_drop_down():
    return widgets.Dropdown(
        options=['Blue_tit_vs_chaffinch.mp4', 'Verl_red_robin_in_water.mp4', "Ground_feeding.mp4", "Ground_feeding2.mp4"],
        value='Blue_tit_vs_chaffinch.mp4',
        description='Video file:',
    )
