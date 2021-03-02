import os
import warnings
import time
from collections import namedtuple
from pathlib import Path
from typing import Iterable
import numpy as np
import tensorflow as tf
from PIL import Image as PIL_Image
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils
from birdbox.tools import Rect


BIRD_CLASS_INDEX = 16
RELATIVE_MARGIN = 0.1

Detection = namedtuple("Detection", ["score", "box", "class_"])


def load_detector(model_name="centernet_hg104_1024x1024_coco17_tpu-32"):

    warnings.filterwarnings('ignore')           # Suppress Matplotlib warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)

    ALL_MODELS_DIR = Path(r"C:\Users\marti\.keras\datasets")
    LABELS_FILE = ALL_MODELS_DIR / "mscoco_label_map.pbtxt"

    model_dir = ALL_MODELS_DIR / model_name
    save_model_dir = model_dir / "saved_model"

    tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

    # Enable GPU dynamic memory allocation
    gpus = tf.config.experimental.list_physical_devices('GPU')
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        if not str(e).startswith("Physical devices cannot be modified"):
            raise e

    print('Loading model...', end='')
    start_time = time.time()

    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(str(save_model_dir))

    print(f"Done! Took {time.time() - start_time}s")
    category_index = label_map_util.create_category_index_from_labelmap(str(LABELS_FILE), use_display_name=True)
    return detect_fn, category_index


def detect_and_classify_birds(image_array, detection_function, classifier, classifier_image_size, **kwargs):

    detections = detect_birds(image_array, detection_function, **kwargs)

    image_pil = PIL_Image.fromarray(image_array)
    classifications = []
    for detection in detections:

        box = detection.box * image_pil.size  # To pixels
        box = round(box.scale_size(1 + RELATIVE_MARGIN))
        bird_image = image_pil.crop(box)
        image_tf = tf.expand_dims(np.array(bird_image.resize(classifier_image_size)), 0)
        prediction = classifier.predict(image_tf)
        class_id_prediction = prediction[0].argmax()
        classifications.append(Detection(1, detection.box, class_id_prediction + 1))

    return classifications


def detect_birds(image_array, detection_function, min_score=0.2, non_max_suppression=True, max_intersection_over_size=0.7):

    input_tensor = tf.convert_to_tensor(image_array)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detection_function(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    boxes = [Rect(left, top, right, bottom) for top, left, bottom, right in detections["detection_boxes"]]

    detections = [Detection(score, box, int(class_)) for score, box, class_
                  in zip(detections["detection_scores"], boxes, detections["detection_classes"])
                  if class_ == BIRD_CLASS_INDEX and score > min_score]

    if non_max_suppression:
        detections = suppress_non_max(detections)
    if max_intersection_over_size is not None:
        detections = suppress_intersections(detections, max_intersection_over_size=max_intersection_over_size)

    return detections


def suppress_non_max(detections: Iterable[Detection]):
    result = []
    for detection in detections:
        for filtered_detection in result:
            if Rect.intersection_over_union(detection.box, filtered_detection.box) > 0.4:
                break
        else:
            result.append(detection)
    return result


def suppress_intersections(detections: Iterable[Detection], max_intersection_over_size: float = 0.7):
    result = []
    for detection in detections:
        for filtered_detection in result:
            intersection = Rect.intersection(detection.box, filtered_detection.box)
            min_size = min(detection.box.area(), filtered_detection.box.area())
            if intersection / min_size > max_intersection_over_size:
                break
        else:
            result.append(detection)
    return result


def visualize_detection_boxes(image_array, detections, category_index, **kwargs):
    visualization_utils.visualize_boxes_and_labels_on_image_array(
        image_array,
        np.array([(d.box.top, d.box.left, d.box.bottom, d.box.right) for d in detections]),
        [d.class_ for d in detections],
        [d.score for d in detections],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0,
        agnostic_mode=False,
        **kwargs
    )
    return