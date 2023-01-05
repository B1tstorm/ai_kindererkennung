import logging
import pathlib

import tensorflow as tf
import numpy as np
from object_detection.utils import label_map_util
from PIL import Image

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_size, get_single_tag_keys
from label_studio.core.settings.base import DATA_UNDEFINED_NAME

logger = logging.getLogger(__name__)


class TFMobileNet(LabelStudioMLBase):

    def __init__(self, image_dir=None, score_threshold=0.3, **kwargs):
        """
        :param image_dir: Directory where images are stored
        :param score_threshold: score threshold to wipe out noisy results
        :param kwargs:
        """
        super(TFMobileNet, self).__init__(**kwargs)

        self.image_dir = image_dir
        # This is the hostname/IP from the other Docker container
        self.hostname = "http://172.16.238.10:8080"

        # Load the exported model from saved_model directory
        PATH_TO_SAVED_MODEL = '/saved_models/centernet_hg104_1024x1024_coco17_tpu-32'
        print("Loading model...")
        # Lead saved model and build detection fuction
        self.detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
        print('Model loading done....')

        # Load label map data
        def download_labels(filename):
            base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
            label_dir = tf.keras.utils.get_file(fname=filename,
                                                origin=base_url + filename,
                                                untar=False)
            label_dir = pathlib.Path(label_dir)
            print("######### " + str(label_dir))

            return str(label_dir)

        LABEL_FILENAME = 'mscoco_label_map.pbtxt'
        PATH_TO_LABELS = download_labels(LABEL_FILENAME)

        # TODO: Testen PATH_TO_LABELS festen Pfad geben /root/.keras/...
        self.category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

        self.from_name, self.to_name, self.value, self.labels_in_config = get_single_tag_keys(
            self.parsed_label_config, 'RectangleLabels', 'Image')

        # Create a list of labels from the label_config. We use category_index instead.
        # schema = list(self.parsed_label_config.values())[0]
        # self.labels_in_config = set(self.labels_in_config)

        self.score_thresh = score_threshold

    def _get_image_url(self, task):  # 'data' stores the fullpath of the img
        image_url = task['data'].get(self.value) or task['data'].get(DATA_UNDEFINED_NAME)

        return image_url

    def load_image_into_numpy_array(self, path):
        return np.array(Image.open(path))

    def predict(self, tasks, **kwargs):
        assert len(tasks) == 1
        task = tasks[0]  # looks like task (img) is dealt one by one
        image_url = self._get_image_url(task)
        print('image_url:', image_url)
        image_path = self.get_local_path(image_url, project_dir=self.image_dir)
        print('image_path:', image_path)

        image_np = self.load_image_into_numpy_array(image_path)

        if len(image_np.shape) == 2:
            image_np = np.stack((image_np,) * 3, axis=-1)

        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]
        detections = self.detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # Only interested in the first num_detections.

        boxes = detections['detection_boxes'][0].numpy()  # Getting the list of box coordinates
        max_boxes_to_draw = boxes.shape[0]  # Getting the number of the list
        scores = detections['detection_scores'][0].numpy()  # Getting scores for threshold evaluation

        results = []
        all_scores = []

        img_width, img_height = get_image_size(image_path)  # This is not used either

        for i in range(boxes.shape[0]):
            if scores is None or scores[i] > self.score_thresh:
                class_name = self.category_index[detections['detection_classes'][0, i].numpy().astype(int)]['name']
                results.append({
                    "from_name": self.from_name,
                    "to_name": self.to_name,
                    "type": "rectanglelabels",
                    "value": {
                        "rectanglelabels": [class_name],
                        "x": boxes[i, 1] * 100,
                        "y": boxes[i, 0] * 100,
                        "width": (boxes[0, 3] - boxes[0, 1]) * 100,
                        "height": (boxes[0, 2] - boxes[0, 0]) * 100
                    }
                })
                all_scores.append(scores[i])
        avg_score = sum(all_scores) / max(len(all_scores), 1)
        print("results", results)
        print("each score:", all_scores)
        return [{
            "result": results,
            "score": avg_score
        }]
