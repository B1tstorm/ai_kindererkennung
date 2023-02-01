#%%
#%matplotlib inline
import time
import warnings
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
import os
import pathlib
import tensorflow as tf

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging (1)
tf.get_logger().setLevel('ERROR')  # Suppress TensorFlow logging (2)
warnings.filterwarnings('ignore')  # Suppress Matplotlib warnings

def get_Path_TO_ALL_IMAGES():
    curDir = os.getcwd()
    os.chdir("images/test_images")
    path = os.getcwd()
    os.chdir(curDir)
    return path


def get_Path_TO_DETECTED_PERSONS():
    curDir = os.getcwd()
    os.chdir("images/detected_persons")
    path = os.getcwd()
    os.chdir(curDir)
    return path

def load_image_into_numpy_array(path):
    return np.array(Image.open(path))

def get_Path_TO_Not_Relevant_Label():
    curDir = os.getcwd()
    os.chdir("images/not_relevant_label")
    path = os.getcwd()
    os.chdir(curDir)
    return path


def get_PATH_TO_MODEL_DIR():
    curDir = os.getcwd()
    os.chdir("TensorFlow/workspace/model1/exported-models/30er/saved_model")
    path = os.getcwd()
    os.chdir(curDir)
    return path

# Download labels file
def download_labels(filename):
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
    label_dir = tf.keras.utils.get_file(fname=filename,
                                        origin=base_url + filename,
                                        untar=False)
    label_dir = pathlib.Path(label_dir)
    return str(label_dir)

# TODO
def get_labels(filename):
    path = ''

LABEL_FILENAME = 'mscoco_label_map.pbtxt'
PATH_TO_LABELS = "TensorFlow/workspace/model1/annotations/label_map.pbtxt"
PATH_TO_ALL_IMAGES = get_Path_TO_ALL_IMAGES()
PATH_TO_MODEL_DIR = get_PATH_TO_MODEL_DIR()



def get_Image_Paths_Array(path):
    """
    @path ist ein pfad zu einem Ordner mit unterordner die Bilder Inhalten
    :returns ein Array mit den pfaden zu jedem Bild in den Unterordner
    """
    array = []
    for image in os.listdir(path):
        fullFileName = os.path.join(path, image)
        array.append(fullFileName)
    return array

IMAGE_PATHS = get_Image_Paths_Array(PATH_TO_ALL_IMAGES)


# Es ist notwendig das matplotlib Backend nach dem viz_utils import wieder umzustellen damit die inline plot funktion funktioniert, da in visualization_utils das matplotlib Backend überschrieben wird

#matplotlib.use("module://matplotlib_inline.backend_inline")
#print(matplotlib.get_backend())

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR
print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
model = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))


category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

 #%%









start_time = time.time()
#% matplotlib inline
import matplotlib

#matplotlib.use("module://matplotlib_inline.backend_inline")
matplotlib.use("agg")


from PIL import Image

import matplotlib.pyplot as plt

id = 1
#detected_Images = np.unique(detected_Images)
for image_path in IMAGE_PATHS:
    print('Running inference for {}... '.format(image_path), end='')

    if '.DS_Store' in image_path:
        continue

    image_np = load_image_into_numpy_array(image_path)
    if len(image_np.shape) == 2:
        image_np = np.stack((image_np,) * 3, axis=-1)

    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    time1 = time.time()
    detections = model(input_tensor) # create detection

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    time2 = time.time()
    print("took: ", time2 - time1)
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],

        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.3,
        agnostic_mode=False)

    plt.figure()
    plt.imshow(image_np_with_detections)
    plt.show()
    matplotlib.pyplot.savefig('images/plots/' + str(id))
    id = id+1
    if id == 40:
        break
    #print('Done')

end_time = time.time()
print("###################################################################################################")
print("completed in", end_time - start_time)
# sphinx_gallery_thumbnail_number = 2

#%%
