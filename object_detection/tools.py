import os
import numpy as np
import cv2
import glob
import datetime
from .object_detection.utils import label_map_util

class DetectionParams():
    CWD_PATH = os.getcwd()
    # What model
    MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = os.path.join(CWD_PATH,'ownLibraries', 'object_detection','object_detection', MODEL_NAME, 'frozen_inference_graph.pb')

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = os.path.join(CWD_PATH,'ownLibraries', 'object_detection','object_detection', 'data', 'mscoco_label_map.pbtxt')
    NUM_CLASSES = 90

    # Loading label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

class DetectionTools():

    @staticmethod
    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    @staticmethod
    def load_images_paths(folder_to_images):
        paths_to_images = []

        # List directory
        for image in glob.glob("{}*.jpg".format(folder_to_images)):
            paths_to_images.append(image)
        return paths_to_images

    @staticmethod
    def resize_images(path_to_image, new_shape):
        """
        Resize target image to normalized size

        :param path_to_image: paths to target image
        :param new_shape: new shape to trasform the original image
        :return:
        """
        # Save path of resized image in dictionary
        resized_path = '{}_resize.jpg'.format(path_to_image[:path_to_image.rfind('.')])

        # Read the target image and resize it.
        raw_image = cv2.imread(path_to_image)
        image = cv2.resize(raw_image, new_shape)

        # Write this image into disk as aux image for detection
        cv2.imwrite(resized_path, image)

        # Return path to re-sized image as dict
        return resized_path

    @staticmethod
    def todaydate():
        """
        Get actual time in call of method and return it.
        :return: Datetime object
        """
        todaydate = datetime.datetime.now().strftime('%Y-%m-%d')
        return todaydate

    @staticmethod
    def get_centroid(x, y, w, h):

        x1 = int(w / 2)
        y1 = int(h / 2)
        cx = x + x1
        cy = y + y1
        return cx, cy

    @staticmethod
    def delete_image(path_to_image):
        """
        Delete some input file, expected path to image
        :param path_to_image: String with path to image
        :return: True is deletion as successfully, false else
        """
        try:
            os.remove(path_to_image)
            return True
        except Exception as e:
            print('CANT DELETE AUX IMAGE by this error: {}'.format(e))
            return False