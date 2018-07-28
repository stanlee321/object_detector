from PIL import Image, ImageDraw
import cv2
import random
from .model_loader import DNNModel
from .tools import  DetectionTools
from .loggs import DetectionLogs as LOGGER


class CarDetector():
    # Taking in account the 5 mpx resolution 2560 x 1920
    def __init__(self):
        """
        Detect, ROI, pass der routes as dict of images into a target folder/
        main: get_objects(folder/)

        """
        self.model = DNNModel()

    def get_coord(self, path_to_image):

        """
        :param: Path to target image
        :return: dict of route to detected images in folder/
        """
        image_paths_and_detection = {}

        # get new images paths

        # Path to the new created detected image.
        path_to_detected_image = '{}_detected.jpg'.format(path_to_image[:path_to_image.rfind('.')])

        # Grab the original and  resized path
        image_paths_and_detection['original_image'] = path_to_image
        image_paths_and_detection['detections'] = []

        # Call the DL model
        response = self.model.get_predictions_from_array(path_to_image)

        # LOGGER.info('car_detector mobilenets response', response)

        _img_res = Image.open(path_to_image)
        draw = ImageDraw.Draw(_img_res, mode="RGBA")

        prediction = response["results"][0]["prediction"]

        for index, pred in enumerate(prediction):
            coord = pred['coord']
            x, y = coord["xmin"], coord["ymin"]
            w, h = coord["xmax"], coord["ymax"]

            AREA = (w - x) * (h - y)

            centroid = DetectionTools.get_centroid(w, h, x, y)

            # Draw fill rectangle around the detected object
            draw.rectangle((x, y, w, h),
                           fill=(random.randint(1, 255),
                                 random.randint(1, 255),
                                 random.randint(1, 255),
                                 127))
            detection = {
                'coord': (x, y, w, h),
                'area': AREA,
                'centroid': centroid,
                'class': pred['class'],
                'prob': pred['prob']
            }

            image_paths_and_detection['detections'].append(detection)

        # Write the image with detections into disk
        _img_res.save(path_to_detected_image)
        return image_paths_and_detection


    def get_objects(self, target_image = '.jpg'):
        """
        Obtain the coord of the detected objects in target-image
        :param target_image: A *.jpg image
        :return: dict with up-scaled coordinates.
        """
        # Get the image paths and detections
        image_paths_and_detection = self.get_coord(target_image)

        return image_paths_and_detection


if __name__ == '__main__':
    pass