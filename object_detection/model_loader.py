from .model_object import ObjectDetection
from .tools import DetectionTools
from PIL import Image
import time


class DNNModel():
    def __init__(self):
        """
        Instantiates the mobilenets model
        """

        self._dnn_model = ObjectDetection()
        self.index_to_string = {3: 'car', 6: 'bus', 8: 'truck', 1: 'person'}
        print('DNNMODEL started')

    def get_predictions_from_array(self, path_to_image=None):
        """
        :param path_to_image: Path to image in disk
        :return: JSON with prediction results for input image.
        """
        response = {'results':
            [
                {'prediction': []
                 }
            ]
        }
        if path_to_image is not None:
            image = Image.open(path_to_image)
            image_array = DetectionTools.load_image_into_numpy_array(image)

            t1 = time.time()
            (boxes, scores, classes, num_detections) = self._dnn_model.predict(image_array)
            t2 = time.time()
            print('Time of prediction', t2-t1)
            for i, b in enumerate(boxes[0]):
                #        person  1       car    3                bus   6               truck   8
                if classes[0][i] == 3 or classes[0][i] == 6 or classes[0][i] == 8:
                    if scores[0][i] >= 0.4:
                        x0 = int(boxes[0][i][3] * image_array.shape[1])
                        y0 = int(boxes[0][i][2] * image_array.shape[0])

                        x1 = int(boxes[0][i][1] * image_array.shape[1])
                        y1 = int(boxes[0][i][0] * image_array.shape[0])

                        response['results'][0]['prediction'].append({
                            'coord': {
                                    'xmin': x0, 'ymin': y0,
                                    'xmax': x1, 'ymax': y1
                            },
                            'class': self.index_to_string[classes[0][i]],
                            'prob': scores[0][i]
                        })

            return response
        else:
            return response