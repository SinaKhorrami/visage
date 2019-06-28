import os
import cv2
from PIL import Image


class FaceDetector(object):
    """
    Face Detector Class
    """

    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.classifier = cv2.CascadeClassifier(os.path.join(os.path.dirname(__file__), self.config['file_name']))
    
    def get_detected_faces(self, image):
        return self.classifier.detectMultiScale(
            image=image,
            scaleFactor=self.config['scale_factor'],
            minNeighbors=self.config['min_neighbors'],
            minSize=(self.config['min_size'][0], self.config['min_size'][1])
        )
    
    @staticmethod
    def get_face_image(frame, dimensions):
        (x, y, w, h) = dimensions
        face_image = frame[y:y+h, x:x+w]

        return Image.fromarray(face_image).convert('L')
