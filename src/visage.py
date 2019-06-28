import cv2

from model.models import FERGModel
from face.detector import FaceDetector
from face.display import WebcamDisplay


class Visage(object):
    """
    Visage Class
    """

    def __init__(self, cfg):
        super().__init__()
        self.model = FERGModel(cfg['model'])
        self.model.load_model_weights()
        self.face_detector = FaceDetector(cfg['face'])
        self.display = WebcamDisplay()

    @staticmethod
    def _draw_rectangle(frame, dimensions):
        (x, y, w, h) = dimensions
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    def _process(self, ret, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.get_detected_faces(gray)

        for face in faces:
            self._draw_rectangle(frame, face)
            face_image = self.face_detector.get_face_image(frame, face)
            predicted_class = self.model.get_face_emotion(face_image)
            
            print(predicted_class)
    
    def run(self):
        self.display.show(self._process)
