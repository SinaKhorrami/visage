import cv2


class WebcamDisplay(object):
    """
    Webcam Display Class
    """
    
    @staticmethod
    def show(process):
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()

            process(ret, frame)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
