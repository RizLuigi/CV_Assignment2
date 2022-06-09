import cv2
import numpy as np

AVAILABLE_DETECTORS = {'SIFT': cv2.SIFT_create,
                       'FAST': cv2.FastFeatureDetector_create,
                       'ORB': cv2.ORB_create,
                       'STAR': cv2.xfeatures2d.StarDetector_create}


class Detector:
    def __init__(self, detector_type, n_points=None, DEBUG=False):
        self.detector_type = detector_type
        if detector_type not in AVAILABLE_DETECTORS.keys():
            raise NotImplementedError('{} detector not implemented yet!'.format(detector_type))
            exit()
        else:
            self.detector = AVAILABLE_DETECTORS[detector_type]()
        self.DEBUG = DEBUG
        return

    def detect(self, img_grey, mask=None):
        img_grey = img_grey.copy()
        if self.DEBUG:
            # Show masked image
            cv2.imshow('Image to detect', cv2.bitwise_and(img_grey, mask))
        # Detection
        kp = self.detector.detect(img_grey, mask=mask)
        # Conversion keypoints -> coordinates
        pts = cv2.KeyPoint_convert(kp)

        if self.DEBUG:
            # Show image with spotted features
            if len(pts) > 0:
                int_kp = pts.astype(int)
            else:
                int_kp = []
            for i, key_point in enumerate(int_kp):
                x, y = key_point.ravel()
                color = np.float64([255, 255, 255])
                cv2.circle(img_grey, (x, y), 3, color)

            text = 'Found {} features'.format(len(pts))
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1
            color = (0, 0, 255)
            thickness = 2
            cv2.imshow('Spotted Features',
                       cv2.putText(img_grey, text, org, font, fontScale, color, thickness, cv2.LINE_AA))
        return pts
