import dlib
import cv2

import numpy as np
from enum import Enum
import skimage.transform as transform


class HaarFaceDetector:
    def __init__(self, filename):
        self.haar_face_cascade = cv2.CascadeClassifier(filename)

    def detect_faces(self,
                     image,
                     scaleFactor=1.2,
                     minNeighbors=2,
                     get_top=None):

        # convert the image to gray image as opencv face detector expects gray images
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # let's detect multiscale (some images may be closer to camera than others) images
        face_rects = list(
            self.haar_face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors))

        myList = []
        for (x, y, w, h) in face_rects:
            myList.append(dlib.rectangle(x, y, x + w, y + h))
        face_rects = myList
        face_rects.sort(key=lambda r: r.width() * r.height(), reverse=True)

        if get_top is not None:
            face_rects = face_rects[:get_top]

        return face_rects


class HOGFaceDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    """
    upscale_factor = The upscale argument in detector is the number of times we want to upscale the image.
    The more you upscale, the better are the chances of detecting smaller faces.
    However, upscaling the image will have substantial impact on the computation speed.

    get_top = a number of selected faces
    """

    def detect_faces(self,
                     image,
                     upscale_factor=1,
                     get_top=None):
        face_rects = list(self.detector(image, upscale_factor))

        face_rects.sort(key=lambda r: r.width() * r.height(), reverse=True)

        if get_top is not None:
            face_rects = face_rects[:get_top]

        return face_rects


class FaceAlignMask(Enum):
    INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]  # landmarks
    OUTER_EYES_AND_NOSE = [36, 45, 33]


class FaceAligner:
    def __init__(self,
                 dlib_predictor_path,
                 face_template_path):
        self.predictor = dlib.shape_predictor(dlib_predictor_path)
        self.face_template = np.load(face_template_path)

    def get_landmarks(self,
                      image,
                      face_rect):
        points = self.predictor(image, face_rect)
        return np.array(list(map(lambda p: [p.x, p.y], points.parts())))

    def align_face(self,
                   image,
                   face_rect,
                   dim=96,
                   border=0,
                   mask=FaceAlignMask.INNER_EYES_AND_BOTTOM_LIP):
        mask = np.array(mask.value)

        landmarks = self.get_landmarks(image, face_rect)
        proper_landmarks = border + dim * self.face_template[mask]

        A = np.hstack([landmarks[mask], np.ones((3, 1))]).astype(np.float64)
        B = np.hstack([proper_landmarks, np.ones((3, 1))]).astype(np.float64)
        T = np.linalg.solve(A, B).T

        wrapped = transform.warp(image,
                                 transform.AffineTransform(T).inverse,
                                 output_shape=(dim + 2 * border, dim + 2 * border),
                                 order=3,
                                 mode='constant',
                                 cval=0,
                                 clip=True,
                                 preserve_range=True)

        return wrapped

    def align_faces(self,
                    image,
                    face_rects,
                    *args,
                    **kwargs):
        result = []

        for rect in face_rects:
            result.append(self.align_face(image, rect, *args, **kwargs))

        return result


def clip_to_range(img):
    return img / 255.0


def convert_to_RGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

