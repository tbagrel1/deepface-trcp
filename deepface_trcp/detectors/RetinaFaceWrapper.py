#from retinaface import RetinaFace #this is not a must dependency
import cv2
from deepface_trcp.commons.face_data import FaceData

def build_model():
    from retinaface import RetinaFace
    face_detector = RetinaFace.build_model()
    return face_detector

def detect_faces(face_detector, img, crop_margin_ratio):

    from retinaface import RetinaFace

    #---------------------------------

    resp = []

    # The BGR2RGB conversion will be done in the preprocessing step of retinaface.
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #retinaface expects RGB but OpenCV read BGR

    """
    face = None
    img_region = [0, 0, img.shape[1], img.shape[0]] #Really?

    faces = RetinaFace.extract_faces(img_rgb, model = face_detector, align = align)

    if len(faces) > 0:
        face = faces[0][:, :, ::-1]

    return face, img_region
    """

    #--------------------------

    obj = RetinaFace.detect_faces(img, model = face_detector, threshold = 0.9)
    img_width = img.shape[1]; img_height = img.shape[0]
    if type(obj) == dict:
        for key in obj:
            identity = obj[key]
            facial_area = identity["facial_area"]
            y = facial_area[1]
            y2 = facial_area[3]
            x = facial_area[0]
            x2 = facial_area[2]
            w = x2 - x
            h = y2 - y
            confidence = identity["score"]
            crop_margin = round(max(w, h) * crop_margin_ratio)
            x = max(0, x - crop_margin)
            x2 = min(x2 + crop_margin, img_width)
            w = x2 - x
            y = max(0, y - crop_margin)
            y2 = min(y2 + crop_margin, img_height)
            h = y2 - y
            #detected_face = img[int(y):int(y+h), int(x):int(x+w)] #opencv
            detected_face = img[y: y2, x: x2]

            landmarks = identity["landmarks"]
            left_eye = landmarks["left_eye"]
            right_eye = landmarks["right_eye"]
            nose = landmarks["nose"]
            mouth_left = landmarks["mouth_left"]
            mouth_right = landmarks["mouth_right"]

            resp.append(FaceData(
                detected_face, x, y, w, h, confidence, left_eye, right_eye, nose, mouth_left, mouth_right
            ))

    return resp
