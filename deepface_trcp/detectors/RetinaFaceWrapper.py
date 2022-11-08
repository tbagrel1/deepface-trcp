#from retinaface import RetinaFace #this is not a must dependency
import cv2
from deepface_trcp.commons.face_data import FaceData

def build_model():
    from retinaface import RetinaFace
    face_detector = RetinaFace.build_model()
    return face_detector

def detect_faces(face_detector, img):

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

    if type(obj) == dict:
        for key in obj:
            identity = obj[key]
            facial_area = identity["facial_area"]

            y = facial_area[1]
            h = facial_area[3] - y
            x = facial_area[0]
            w = facial_area[2] - x
            confidence = identity["score"]

            #detected_face = img[int(y):int(y+h), int(x):int(x+w)] #opencv
            detected_face = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]

            landmarks = identity["landmarks"]
            left_eye = landmarks["left_eye"]
            right_eye = landmarks["right_eye"]
            nose = landmarks["nose"]
            mouse_left = landmarks["mouse_left"]
            mouse_right = landmarks["mouse_right"]

            resp.append(FaceData(
                detected_face, x, y, w, h, confidence, left_eye, right_eye, nose, mouse_left, mouse_right
            ))

    return resp
