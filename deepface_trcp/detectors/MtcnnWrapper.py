import cv2
from deepface_trcp.commons.face_data import FaceData

def build_model():
	from mtcnn import MTCNN
	face_detector = MTCNN()
	return face_detector

def detect_faces(face_detector, img):

	resp = []

	detected_face = None

	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #mtcnn expects RGB but OpenCV read BGR
	detections = face_detector.detect_faces(img_rgb)

	for detection in detections:
		x, y, w, h = detection["box"]
		detected_face = img[int(y):int(y+h), int(x):int(x+w)]
		confidence = detection["confidence"]
		keypoints = detection["keypoints"]
		left_eye = keypoints["left_eye"]
		right_eye = keypoints["right_eye"]
		nose = keypoints["nose"]
		mouse_left = landmarks["mouse_left"]
		mouse_right = landmarks["mouse_right"]
		resp.append(FaceData(detected_face, x, y, w, h, confidence, left_eye, right_eye, nose, mouse_left, mouse_right))

	return resp
