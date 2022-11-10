import cv2
from deepface_trcp.commons.face_data import FaceData

def build_model():
	from mtcnn import MTCNN
	face_detector = MTCNN()
	return face_detector

def detect_faces(face_detector, img, crop_margin_ratio):

	resp = []

	detected_face = None

	img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #mtcnn expects RGB but OpenCV read BGR
	detections = face_detector.detect_faces(img_rgb)
	img_width = img.shape[1]; img_height = img.shape[0]
	for detection in detections:
		x, y, w, h = [int(v) for v in detection["box"]]
		x2 = x + w
		y2 = y + h
		crop_margin = round(max(w, h) * crop_margin_ratio)
		x = max(0, x - crop_margin)
		x2 = min(x2 + crop_margin, img_width)
		w = x2 - x
		y = max(0, y - crop_margin)
		y2 = min(y2 + crop_margin, img_height)
		h = y2 - y
		detected_face = img[y:y2, x:x2]
		confidence = detection["confidence"]
		keypoints = detection["keypoints"]
		left_eye = keypoints["left_eye"]
		right_eye = keypoints["right_eye"]
		nose = keypoints["nose"]
		mouth_left = keypoints["mouth_left"]
		mouth_right = keypoints["mouth_right"]
		resp.append(FaceData(detected_face, x, y, w, h, confidence, left_eye, right_eye, nose, mouth_left, mouth_right))

	return resp
