from pathlib import Path
from deepface_trcp.commons.face_data import FaceData
import gdown
import bz2
import os
import numpy as np

from deepface.commons import functions

def build_model():

	home = functions.get_deepface_home()

	import dlib #this requirement is not a must that's why imported here

	#check required file exists in the home/.deepface/weights folder
	if os.path.isfile(home+'/.deepface/weights/shape_predictor_5_face_landmarks.dat') != True:

		print("shape_predictor_5_face_landmarks.dat.bz2 is going to be downloaded")

		url = "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2"
		output = home+'/.deepface/weights/'+url.split("/")[-1]

		gdown.download(url, output, quiet=False)

		zipfile = bz2.BZ2File(output)
		data = zipfile.read()
		newfilepath = output[:-4] #discard .bz2 extension
		open(newfilepath, 'wb').write(data)

	face_detector = dlib.get_frontal_face_detector()
	sp = dlib.shape_predictor(home+"/.deepface/weights/shape_predictor_5_face_landmarks.dat")

	detector = {}
	detector["face_detector"] = face_detector
	detector["sp"] = sp
	return detector

def dlib_shape_to_np_array(shape):
    """Converts a dlib shape object (i.e. dlib.full_object_detection) to a numpy array
    Parameters
    ----------
    shape : dlib.full_object_detection
        The shape to be converted
    Returns
    -------
    numpy.ndarray
        The resulting array
    """
    # Init empty array
    arr = np.zeros((shape.num_parts, 2), dtype=np.int)

    # Convert each landmark to a (x, y) tuple
    for i in range(0, shape.num_parts):
        arr[i] = (shape.part(i).x, shape.part(i).y)

    return arr

def detect_faces(detector, img, crop_margin_ratio):
	import dlib #this requirement is not a must that's why imported here

	resp = []

	home = str(Path.home())

	sp = detector["sp"]

	detected_face = None

	face_detector = detector["face_detector"]

	#note that, by design, dlib's fhog face detector scores are >0 but not capped at 1
	detections, scores, _ = face_detector.run(img, 1)

	for idx, d in enumerate(detections):
		left = d.left(); right = d.right()
		top = d.top(); bottom = d.bottom()
		crop_margin = round(max(right - left, bottom - top) * crop_margin_ratio)
		x = max(0, left - crop_margin)
		x2 = min(right + crop_margin, img.shape[1])
		y = max(0, top - crop_margin)
		y2 = min(bottom + crop_margin, img.shape[0])
		w = x2 - x
		h = y2 - y
		detected_face = img[y: y2, x: x2]
		confidence = scores[idx]
		
		shape_detection = sp(img, detections[idx])
		landmarks = dlib_shape_to_np_array(shape_detection)
		# Get eye and nose points
		eye_l_points = landmarks[0:1]
		eye_r_points = landmarks[2:3]
		nose = landmarks[4]

		# Calculate eye center
		left_eye = eye_l_points.mean(axis=0).astype(np.int)
		right_eye = eye_r_points.mean(axis=0).astype(np.int)

		resp.append(FaceData(detected_face, x, y, w, h, confidence, left_eye, right_eye, nose, None, None))


	return resp
