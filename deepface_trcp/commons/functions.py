import os
import numpy as np
import pandas as pd
import cv2
import base64
from pathlib import Path
from PIL import Image
import requests
from deepface_trcp.commons.face_data import FacesData

from deepface_trcp.detectors import DlibWrapper, MtcnnWrapper, RetinaFaceWrapper,MediapipeWrapper
from PIL import Image
import math
import numpy as np
from deepface_trcp.commons import distance

# angle in degrees that two faces in the same picture should not exceed inbetween them for the fine ajust rotation to happen 
MULTI_FACE_ANGLE_THRESHOLD = 120

import tensorflow as tf
tf_version = tf.__version__
tf_major_version = int(tf_version.split(".")[0])
tf_minor_version = int(tf_version.split(".")[1])

if tf_major_version == 1:
	import keras
	from keras.preprocessing.image import load_img, save_img, img_to_array
	from keras.applications.imagenet_utils import preprocess_input
	from keras.preprocessing import image
elif tf_major_version == 2:
	from tensorflow import keras
	from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array
	from tensorflow.keras.applications.imagenet_utils import preprocess_input
	from tensorflow.keras.preprocessing import image

#--------------------------------------------------

def initialize_input(img1_path, img2_path = None):

	if type(img1_path) == list:
		bulkProcess = True
		img_list = img1_path.copy()
	else:
		bulkProcess = False

		if (
			(type(img2_path) == str and img2_path != None) #exact image path, base64 image
			or (isinstance(img2_path, np.ndarray) and img2_path.any()) #numpy array
		):
			img_list = [[img1_path, img2_path]]
		else: #analyze function passes just img1_path
			img_list = [img1_path]

	return img_list, bulkProcess

def initialize_folder():
	home = get_deepface_home()

	if not os.path.exists(home+"/.deepface"):
		os.makedirs(home+"/.deepface")
		print("Directory ", home, "/.deepface created")

	if not os.path.exists(home+"/.deepface/weights"):
		os.makedirs(home+"/.deepface/weights")
		print("Directory ", home, "/.deepface/weights created")

def get_deepface_home():
	return str(os.getenv('DEEPFACE_HOME', default=Path.home()))

def loadBase64Img(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img

def load_image(img):
	exact_image = False; base64_img = False; url_img = False

	if type(img).__module__ == np.__name__:
		exact_image = True

	elif len(img) > 11 and img[0:11] == "data:image/":
		base64_img = True

	elif len(img) > 11 and img.startswith("http"):
		url_img = True

	#---------------------------

	if base64_img == True:
		img = loadBase64Img(img)

	elif url_img:
		img = np.array(Image.open(requests.get(img, stream=True).raw).convert('RGB'))

	elif exact_image != True: #image path passed as input
		if os.path.isfile(img) != True:
			raise ValueError("Confirm that ",img," exists")

		img = cv2.imread(img)

	return img

def build_model(detector_backend):

	global face_detector_obj #singleton design pattern

	backends = {
		'dlib': DlibWrapper.build_model,
		'mtcnn': MtcnnWrapper.build_model,
		'retinaface': RetinaFaceWrapper.build_model,
		'mediapipe': MediapipeWrapper.build_model
	}

	if not "face_detector_obj" in globals():
		face_detector_obj = {}

	if not detector_backend in face_detector_obj.keys():
		face_detector = backends.get(detector_backend)

		if face_detector:
			face_detector = face_detector()
			face_detector_obj[detector_backend] = face_detector
			#print(detector_backend," built")
		else:
			raise ValueError("invalid detector_backend passed - " + detector_backend)

	return face_detector_obj[detector_backend]

def compute_score(faces_data, avg_angle):
	return max(0.01, np.linalg.norm([fd.confidence for fd in faces_data]) - (abs(avg_angle)/360)) if len(faces_data) > 0 else 0

def detect_faces(img, detector_backend = 'dlib', align_individual_faces = False, try_all_global_rotations = True, fine_adjust_global_rotation = False):
	#img might be path, base64 or numpy array. Convert it to numpy whatever it is.
	img = load_image(img)
	#original_img = img.copy()

	#detector stored in a global variable in FaceDetector object.
	#this call should be completed very fast because it will return found in memory
	#it will not build face detector model in each call (consider for loops)
	face_detector = build_model(detector_backend)
	backends = {
		'dlib': DlibWrapper.detect_faces,
		'mtcnn': MtcnnWrapper.detect_faces,
		'retinaface': RetinaFaceWrapper.detect_faces,
		'mediapipe': MediapipeWrapper.detect_faces
	}

	detect_faces = backends.get(detector_backend)
	faces_data = detect_faces(face_detector, img)
	score = np.linalg.norm([fd.confidence for fd in faces_data])
	global_scores = [(score, 0, img, faces_data)]
	# TODO: debug
	print("rotate {}: score = {:.2f}".format(0, score))
	if try_all_global_rotations:
		for i in range(1, 3+1):
			angle = i * 90
			rotated_img = np.array(Image.fromarray(img).rotate(angle))
			faces_data = detect_faces(face_detector, rotated_img)
			face_angles = [fd.angle for fd in faces_data]
			avg_angle = np.mean(face_angles)
			score = compute_score(faces_data, avg_angle)
			global_scores.append((score, angle, rotated_img, faces_data))
			# TODO: debug
			print("rotate {}: score = {:.2f}".format(angle, score))
	(score, global_angle, rotated_img, faces_data) = max(global_scores, key=lambda x: x[0])
	face_angles = [fd.angle for fd in faces_data]
	avg_angle = np.mean(face_angles)
	old_global_angle = global_angle
	if fine_adjust_global_rotation and len(faces_data) > 0:
		too_different = False
		for ang1 in face_angles:
			for ang2 in face_angles:
				if abs(ang2 - ang1) > MULTI_FACE_ANGLE_THRESHOLD:
					too_different = True
		print("angles = {} ; avg angle = {:.2f} ; too_different = {}".format(face_angles, avg_angle, too_different))
		if not too_different:
			rotated_img2 = np.array(Image.fromarray(img).rotate(avg_angle))
			faces_data2 = detect_faces(face_detector, rotated_img2)
			face_angles2 = [fd.angle for fd in faces_data2]
			avg_angle2 = np.mean(face_angles2)
			score2 = compute_score(faces_data2, avg_angle2)
			global_angle2 = global_angle + avg_angle
			if score2 > score:
				score, global_angle, rotated_img, faces_data, old_global_angle = score2, global_angle2, rotated_img2, faces_data2, global_angle
			else:
				print("score didn't improve: new {:.2f} vs old {:.2f}".format(score2, score))
	for fd in faces_data:
		if align_individual_faces:
			fd.al_sub_img = np.array(Image.fromarray(fd.sub_img).rotate(fd.angle))
		else:
			fd.al_sub_img = fd.sub_img
	return FacesData(img, score, global_angle, old_global_angle, rotated_img, faces_data)

		# img = Image.fromarray(img)
		# img = np.array(img.rotate())

	#-----------------------

	return img #return img anyway

def normalize_input(img, normalization = 'base'):

	#issue 131 declares that some normalization techniques improves the accuracy

	if normalization == 'base':
		return img
	else:
		#@trevorgribble and @davedgd contributed this feature

		img *= 255 #restore input in scale of [0, 255] because it was normalized in scale of  [0, 1] in preprocess_face

		if normalization == 'raw':
			pass #return just restored pixels

		elif normalization == 'Facenet':
			mean, std = img.mean(), img.std()
			img = (img - mean) / std

		elif(normalization=="Facenet2018"):
			# simply / 127.5 - 1 (similar to facenet 2018 model preprocessing step as @iamrishab posted)
			img /= 127.5
			img -= 1

		elif normalization == 'VGGFace':
			# mean subtraction based on VGGFace1 training data
			img[..., 0] -= 93.5940
			img[..., 1] -= 104.7624
			img[..., 2] -= 129.1863

		elif(normalization == 'VGGFace2'):
			# mean subtraction based on VGGFace2 training data
			img[..., 0] -= 91.4953
			img[..., 1] -= 103.8827
			img[..., 2] -= 131.0912

		elif(normalization == 'ArcFace'):
			#Reference study: The faces are cropped and resized to 112×112,
			#and each pixel (ranged between [0, 255]) in RGB images is normalised
			#by subtracting 127.5 then divided by 128.
			img -= 127.5
			img /= 128

	#-----------------------------

	return img

def preprocess_face(facesdata, target_size=(224, 224), grayscale = False, copy = True):
	#--------------------------
	new_faces_data = []
	for fd in facesdata.faces_data:
		if fd.al_sub_img.shape[0] == 0 or fd.al_sub_img.shape[1] == 0:
			print("removing empty face rectangle")
			continue
		if copy:
			fd.pp_sub_img = fd.al_sub_img.copy()
		else:
			fd.pp_sub_img = fd.al_sub_img

		#--------------------------

		#post-processing
		if grayscale == True:
			fd.pp_sub_img = cv2.cvtColor(fd.pp_sub_img, cv2.COLOR_BGR2GRAY)

		#---------------------------------------------------
		#resize image to expected shape

		# fd.pp_sub_img = cv2.resize(fd.pp_sub_img, target_size) #resize causes transformation on base image, adding black pixels to resize will not deform the base image

		if fd.pp_sub_img.shape[0] > 0 and fd.pp_sub_img.shape[1] > 0:
			factor_0 = target_size[0] / fd.pp_sub_img.shape[0]
			factor_1 = target_size[1] / fd.pp_sub_img.shape[1]
			factor = min(factor_0, factor_1)

			dsize = (int(fd.pp_sub_img.shape[1] * factor), int(fd.pp_sub_img.shape[0] * factor))
			fd.pp_sub_img = cv2.resize(fd.pp_sub_img, dsize)

			# Then pad the other side to the target size by adding black pixels
			diff_0 = target_size[0] - fd.pp_sub_img.shape[0]
			diff_1 = target_size[1] - fd.pp_sub_img.shape[1]
			if grayscale == False:
				# Put the base image in the middle of the padded image
				fd.pp_sub_img = np.pad(fd.pp_sub_img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2), (0, 0)), 'constant')
			else:
				fd.pp_sub_img = np.pad(fd.pp_sub_img, ((diff_0 // 2, diff_0 - diff_0 // 2), (diff_1 // 2, diff_1 - diff_1 // 2)), 'constant')

		#------------------------------------------

		#double check: if target image is not still the same size with target.
		if fd.pp_sub_img.shape[0:2] != target_size:
			fd.pp_sub_img = cv2.resize(fd.pp_sub_img, target_size)

		#---------------------------------------------------

		#normalizing the image pixels

		img_pixels = image.img_to_array(fd.pp_sub_img) #what this line doing? must?
		img_pixels = np.expand_dims(img_pixels, axis = 0)
		img_pixels /= 255 #normalize input in [0, 1]
		fd.pp_sub_img = img_pixels

		#---------------------------------------------------

		new_faces_data.append(fd)

	facesdata.faces_data = new_faces_data
	return facesdata

def find_input_shape(model):

	#face recognition models have different size of inputs
	#my environment returns (None, 224, 224, 3) but some people mentioned that they got [(None, 224, 224, 3)]. I think this is because of version issue.

	input_shape = model.layers[0].input_shape

	if type(input_shape) == list:
		input_shape = input_shape[0][1:3]
	else:
		input_shape = input_shape[1:3]

	#----------------------
	#issue 289: it seems that tf 2.5 expects you to resize images with (x, y)
	#whereas its older versions expect (y, x)

	if tf_major_version == 2 and tf_minor_version >= 5:
		x = input_shape[0]; y = input_shape[1]
		input_shape = (y, x)

	#----------------------

	if type(input_shape) == list: #issue 197: some people got array here instead of tuple
		input_shape = tuple(input_shape)

	return input_shape
