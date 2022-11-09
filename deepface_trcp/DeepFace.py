import warnings
warnings.filterwarnings("ignore")

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from os import path
import numpy as np

from deepface_trcp.extendedmodels import Age, Gender, Race, Emotion
from deepface_trcp.commons import functions, distance as dst

import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])
if tf_version == 2:
	import logging
	tf.get_logger().setLevel(logging.ERROR)

def build_model(model_name):

	"""
	This function builds a deepface model
	Parameters:
		model_name (string): face recognition or facial attribute model
			VGG-Face, Facenet, OpenFace, DeepFace, DeepID for face recognition
			Age, Gender, Emotion, Race for facial attributes

	Returns:
		built deepface model
	"""

	global model_obj #singleton design pattern

	models = {
		# 'VGG-Face': VGGFace.loadModel,
		# 'OpenFace': OpenFace.loadModel,
		# 'Facenet': Facenet.loadModel,
		# 'Facenet512': Facenet512.loadModel,
		# 'DeepFace': FbDeepFace.loadModel,
		# 'DeepID': DeepID.loadModel,
		# 'Dlib': DlibWrapper.loadModel,
		# 'ArcFace': ArcFace.loadModel,
		# 'SFace': SFace.load_model,
		'emotion': Emotion.loadModel,
		'age': Age.loadModel,
		'gender': Gender.loadModel,
		'race': Race.loadModel
	}

	if not "model_obj" in globals():
		model_obj = {}

	if not model_name in model_obj.keys():
		model = models.get(model_name)
		if model:
			model = model()
			model_obj[model_name] = model
			#print(model_name," built")
		else:
			raise ValueError('Invalid model_name passed - {}'.format(model_name))

	return model_obj[model_name]

def analyze(img_path, actions = ('emotion', 'age', 'gender', 'race'), detector_backend = 'dlib', align_individual_faces = False, try_all_global_rotations = True, fine_adjust_global_rotation = 'off', force_copy = False):

	"""
	This function analyzes facial attributes including age, gender, emotion and race

	Parameters:
		img_path: exact image path, numpy array (BGR) or base64 encoded image could be passed.

		actions (tuple): The default is ('age', 'gender', 'emotion', 'race'). You can drop some of those attributes.

		detector_backend (string): set face detector backend in ('retinaface', 'mtcnn', 'dlib', 'mediapipe').

		align_individual_faces (boolean): should every face be aligned on its own before executing actions?

		try_all_global_rotations (boolean): should all four rotations be tested to detect a maximum of faces?

		fine_adjust_global_rotation (string):
			'off': the global image will just be rotated by 1/4 of turns
			'safe': the global image will get fine-grained rotated if it improves the face detection score
			'force': the global image will be rotated to align a maximum of faces, even if it decreases the new detection score
		
		force_copy (boolean): should intermediate buffer copies be stored in the FaceData object for each detected faces? This is useful for debug only.
	"""

	actions = list(actions)
	models = {}

	img_paths, _ = functions.initialize_input(img_path)
	if len(img_paths) > 1:
		raise ValueError("Multiple images are not supported in this mode")
	img_path = img_paths[0]

	#---------------------------------

	if 'emotion' in actions:
		models['emotion'] = build_model("emotion")

	if 'age' in actions:
		models['age'] = build_model("age")

	if 'gender' in actions:
		models['gender'] = build_model('gender')

	if 'race' in actions:
		models['race'] = build_model('race')

	#---------------------------------

	facesdata = functions.detect_faces(img_path, detector_backend, align_individual_faces, try_all_global_rotations, fine_adjust_global_rotation)
	pp_facesdata = facesdata # for the case with actions = []
	copy_in_preprocess = ("emotion" in actions and len(actions) > 1) or force_copy
	if "emotion" in actions:
		emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
		copy_in_preprocess = len(actions) > 1
		pp_facesdata = functions.preprocess_face(facesdata, target_size = (48, 48), grayscale = True, copy = copy_in_preprocess)
		for fd in pp_facesdata.faces_data:
			emotion_predictions = models['emotion'].predict(fd.pp_sub_img, verbose=0)[0,:]
			sum_of_predictions = emotion_predictions.sum()
			resp_obj = {}
			for i in range(0, len(emotion_labels)):
				emotion_label = emotion_labels[i]
				emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
				resp_obj[emotion_label] = float(emotion_prediction)
			resp_obj["dominant_emotion"] = emotion_labels[np.argmax(emotion_predictions)]
			fd.emotion = resp_obj
	if "age" in actions or "gender" in actions or "race" in actions:
		gender_labels = ["woman", "man"]
		race_labels = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']
		pp_facesdata = functions.preprocess_face(facesdata, target_size = (224, 224), grayscale = False, copy = force_copy)
		for fd in pp_facesdata.faces_data:
			if "age" in actions:
				age_predictions = models['age'].predict(fd.pp_sub_img, verbose=0)[0,:]
				apparent_age = Age.findApparentAge(age_predictions)
				fd.age = float(apparent_age)
			if "gender" in actions:
				gender_predictions = models['gender'].predict(fd.pp_sub_img, verbose=0)[0,:]
				resp_obj = {}
				for i, gender_label in enumerate(gender_labels):
					gender_prediction = 100 * gender_predictions[i]
					resp_obj[gender_label] = float(gender_prediction)
				resp_obj["dominant_gender"] = gender_labels[np.argmax(gender_predictions)]
				fd.gender = resp_obj
			if "race" in actions:
				race_predictions = models['race'].predict(fd.pp_sub_img, verbose=0)[0,:]
				sum_of_predictions = race_predictions.sum()
				resp_obj = {}
				for i in range(0, len(race_labels)):
					race_label = race_labels[i]
					race_prediction = 100 * race_predictions[i] / sum_of_predictions
					resp_obj[race_label] = float(race_prediction)
				resp_obj["dominant_race"] = race_labels[np.argmax(race_predictions)]
				fd.race = resp_obj
	#---------------------------------
	return pp_facesdata


functions.initialize_folder()
