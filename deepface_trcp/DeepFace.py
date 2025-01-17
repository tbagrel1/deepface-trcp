import warnings
warnings.filterwarnings("ignore")

import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import cv2
from deepface_trcp.extendedmodels import Age, Gender, Race, Emotion
from deepface_trcp.commons import functions, distance as dst

import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])
if tf_version == 2:
	import logging
	tf.get_logger().setLevel(logging.ERROR)

# https://github.com/yu4u/age-gender-estimation
EXTERNAL_WEIGHTS_URL = "https://github.com/yu4u/age-gender-estimation/releases/download/v0.6/EfficientNetB3_224_weights.11-3.44.hdf5"
EXTERNAL_WEIGHTS_FILE_NAME = "EfficientNetB3_224_weights.11-3.44.hdf5"
EXTERNAL_WEIGHTS_HASH = "6d7f7b7ced093a8b3ef6399163da6ece"
EXTERNAL_MODEL_SIZE = None

def load_external_model():
	global EXTERNAL_MODEL_SIZE
	from tensorflow.keras.utils import get_file
	from pathlib import Path
	from omegaconf import OmegaConf
	from age_gender_estimation_trcp.factory import get_model
	weights_file = get_file(EXTERNAL_WEIGHTS_FILE_NAME, EXTERNAL_WEIGHTS_URL, file_hash=EXTERNAL_WEIGHTS_HASH)
	model_name, EXTERNAL_MODEL_SIZE = Path(weights_file).stem.split("_")[:2]
	EXTERNAL_MODEL_SIZE = int(EXTERNAL_MODEL_SIZE)
	config = OmegaConf.from_dotlist([f"model.model_name={model_name}", f"model.img_size={EXTERNAL_MODEL_SIZE}"])
	model = get_model(config)
	model.load_weights(weights_file)
	return model

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
		'race': Race.loadModel,
		'external_age_gender': load_external_model
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

def analyze(img_path, actions = ('emotion', 'age', 'gender', 'race', 'external_age_gender'), detector_backends = (('dlib', 0.11), ('retinaface', 0.07)), align_individual_faces = True, try_global_rotations = 'eco', fine_adjust_global_rotation = 'quarter_safe', force_copy = False):

	"""
	This function analyzes facial attributes including age, gender, emotion and race

	Parameters:
		img_path: exact image path, numpy array (BGR) or base64 encoded image could be passed.

		actions (tuple): The default is ('age', 'gender', 'emotion', 'race', 'external_age_gender'). You can drop some of those attributes.

		detector_backends (list of tuples): list of (backend, crop_margin_ratio) that should be tried in order until a face is found. backend in a value in ['retinaface', 'mtcnn', 'dlib'] and crop_margin_ratio is a value in [-1..1] corresponding to the added or reduced margin around each face box.
		Crop margin ratio recommanded values are ["dlib" -> 0.11), "mtcnn" -> 0.08, "retinaface" -> 0.07]

		align_individual_faces (boolean): should every face be aligned on its own before executing actions?

		try_global_rotations (string):
			'off': detection will fail if no face can't be found with the original image orientation
			'eco': the image will be rotated by 1/4 of turns until at least one face if found
			'full': all four rotations will be tested to detect a maximum of faces

		fine_adjust_global_rotation (string):
			'off': the global image won't be rotated after face detection even if faces are upside-down
			'quarter_safe': the global image will just be rotated by 1/4 of turns if it improves the face detection score
			'safe': the global image will get fine-grained rotated if it improves the face detection score
			'force': the global image will be fine-grained rotated rotated to align a maximum of faces, even if it decreases the new detection score

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
	
	if 'external_age_gender' in actions:
		models['external_age_gender'] = build_model('external_age_gender')

	#---------------------------------

	for backend, _ in detector_backends:
		functions.build_model(backend)
	
	for detector_backend, crop_margin_ratio in detector_backends:
		facesdata = functions.detect_faces(img_path, detector_backend, align_individual_faces, try_global_rotations, fine_adjust_global_rotation, crop_margin_ratio)
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
			# gender_labels = ["woman", "man"]
			race_labels = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']
			copy_in_preprocess = "external_age_gender" in actions or force_copy
			pp_facesdata = functions.preprocess_face(facesdata, target_size = (224, 224), grayscale = False, copy = copy_in_preprocess)
			for fd in pp_facesdata.faces_data:
				if "age" in actions:
					age_predictions = models['age'].predict(fd.pp_sub_img, verbose=0)[0,:]
					apparent_age = Age.findApparentAge(age_predictions)
					if fd.age is None:
						fd.age = {}
					fd.age["deepface"] = float(apparent_age)
				if "gender" in actions:
					gender_predictions = models['gender'].predict(fd.pp_sub_img, verbose=0)[0,:]
					if fd.gender is None:
						fd.gender = {}
					fd.gender["deepface"] = float(gender_predictions[0])
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
		if "external_age_gender" in actions and len(pp_facesdata.faces_data) > 0:
			face_frames = np.empty((len(pp_facesdata.faces_data), EXTERNAL_MODEL_SIZE, EXTERNAL_MODEL_SIZE, 3))
			for i, fd in enumerate(pp_facesdata.faces_data):
				if force_copy:
					bgr_image = fd.al_sub_img.copy()
				else:
					bgr_image = fd.al_sub_img
				pp_sub_img = cv2.resize(bgr_image, (EXTERNAL_MODEL_SIZE, EXTERNAL_MODEL_SIZE))
				fd.pp_sub_img = pp_sub_img
				face_frames[i] = pp_sub_img
			results = models["external_age_gender"].predict(face_frames)
			predicted_genders = results[0]
			predicted_ages = results[1].dot(np.arange(0, 101).reshape(101, 1)).flatten()
			for fd, predicted_gender, predicted_age in zip(pp_facesdata.faces_data, predicted_genders, predicted_ages):
				if fd.age is None:
					fd.age = {}
				fd.age["external"] = float(predicted_age)
				if fd.gender is None:
					fd.gender = {}
				fd.gender["external"] = float(predicted_gender[0])
			
		#---------------------------------
		for fd in pp_facesdata.faces_data:
			if "external_age_gender" in actions and "age" in actions:
				fd.age["dominant_age"] = np.mean([fd.age["deepface"], fd.age["external"]])
			elif "external_age_gender" in actions:
				fd.age["dominant_age"] = fd.age["external"]
			elif "age" in actions:
				fd.age["dominant_age"] = fd.age["deepface"]
			
			if "external_age_gender" in actions and "gender" in actions:
				if fd.gender["deepface"] < 0.5 and fd.gender["external"] < 0.5:
					fd.gender["dominant_gender"] = "male"
				elif fd.gender["deepface"] >= 0.5 and fd.gender["external"] >= 0.5:
					fd.gender["dominant_gender"] = "female"
				else:
					fd.gender["dominant_gender"] = "unknown"
			elif "external_age_gender" in actions:
				fd.gender["dominant_gender"] = "male" if fd.gender["external"] < 0.5 else "female"
			elif "gender" in actions:
				fd.gender["dominant_gender"] = "male" if fd.gender["deepface"] < 0.5 else "female"

		if len(pp_facesdata.faces_data) > 0:
			break
	return pp_facesdata


functions.initialize_folder()
