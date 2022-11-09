import numpy as np
import math
from deepface_trcp.commons import distance

def sort_asc_x(a, b): return (a, b) if a[0] <= b[0] else (b, a)
def sort_asc_y(a, b): return (a, b) if a[1] <= b[1] else (b, a)

def get_face_angle(left_eye, right_eye):
	# TODO: use nose?

	#this function aligns given face in img based on left and right eye coordinates

	left_eye_x, left_eye_y = left_eye
	right_eye_x, right_eye_y = right_eye

	#-----------------------
	#find rotation direction

	if left_eye_y > right_eye_y:
		point_3rd = (right_eye_x, left_eye_y)
		direction = -1 #rotate same direction to clock
	else:
		point_3rd = (left_eye_x, right_eye_y)
		direction = 1 #rotate inverse direction of clock

	#-----------------------
	#find length of triangle edges

	a = distance.findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
	b = distance.findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
	c = distance.findEuclideanDistance(np.array(right_eye), np.array(left_eye))

	#-----------------------

	#apply cosine rule

	if b != 0 and c != 0: #this multiplication causes division by zero in cos_a calculation

		cos_a = (b*b + c*c - a*a)/(2*b*c)
		angle = np.arccos(cos_a) #angle in radian
		angle = (angle * 180) / math.pi #radian to degree

		#-----------------------
		#rotate base image

		if direction == -1:
			angle = 90 - angle
		final_angle = direction * angle
	else:
		final_angle = 0
	return final_angle

class FaceData:
    def __init__(
        self,
        sub_img,
        x,
        y,
        w,
        h,
        confidence,
        left_eye,
        right_eye,
        nose,
        mouth_left,
        mouth_right
    ) -> None:
        self.sub_img = sub_img
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.region = [x, y, w, h]
        self.confidence = confidence
        mid_eye_point = (np.array(left_eye) + np.array(right_eye)) / 2
        if nose[1] >= left_eye[1] and nose[1] >= right_eye[1]:
            (self.left_eye, self.right_eye) = sort_asc_x(left_eye, right_eye)
        elif nose[1] <= left_eye[1] and nose[1] <= right_eye[1]:
            (self.right_eye, self.left_eye) = sort_asc_x(left_eye, right_eye)
        elif nose[0] >= left_eye[0] and nose[0] >= right_eye[0]:
            (self.right_eye, self.left_eye) = sort_asc_y(left_eye, right_eye)
        elif nose[0] <= left_eye[0] and nose[0] <= right_eye[0]:
            (self.left_eye, self.right_eye) = sort_asc_y(left_eye, right_eye)
        elif nose[1] >= mid_eye_point[1]:
            (self.left_eye, self.right_eye) = sort_asc_x(left_eye, right_eye)
        elif nose[1] <= mid_eye_point[1]:
            (self.right_eye, self.left_eye) = sort_asc_x(left_eye, right_eye)
        elif nose[0] >= mid_eye_point[0]:
            (self.right_eye, self.left_eye) = sort_asc_y(left_eye, right_eye)
        else:
            # nose[0] <= mid_eye_point[0]:
            (self.left_eye, self.right_eye) = sort_asc_y(left_eye, right_eye)
        self.nose = nose
        self.mouth_left = mouth_left
        self.mouth_right = mouth_right
        self.angle = get_face_angle(self.left_eye, self.right_eye)
        self.al_sub_img = None
        self.pp_sub_img = None
        self.emotion = None
        self.age = None
        self.gender = None
        self.race = None
    def as_dict(self):
        mouth_left = {"x": int(self.mouth_left[0]), "y": int(self.mouth_left[1])} if self.mouth_left is not None else None
        mouth_right = {"x": int(self.mouth_right[0]), "y": int(self.mouth_right[1])} if self.mouth_right is not None else None
        return {
            "x": int(self.x),
            "y": int(self.y),
            "w": int(self.w),
            "h": int(self.h),
            "confidence": float(self.confidence),
            "face_landmarks": {
                "left_eye": {"x": int(self.left_eye[0]), "y": int(self.left_eye[1])},
                "right_eye": {"x": int(self.right_eye[0]), "y": int(self.right_eye[1])},
                "nose": {"x": int(self.nose[0]), "y": int(self.nose[1])},
                "mouth_left": mouth_left,
                "mouth_right": mouth_right
            },
            "angle": float(self.angle),
            # already sanitized below
            "emotion": self.emotion,
            "age": self.age,
            "gender": self.gender,
            "race": self.race
        }

class FacesData:
    def __init__(self, original_img, global_score, global_angle, original_global_angle, rotated_img, faces_data):
        self.original_img = original_img
        self.global_score = global_score
        self.global_angle = global_angle
        self.original_global_angle = original_global_angle
        self.rotated_img = rotated_img
        self.faces_data = faces_data
    def as_dict(self):
        return {
            "score": float(self.global_score),
            "total_rotation": float(self.global_angle),
            "initial_rotation": float(self.original_global_angle),
            "fine_grained_rotation": float(self.global_angle - self.original_global_angle),
            "faces": [fd.as_dict() for fd in self.faces_data]
        }
