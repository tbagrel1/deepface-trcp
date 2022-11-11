import numpy as np
import math
from deepface_trcp.commons import distance
from PIL import Image
from PIL.Image import Resampling
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def sort_asc_x(a, b): return (a, b) if a[0] <= b[0] else (b, a)
def sort_asc_y(a, b): return (a, b) if a[1] <= b[1] else (b, a)

def get_face_angle(left_eye, right_eye):
    return np.rad2deg(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
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
            "top_left_corner_x": int(self.x),
            "top_left_corner_y": int(self.y),
            "width": int(self.w),
            "height": int(self.h),
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
    def __init__(self, backend, crop_margin_ratio, original_img, global_score, global_angle, original_global_angle, rotated_img, faces_data):
        self.backend = backend
        self.crop_margin_ratio = crop_margin_ratio
        self.original_img = original_img
        self.global_score = global_score
        self.global_angle = global_angle
        self.original_global_angle = original_global_angle
        self.rotated_img = rotated_img
        self.faces_data = faces_data
    def as_dict(self):
        return {
            "backend": self.backend,
            "crop_margin_ratio": float(self.crop_margin_ratio),
            "score": float(self.global_score),
            "total_rotation": float(self.global_angle),
            "initial_rotation": float(self.original_global_angle),
            "fine_grained_rotation": float(self.global_angle - self.original_global_angle),
            "faces": [fd.as_dict() for fd in self.faces_data]
        }
    def draw(self):
        # Create figure and axes
        fig, ax = plt.subplots()
        rotated_img = np.array(Image.fromarray(self.original_img).rotate(self.original_global_angle))
        rotated_img2 = np.array(Image.fromarray(rotated_img).rotate(self.global_angle - self.original_global_angle, resample=Resampling.BILINEAR))
        # Display the image
        ax.imshow(rotated_img2)
        for fd in self.faces_data:
            rect = patches.Rectangle((fd.x, fd.y), fd.w, fd.h, linewidth=1, edgecolor='r', facecolor='none')
            left_eye = patches.Circle(fd.left_eye, 2, linewidth=1, edgecolor='g', facecolor='none')
            right_eye = patches.Circle(fd.right_eye, 2, linewidth=1, edgecolor='b', facecolor='none')
            nose = patches.Circle(fd.nose, 2, linewidth=1, edgecolor='y', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
            ax.add_patch(left_eye)
            ax.add_patch(right_eye)
            ax.add_patch(nose)
        plt.show()
