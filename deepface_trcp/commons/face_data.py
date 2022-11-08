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
        self.left_eye = left_eye
        self.right_eye = right_eye
        self.nose = nose
        self.mouth_left = mouth_left
        self.mouth_right = mouth_right
        self.angle = None
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
