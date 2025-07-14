import time
from collections import deque
from enum import Enum

# Attention state constants
ATTENTIVE = "attentive"
LOOKING_AWAY = "looking_away"
ABSENT = "absent"
DROWSY = "drowsy"
SLEEPING = "sleeping"

DARKNESS = "darkness"

class AttentionCategory(Enum):
    ATTENTIVE = "attentive"
    DISTRACTED = "distracted"
    INACTIVE = "inactive"
    SLEEPING = "sleeping"

class AttentionState(Enum):
    ATTENTIVE = ATTENTIVE
    LOOKING_AWAY = LOOKING_AWAY
    ABSENT = ABSENT
    DROWSY = DROWSY
    SLEEPING = SLEEPING
    DARKNESS = DARKNESS

class Measurement:
    def __init__(self, brightness, contrast, face_presence, eye_openness, looking_score, drowsiness_score=0, sleeping_score=0, timestamp=None):
        self.brightness = brightness
        self.contrast = contrast
        self.face_presence = face_presence
        self.eye_openness = eye_openness
        self.looking_score = looking_score
        self.drowsiness_score = drowsiness_score
        self.sleeping_score = sleeping_score
        self.timestamp = timestamp or time.time()

class UserAttentionData:
    def __init__(self, user_id):
        self.user_id = user_id
        self.measurements = deque(maxlen=5)
        self.state_history = deque(maxlen=10)
        self.calibration_images = []
        self.last_activity = time.time()
        self.current_state = ABSENT
        self.state_since = int(time.time() * 1000)
        self.history = []

class UserCalibration:
    def __init__(self, user_id):
        self.user_id = user_id
        self.brightness_baseline = 0
        self.contrast_baseline = 0
        self.time = time.time()

class AttentionResponse:
    def __init__(self, user_id, attention_state, attention_percentage, confidence, timestamp=None):
        self.user_id = user_id
        self.attention_state = attention_state
        self.attention_percentage = attention_percentage
        self.confidence = confidence
        self.timestamp = timestamp or int(time.time() * 1000)
        self.state_since = self.timestamp
        self.attention_category = self._get_attention_category(attention_state)
    
    def _get_attention_category(self, state):
        if state == SLEEPING:
            return AttentionCategory.SLEEPING.value
        elif state in [LOOKING_AWAY, DROWSY]:
            return AttentionCategory.DISTRACTED.value
        elif state in [ABSENT, DARKNESS]:
            return AttentionCategory.INACTIVE.value
        else:
            return AttentionCategory.ATTENTIVE.value
    
    def to_dict(self):
        return {
            'userId': self.user_id,
            'attentionState': self.attention_state,
            'attentionCategory': self.attention_category,
            'stateSince': self.state_since,
            'attentionPercentage': self.attention_percentage,
            'confidence': round(self.confidence * 100, 1),
            'timestamp': self.timestamp
        }

class RoomAttentionResponse:
    def __init__(self, room_id, attention_data, timestamp=None):
        self.room_id = room_id
        self.attention = attention_data
        self.timestamp = timestamp or int(time.time() * 1000)
    
    def to_dict(self):
        return {
            'roomId': self.room_id,
            'attention': self.attention,
            'timestamp': self.timestamp
        }

class CalibrationResponse:
    def __init__(self, user_id, success, timestamp=None):
        self.user_id = user_id
        self.calibration_success = success
        self.timestamp = timestamp or int(time.time() * 1000)
    
    def to_dict(self):
        return {
            'userId': self.user_id,
            'calibrationSuccess': self.calibration_success,
            'timestamp': self.timestamp
        } 