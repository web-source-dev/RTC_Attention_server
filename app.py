import os
import time
import base64
import json
import io
import math
import threading
import numpy as np
import cv2
import mediapipe as mp
from statistics import mean, stdev
from collections import deque
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageStat, ImageFilter, ImageEnhance, ImageOps

app = Flask(__name__)
CORS(app)

mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

face_detection = mp_face_detection.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

pose_detection = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

user_attention_data = {}

user_calibration = {}

ATTENTIVE = "attentive"
LOOKING_AWAY = "looking_away"
ABSENT = "absent"
ACTIVE = "active"
DROWSY = "drowsy"
SLEEPING = "sleeping"
DARKNESS = "darkness"

processing_lock = threading.Lock()

def decode_base64_image(base64_string):
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]
    
    image_bytes = base64.b64decode(base64_string)
    pil_image = Image.open(io.BytesIO(image_bytes))
    
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return pil_image, cv_image

def analyze_image_brightness(image):
    gray_image = image.convert('L')
    stat = ImageStat.Stat(gray_image)
    brightness = stat.mean[0]
    
    return brightness

def analyze_image_contrast(image):
    gray_image = image.convert('L')
    
    hist = gray_image.histogram()
    
    pixel_count = sum(hist)
    if pixel_count == 0:
        return 0
    
    mean_val = sum(i * hist[i] for i in range(256)) / pixel_count
    variance = sum(((i - mean_val) ** 2) * hist[i] for i in range(256)) / pixel_count
    contrast = math.sqrt(variance)
    
    return contrast

def detect_face_mediapipe(cv_image):

    image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)
    
    if not results.detections:
        return None, 0.0
    
    detection = results.detections[0]
    confidence = detection.score[0]
    
    bbox = detection.location_data.relative_bounding_box
    h, w, _ = cv_image.shape
    bbox_coords = {
        'xmin': int(bbox.xmin * w),
        'ymin': int(bbox.ymin * h),
        'width': int(bbox.width * w),
        'height': int(bbox.height * h)
    }
    
    return bbox_coords, confidence

def detect_face_mesh_mediapipe(cv_image):
    image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    
    return results

def detect_pose_mediapipe(cv_image):
    image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    results = pose_detection.process(image_rgb)
    
    return results

def get_eye_landmarks(face_landmarks, face_oval_indices):
    left_eye_indices = list(range(362, 374))
    right_eye_indices = list(range(33, 46))
    
    left_eye_landmarks = [face_landmarks.landmark[i] for i in left_eye_indices]
    right_eye_landmarks = [face_landmarks.landmark[i] for i in right_eye_indices]
    
    return left_eye_landmarks, right_eye_landmarks

def calculate_eye_aspect_ratio(eye_landmarks, image_shape):
    if not eye_landmarks:
        return 0.0
    
    h, w = image_shape[0:2]
    landmarks_px = [(int(point.x * w), int(point.y * h)) for point in eye_landmarks]
    
    center_x = sum(x for x, _ in landmarks_px) / len(landmarks_px)
    center_y = sum(y for _, y in landmarks_px) / len(landmarks_px)
    
    horizontal_points = [landmarks_px[0], landmarks_px[3]]
    vertical_points = [landmarks_px[1], landmarks_px[5]]
    
    width = math.dist(horizontal_points[0], horizontal_points[1])
    height = math.dist(vertical_points[0], vertical_points[1])
    
    ear = height / (width + 1e-6)
    
    return ear

def detect_head_orientation(face_landmarks, image_shape):   
    if not face_landmarks:
        return 0.0, 0.0, 0.0
    
    h, w = image_shape[0:2]
    
    nose_tip = face_landmarks.landmark[4]
    nose_tip_px = (int(nose_tip.x * w), int(nose_tip.y * h))
    
    left_cheek = face_landmarks.landmark[234]
    right_cheek = face_landmarks.landmark[454]
    
    left_cheek_px = (int(left_cheek.x * w), int(left_cheek.y * h))
    right_cheek_px = (int(right_cheek.x * w), int(right_cheek.y * h))
    
    face_width = math.dist(left_cheek_px, right_cheek_px)
    face_center_x = (left_cheek_px[0] + right_cheek_px[0]) / 2

    horizontal_position = (face_center_x - (w / 2)) / (w / 2)
    
    nose_to_center_line = abs(nose_tip_px[0] - face_center_x)
    
    yaw = horizontal_position
    
    roll = math.atan2(right_cheek_px[1] - left_cheek_px[1], right_cheek_px[0] - left_cheek_px[0])
    roll = math.degrees(roll)
    
    forehead = face_landmarks.landmark[10]
    chin = face_landmarks.landmark[152]
    
    forehead_px = (int(forehead.x * w), int(forehead.y * h))
    chin_px = (int(chin.x * w), int(chin.y * h))
    
    face_height = math.dist(forehead_px, chin_px)
    nose_to_face_height_ratio = (nose_tip_px[1] - forehead_px[1]) / (face_height + 1e-6)
    
    pitch = (nose_to_face_height_ratio - 0.5) * 2
    
    return yaw, pitch, roll

def analyze_face_present(pil_image, cv_image):

    face_bbox, confidence = detect_face_mediapipe(cv_image)
    
    if face_bbox is None:
        return 0
    
    h, w, _ = cv_image.shape
    face_x = face_bbox['xmin'] + (face_bbox['width'] / 2)
    face_y = face_bbox['ymin'] + (face_bbox['height'] / 2)
    
    rel_x = (face_x - (w/2)) / (w/2)
    rel_y = (face_y - (h/2)) / (h/2)
    
    center_distance = math.sqrt(rel_x**2 + rel_y**2)
    
    face_size_ratio = (face_bbox['width'] * face_bbox['height']) / (w * h)
    
    face_aspect_ratio = face_bbox['width'] / max(face_bbox['height'], 1)
    
    print(f"  Face position: ({rel_x:.2f}, {rel_y:.2f}), distance from center: {center_distance:.2f}")
    print(f"  Face size ratio: {face_size_ratio:.3f}, aspect ratio: {face_aspect_ratio:.2f}")
    
    adjusted_confidence = confidence
    
    if center_distance > 0.7:
        adjusted_confidence *= (1 - (center_distance - 0.7) / 0.3)
    
    if face_size_ratio < 0.05:
        adjusted_confidence *= (face_size_ratio / 0.05)
    
    if face_aspect_ratio < 0.7:
        adjusted_confidence *= (face_aspect_ratio / 0.7)
    
    return adjusted_confidence * 100

def analyze_eye_area(pil_image, cv_image):

    face_mesh_results = detect_face_mesh_mediapipe(cv_image)
    
    if not face_mesh_results.multi_face_landmarks:
        return 0
    
    face_landmarks = face_mesh_results.multi_face_landmarks[0]
    
    face_oval_indices = list(mp_face_mesh.FACEMESH_FACE_OVAL)
    
    left_eye_landmarks, right_eye_landmarks = get_eye_landmarks(face_landmarks, face_oval_indices)
    
    left_ear = calculate_eye_aspect_ratio(left_eye_landmarks, cv_image.shape)
    right_ear = calculate_eye_aspect_ratio(right_eye_landmarks, cv_image.shape)
    
    eye_difference = abs(left_ear - right_ear)
    eye_difference_ratio = eye_difference / max(max(left_ear, right_ear), 0.01)

    avg_ear = (left_ear + right_ear) / 2
    
    if avg_ear < 0.1:
        openness_score = avg_ear * 50
    elif avg_ear < 0.2:
        openness_score = 5 + ((avg_ear - 0.1) * 100)
    elif avg_ear < 0.3:
        openness_score = 15 + ((avg_ear - 0.2) * 150)
    else:
        openness_score = 30 + ((avg_ear - 0.3) * 200)
    
    openness_score = min(100, max(0, openness_score))
    
    print(f"  Left EAR: {left_ear:.3f}, Right EAR: {right_ear:.3f}, Avg: {avg_ear:.3f}")
    print(f"  Eye difference ratio: {eye_difference_ratio:.3f}")
    print(f"  Eye openness score: {openness_score:.1f}")
    
    if eye_difference_ratio > 0.4:
        print("  Detected asymmetric eyes - possibly looking to the side")
        openness_score = max(0, openness_score * 0.7)
    
    return openness_score

def analyze_head_position(pil_image, cv_image):

    face_mesh_results = detect_face_mesh_mediapipe(cv_image)
    
    if not face_mesh_results.multi_face_landmarks:
            return 0.0
    
    face_landmarks = face_mesh_results.multi_face_landmarks[0]
    
    yaw, pitch, roll = detect_head_orientation(face_landmarks, cv_image.shape)
    
    
    yaw_factor = max(0, 1.0 - pow(abs(yaw) * 2.5, 2))
    
    pitch_factor = max(0, 1.0 - pow(abs(pitch) * 2, 2))
    
    roll_normalized = abs(roll) / 90.0
    roll_factor = max(0, 1.0 - roll_normalized)
    
    looking_score = (yaw_factor * 0.6) + (pitch_factor * 0.3) + (roll_factor * 0.1)
    
    print(f"  Head position - yaw: {yaw:.2f}, pitch: {pitch:.2f}, roll: {roll:.2f}")
    print(f"  Looking score: {looking_score:.2f}")
    
    return looking_score

def calibrate_user(pil_image, cv_image, user_id):
    if user_id not in user_calibration:
        face_presence = analyze_face_present(pil_image, cv_image)
        
        if face_presence > 20:
            user_calibration[user_id] = {
                'brightness_baseline': analyze_image_brightness(pil_image),
                'contrast_baseline': analyze_image_contrast(pil_image),
                'time': time.time()
            }
            return True
    
    return False

def get_attention_state_confidence(measurements, current_state, user_id):
    if len(measurements) < 3:
            return 0.6
    
    edge_intensities = [m.get('eye_openness', 0) for m in measurements]
    face_presences = [m.get('face_presence', 0) for m in measurements]
    
    try:
        edge_consistency = 1.0 - min(1.0, stdev(edge_intensities) / max(1, mean(edge_intensities)))
        face_consistency = 1.0 - min(1.0, stdev(face_presences) / max(1, mean(face_presences)))
    except:
        edge_consistency = 0.5
        face_consistency = 0.5
    
    confidence = (edge_consistency * 0.6) + (face_consistency * 0.4)
    
    if current_state == ABSENT and mean(face_presences) < 5:
        confidence = max(confidence, 0.9)
    elif current_state == DARKNESS:
        confidence = max(confidence, 0.95)
    elif current_state == ATTENTIVE and mean(edge_intensities) > 30:
        confidence = max(confidence, 0.8)
    
    return min(1.0, confidence)

def detect_attention(pil_image, cv_image, user_id):
    if user_id not in user_attention_data:
        user_attention_data[user_id] = {
            'measurements': deque(maxlen=10),
            'state_history': deque(maxlen=20),
            'calibration_images': []
        }
        
        calibrate_user(pil_image, cv_image, user_id)
    
    brightness = analyze_image_brightness(pil_image)
    if brightness < 15:
        return DARKNESS
    
    face_presence = analyze_face_present(pil_image, cv_image)
    
    eye_openness = analyze_eye_area(pil_image, cv_image)
    
    looking_score = analyze_head_position(pil_image, cv_image)
    
    contrast = analyze_image_contrast(pil_image)
    
    measurement = {
        'brightness': brightness,
        'contrast': contrast,
        'face_presence': face_presence,
        'eye_openness': eye_openness,
        'looking_score': looking_score,
        'timestamp': time.time()
    }
    
    user_attention_data[user_id]['measurements'].append(measurement)
    measurements = user_attention_data[user_id]['measurements']
    
    if face_presence < 8:
        user_attention_data[user_id]['state_history'].append(ABSENT)
        return ABSENT
    
    weights = [0.5 + (0.5 * i / max(1, len(measurements) - 1)) for i in range(len(measurements))]
    total_weight = sum(weights)
    
    avg_eye_openness = sum(m.get('eye_openness', 0) * w for m, w in zip(measurements, weights)) / total_weight
    avg_face_presence = sum(m.get('face_presence', 0) * w for m, w in zip(measurements, weights)) / total_weight
    avg_looking_score = sum(m.get('looking_score', 0) * w for m, w in zip(measurements, weights)) / total_weight
    
    print(f"Debug - User {user_id}:")
    print(f"  Face presence: {avg_face_presence:.2f}")
    print(f"  Eye openness: {avg_eye_openness:.2f}")
    print(f"  Looking score: {avg_looking_score:.2f}")
    
    state = None
    
    if avg_eye_openness < 5:
        state = SLEEPING
    
    elif avg_eye_openness < 12:
        state = DROWSY
    
    elif avg_looking_score < 0.6:
        state = LOOKING_AWAY
    
    elif avg_face_presence > 20 and avg_eye_openness > 20 and avg_looking_score > 0.8:
        state = ATTENTIVE
    
    elif avg_face_presence > 15:
        state = ACTIVE
    
    else:
        state = ABSENT
    
    user_attention_data[user_id]['state_history'].append(state)
    
    recent_states = list(user_attention_data[user_id]['state_history'])[-5:]
    
    state_counts = {}
    for s in recent_states:
        state_counts[s] = state_counts.get(s, 0) + 1
    
    most_common_state = max(state_counts.items(), key=lambda x: x[1])[0]
    most_common_count = state_counts[most_common_state]
    
    confidence = get_attention_state_confidence(measurements, state, user_id)
    
    if most_common_count >= 3:
        return most_common_state
    elif confidence > 0.8:
        return state
    elif len(recent_states) > 0:
        return recent_states[-1]
    else:
        return state

def update_attention_history(user_id, attention_state):
    current_time = int(time.time() * 1000)
    
    if user_id not in user_attention_data:
        user_attention_data[user_id] = {
            "current_state": attention_state,
            "state_since": current_time,
            "history": []
        }
    else:
        if user_attention_data[user_id].get("current_state") != attention_state:
            prev_state = user_attention_data[user_id].get("current_state")
            prev_since = user_attention_data[user_id].get("state_since", current_time)
            
          
            duration = (current_time - prev_since) / 1000.0
            
            if duration > 1:
                if "history" not in user_attention_data[user_id]:
                    user_attention_data[user_id]["history"] = []
                
                user_attention_data[user_id]["history"].append({
                    "state": prev_state,
                    "start_time": prev_since,
                    "end_time": current_time,
                    "duration": duration
                })
            
            user_attention_data[user_id]["current_state"] = attention_state
            user_attention_data[user_id]["state_since"] = current_time
    
    if "history" in user_attention_data[user_id] and len(user_attention_data[user_id]["history"]) > 30:
        user_attention_data[user_id]["history"] = user_attention_data[user_id]["history"][-30:]
    
    return user_attention_data[user_id]

@app.route('/api/detect_attention', methods=['POST'])
def api_detect_attention():
    data = request.json
    
    if not data or 'image' not in data or 'userId' not in data:
        return jsonify({'error': 'Missing required data'}), 400
    
    try:
        with processing_lock:
            pil_image, cv_image = decode_base64_image(data['image'])
            user_id = data['userId']
            
            attention_state = detect_attention(pil_image, cv_image, user_id)
            
            user_data = update_attention_history(user_id, attention_state)
            
            total_time = 0
            attentive_time = 0
            
            if "history" in user_data:
                for entry in user_data["history"]:
                    duration = entry["duration"]
                    total_time += duration
                    if entry["state"] in [ATTENTIVE, ACTIVE]:
                        attentive_time += duration
            
            current_timestamp = int(time.time() * 1000)
            
            state_since = user_data.get("state_since", current_timestamp)
            current_duration = (current_timestamp - state_since) / 1000.0
            
            if current_duration < 0:
                current_duration = 0
                
            total_time += current_duration
            if user_data.get("current_state") in [ATTENTIVE, ACTIVE]:
                attentive_time += current_duration
            
            attention_percentage = (attentive_time / total_time * 100) if total_time > 0 else 0
            
            measurements = []
            if 'measurements' in user_attention_data.get(user_id, {}):
                measurements = list(user_attention_data[user_id]['measurements'])[-3:]
            
            confidence = get_attention_state_confidence(
                measurements, 
                attention_state, 
                user_id
            )
            
            attention_category = "attentive"
            if attention_state in [LOOKING_AWAY, DROWSY]:
                attention_category = "distracted"
            elif attention_state in [SLEEPING, ABSENT, DARKNESS]:
                attention_category = "inactive"
            
            return jsonify({
                'userId': user_id,
                'attentionState': attention_state,
                'attentionCategory': attention_category,
                'stateSince': user_data.get("state_since", current_timestamp),
                'attentionPercentage': attention_percentage,
                'confidence': round(confidence * 100, 1),
                'timestamp': current_timestamp
            })
    
    except Exception as e:
        import traceback
        print(f"Error in detect_attention: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/calibrate', methods=['POST'])
def api_calibrate():
    data = request.json
    
    if not data or 'image' not in data or 'userId' not in data:
        return jsonify({'error': 'Missing required data'}), 400
    
    try:
        pil_image, cv_image = decode_base64_image(data['image'])
        user_id = data['userId']
        
        success = calibrate_user(pil_image, cv_image, user_id)
        current_timestamp = int(time.time() * 1000)
        
        return jsonify({
            'userId': user_id,
            'calibrationSuccess': success,
            'timestamp': current_timestamp
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/room_attention', methods=['POST'])
def api_room_attention():
    data = request.json
    
    if not data or 'roomId' not in data or 'userIds' not in data:
        return jsonify({'error': 'Missing required data'}), 400
    
    room_id = data['roomId']
    user_ids = data['userIds']
    
    room_attention = {}
    current_timestamp = int(time.time() * 1000)
    
    for user_id in user_ids:
        if user_id in user_attention_data:
            user_data = user_attention_data[user_id]

            total_time = 0
            attentive_time = 0
            
            if "history" in user_data:
                for entry in user_data["history"]:
                    duration = entry["duration"]
                    total_time += duration
                    if entry["state"] in [ATTENTIVE, ACTIVE]:
                        attentive_time += duration
            
            state_since = user_data.get("state_since", current_timestamp)
            current_duration = (current_timestamp - state_since) / 1000.0
            
            if current_duration < 0:
                current_duration = 0
                
            total_time += current_duration
            if user_data.get("current_state") in [ATTENTIVE, ACTIVE]:
                attentive_time += current_duration
            
            attention_percentage = (attentive_time / total_time * 100) if total_time > 0 else 0
            
            measurements = []
            if 'measurements' in user_attention_data.get(user_id, {}):
                measurements = list(user_attention_data[user_id]['measurements'])[-5:]
            
            confidence = get_attention_state_confidence(
                measurements, 
                user_data.get("current_state", ABSENT), 
                user_id
            )
            
            current_state = user_data.get("current_state", ABSENT)

            attention_category = "attentive"
            if current_state in [LOOKING_AWAY, DROWSY]:
                attention_category = "distracted"
            elif current_state in [SLEEPING, ABSENT, DARKNESS]:
                attention_category = "inactive"
                
            room_attention[user_id] = {
                'attentionState': current_state,
                'attentionCategory': attention_category,
                'stateSince': user_data.get("state_since", current_timestamp),
                'attentionPercentage': attention_percentage,
                'confidence': round(confidence * 100, 1)
            }
        else:
            room_attention[user_id] = {
                'attentionState': ABSENT,
                'attentionCategory': 'inactive',
                'stateSince': current_timestamp,
                'attentionPercentage': 0,
                'confidence': 100
            }
    
    return jsonify({
        'roomId': room_id,
        'attention': room_attention,
        'timestamp': current_timestamp
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    current_timestamp = int(time.time() * 1000)
    return jsonify({
        'status': 'ok', 
        'timestamp': current_timestamp,
        'users_tracked': len(user_attention_data)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True) 