import time
import threading
from collections import deque
from models import (
    ATTENTIVE, LOOKING_AWAY, ABSENT, DROWSY, SLEEPING, DARKNESS,
    UserAttentionData, UserCalibration
)
from analysis import (
    analyze_image_brightness, analyze_image_contrast,
    analyze_face_present, analyze_eye_area, analyze_head_position,
    analyze_drowsiness, detect_sleeping_state, detect_face_mesh_mediapipe
)
from utils import (
    get_user_attention_data, get_user_calibration, set_user_calibration,
    update_attention_history, get_attention_state_confidence
)

# Global processing lock for thread safety
processing_lock = threading.Lock()

def calibrate_user(pil_image, cv_image, user_id):
    """Calibrate user based on initial image"""
    calibration = get_user_calibration(user_id)
    if not calibration:
        face_presence = analyze_face_present(pil_image, cv_image)
        
        if face_presence > 20:
            calibration_data = {
                'brightness_baseline': analyze_image_brightness(pil_image),
                'contrast_baseline': analyze_image_contrast(pil_image),
                'time': time.time()
            }
            set_user_calibration(user_id, calibration_data)
            return True
    
    return False

def detect_attention(pil_image, cv_image, user_id):
    """Main attention detection function with enhanced detection"""
    user_data = get_user_attention_data(user_id)
    
    if user_id not in user_data:
        user_data[user_id] = {
            'measurements': deque(maxlen=5),
            'state_history': deque(maxlen=10),
            'calibration_images': [],
            'last_activity': time.time()
        }
        
        calibrate_user(pil_image, cv_image, user_id)
    
    brightness = analyze_image_brightness(pil_image)
    print(f"DEBUG - User {user_id} - Brightness: {brightness:.2f}")
    
    # Immediate darkness detection
    if brightness < 15:
        print(f"DEBUG - User {user_id} - Detected DARKNESS (brightness < 15)")
        return DARKNESS
    
    face_presence = analyze_face_present(pil_image, cv_image)
    eye_openness = analyze_eye_area(pil_image, cv_image)
    looking_score = analyze_head_position(pil_image, cv_image)
    drowsiness_score = analyze_drowsiness(pil_image, cv_image)
    contrast = analyze_image_contrast(pil_image)
    
    # Enhanced sleeping detection
    face_mesh_results = detect_face_mesh_mediapipe(cv_image)
    is_sleeping = False
    sleeping_score = 0.0
    
    if face_mesh_results.multi_face_landmarks:
        face_landmarks = face_mesh_results.multi_face_landmarks[0]
        is_sleeping, sleeping_score = detect_sleeping_state(face_landmarks, cv_image.shape, eye_openness)
    
    measurement = {
        'brightness': brightness,
        'contrast': contrast,
        'face_presence': face_presence,
        'eye_openness': eye_openness,
        'looking_score': looking_score,
        'drowsiness_score': drowsiness_score,
        'sleeping_score': sleeping_score,
        'timestamp': time.time()
    }
    
    user_data[user_id]['measurements'].append(measurement)
    
    print(f"DEBUG - User {user_id} - Current snapshot values:")
    print(f"  Face presence: {face_presence:.2f}")
    print(f"  Eye openness: {eye_openness:.2f}")
    print(f"  Looking score: {looking_score:.2f}")
    print(f"  Drowsiness score: {drowsiness_score:.2f}")
    print(f"  Sleeping score: {sleeping_score:.2f}")
    print(f"  Contrast: {contrast:.2f}")
    
    # ENHANCED STATE DETECTION with improved thresholds
    
    # 1. DARKNESS: Very low brightness
    if brightness < 15:
        print(f"DEBUG - User {user_id} - Detected DARKNESS (brightness < 15)")
        user_data[user_id]['state_history'].append(DARKNESS)
        return DARKNESS
    
    # 2. ABSENT: No face detected
    if face_presence < 8:
        print(f"DEBUG - User {user_id} - Detected ABSENT (face_presence < 8)")
        user_data[user_id]['state_history'].append(ABSENT)
        return ABSENT
    
    # 2. SLEEPING: Eyes completely closed
    if eye_openness < 5 or sleeping_score > 0.7:
        print(f"DEBUG - User {user_id} - Detected SLEEPING (eye_openness < 5 or sleeping_score > 0.7)")
        user_data[user_id]['state_history'].append(SLEEPING)
        return SLEEPING
    
    # 3. DROWSY: Eyes partially closed
    if eye_openness < 20 or drowsiness_score > 50:
        print(f"DEBUG - User {user_id} - Detected DROWSY (eye_openness < 20 or drowsiness_score > 50)")
        user_data[user_id]['state_history'].append(DROWSY)
        return DROWSY
    
    # 4. LOOKING_AWAY: Head tilted or turned
    if looking_score < 0.6:
        print(f"DEBUG - User {user_id} - Detected LOOKING_AWAY (looking_score < 0.6)")
        user_data[user_id]['state_history'].append(LOOKING_AWAY)
        return LOOKING_AWAY
    
    # Enhanced attentive detection - high standards for immediate response
    if (face_presence > 30 and eye_openness > 30 and looking_score > 0.8 and drowsiness_score < 30):
        print(f"DEBUG - User {user_id} - Detected ATTENTIVE (all high conditions met)")
        user_data[user_id]['state_history'].append(ATTENTIVE)
        return ATTENTIVE
    
    # Default to looking away if no clear state detected
    print(f"DEBUG - User {user_id} - Detected LOOKING_AWAY (default case)")
    user_data[user_id]['state_history'].append(LOOKING_AWAY)
    return LOOKING_AWAY

def process_attention_request(pil_image, cv_image, user_id):
    """Process attention detection request with thread safety"""
    with processing_lock:
        attention_state = detect_attention(pil_image, cv_image, user_id)
        user_data = update_attention_history(user_id, attention_state)
        
        # Calculate immediate attention percentage based on current state
        attention_percentage = 0
        
        if attention_state == ATTENTIVE:
            attention_percentage = 95  # High attention
        elif attention_state == LOOKING_AWAY:
            attention_percentage = 40  # Low attention
        elif attention_state == DROWSY:
            attention_percentage = 25  # Very low attention
        elif attention_state == SLEEPING:
            attention_percentage = 5   # Minimal attention (sleeping)
        elif attention_state == ABSENT:
            attention_percentage = 0   # No attention
        elif attention_state == DARKNESS:
            attention_percentage = 0   # No attention (darkness)
        
        current_timestamp = int(time.time() * 1000)
        
        measurements = []
        if 'measurements' in user_data:
            measurements = list(user_data['measurements'])[-3:]
        
        confidence = get_attention_state_confidence(
            measurements, 
            attention_state, 
            user_id
        )
        
        # Get current measurements for logging
        current_measurements = {}
        if 'measurements' in user_data and user_data['measurements']:
            latest_measurement = user_data['measurements'][-1]
            current_measurements = {
                'brightness': latest_measurement.get('brightness', 0),
                'contrast': latest_measurement.get('contrast', 0),
                'facePresence': latest_measurement.get('face_presence', 0),
                'eyeOpenness': latest_measurement.get('eye_openness', 0),
                'lookingScore': latest_measurement.get('looking_score', 0),
                'drowsinessScore': latest_measurement.get('drowsiness_score', 0),
                'sleepingScore': latest_measurement.get('sleeping_score', 0)
            }
        
        return {
            'userId': user_id,
            'attentionState': attention_state,
            'stateSince': user_data.get("state_since", current_timestamp),
            'attentionPercentage': attention_percentage,
            'confidence': round(confidence * 100, 1),
            'timestamp': current_timestamp,
            'measurements': current_measurements
        }

def get_room_attention_data(room_id, user_ids):
    """Get attention data for all users in a room"""
    room_attention = {}
    current_timestamp = int(time.time() * 1000)
    
    for user_id in user_ids:
        user_data = get_user_attention_data(user_id)
        
        if user_data and user_id in user_data:
            # Calculate immediate attention percentage based on current state
            current_state = user_data.get("current_state", ABSENT)
            attention_percentage = 0
            
            if current_state == ATTENTIVE:
                attention_percentage = 95  # High attention
            elif current_state == LOOKING_AWAY:
                attention_percentage = 40  # Low attention
            elif current_state == DROWSY:
                attention_percentage = 25  # Very low attention
            elif current_state == SLEEPING:
                attention_percentage = 5   # Minimal attention (sleeping)
            elif current_state == ABSENT:
                attention_percentage = 0   # No attention
            elif current_state == DARKNESS:
                attention_percentage = 0   # No attention (darkness)
            
            measurements = []
            if 'measurements' in user_data:
                measurements = list(user_data['measurements'])[-5:]
            
            confidence = get_attention_state_confidence(
                measurements, 
                current_state, 
                user_id
            )

            attention_category = "attentive"
            if current_state == SLEEPING:
                attention_category = "sleeping"
            elif current_state in [LOOKING_AWAY, DROWSY]:
                attention_category = "distracted"
            elif current_state in [ABSENT, DARKNESS]:
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
    
    return room_attention 