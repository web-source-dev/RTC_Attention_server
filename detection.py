import time
import threading
from collections import deque
from models import (
    ATTENTIVE, LOOKING_AWAY, ABSENT, ACTIVE, DROWSY, DARKNESS,
    UserAttentionData, UserCalibration
)
from analysis import (
    analyze_image_brightness, analyze_image_contrast,
    analyze_face_present, analyze_eye_area, analyze_head_position
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
    """Main attention detection function"""
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
    contrast = analyze_image_contrast(pil_image)
    
    measurement = {
        'brightness': brightness,
        'contrast': contrast,
        'face_presence': face_presence,
        'eye_openness': eye_openness,
        'looking_score': looking_score,
        'timestamp': time.time()
    }
    
    user_data[user_id]['measurements'].append(measurement)
    
    print(f"DEBUG - User {user_id} - Current snapshot values:")
    print(f"  Face presence: {face_presence:.2f}")
    print(f"  Eye openness: {eye_openness:.2f}")
    print(f"  Looking score: {looking_score:.2f}")
    print(f"  Contrast: {contrast:.2f}")
    
    # IMMEDIATE STATE DETECTION - No weighted averages
    # Use current snapshot values directly for immediate response
    
    # Immediate absence detection
    if face_presence < 8:
        print(f"DEBUG - User {user_id} - Detected ABSENT (face_presence < 8)")
        user_data[user_id]['state_history'].append(ABSENT)
        return ABSENT
    

    
    # Immediate drowsy detection
    if eye_openness < 12:
        print(f"DEBUG - User {user_id} - Detected DROWSY (eye_openness < 12)")
        user_data[user_id]['state_history'].append(DROWSY)
        return DROWSY
    
    # Immediate looking away detection
    if looking_score < 0.6:
        print(f"DEBUG - User {user_id} - Detected LOOKING_AWAY (looking_score < 0.6)")
        user_data[user_id]['state_history'].append(LOOKING_AWAY)
        return LOOKING_AWAY
    
    # Immediate attentive detection - high standards for immediate response
    if face_presence > 25 and eye_openness > 25 and looking_score > 0.85:
        print(f"DEBUG - User {user_id} - Detected ATTENTIVE (all high conditions met)")
        user_data[user_id]['state_history'].append(ATTENTIVE)
        return ATTENTIVE
    
    # Immediate active detection - moderate standards
    if face_presence > 15 and eye_openness > 15 and looking_score > 0.7:
        print(f"DEBUG - User {user_id} - Detected ACTIVE (moderate conditions met)")
        user_data[user_id]['state_history'].append(ACTIVE)
        return ACTIVE
    
    # Default to absent if no clear state detected
    print(f"DEBUG - User {user_id} - Detected ABSENT (default case)")
    user_data[user_id]['state_history'].append(ABSENT)
    return ABSENT

def process_attention_request(pil_image, cv_image, user_id):
    """Process attention detection request with thread safety"""
    with processing_lock:
        attention_state = detect_attention(pil_image, cv_image, user_id)
        user_data = update_attention_history(user_id, attention_state)
        
        # Calculate immediate attention percentage based on current state
        attention_percentage = 0
        
        if attention_state == ATTENTIVE:
            attention_percentage = 95  # High attention
        elif attention_state == ACTIVE:
            attention_percentage = 75  # Moderate attention
        elif attention_state == LOOKING_AWAY:
            attention_percentage = 40  # Low attention
        elif attention_state == DROWSY:
            attention_percentage = 25  # Very low attention

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
        
        return {
            'userId': user_id,
            'attentionState': attention_state,
            'stateSince': user_data.get("state_since", current_timestamp),
            'attentionPercentage': attention_percentage,
            'confidence': round(confidence * 100, 1),
            'timestamp': current_timestamp
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
            elif current_state == ACTIVE:
                attention_percentage = 75  # Moderate attention
            elif current_state == LOOKING_AWAY:
                attention_percentage = 40  # Low attention
            elif current_state == DROWSY:
                attention_percentage = 25  # Very low attention

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
            if current_state in [LOOKING_AWAY, DROWSY]:
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