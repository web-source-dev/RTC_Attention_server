import base64
import io
import time
import gc
from PIL import Image
import numpy as np
import cv2
from models import Measurement, UserAttentionData, UserCalibration, ABSENT

# Memory management settings
MAX_USERS = 1000
MAX_HISTORY_ENTRIES = 20
CLEANUP_INTERVAL = 300
last_cleanup_time = time.time()

# Global data storage
user_attention_data = {}
user_calibration = {}

def decode_base64_image(base64_string):
    """Decode base64 image string to PIL and OpenCV formats"""
    if "base64," in base64_string:
        base64_string = base64_string.split("base64,")[1]
    
    image_bytes = base64.b64decode(base64_string)
    pil_image = Image.open(io.BytesIO(image_bytes))
    
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return pil_image, cv_image

def cleanup_old_data():
    """Clean up old user data to prevent memory leaks"""
    global user_attention_data, user_calibration, last_cleanup_time
    
    current_time = time.time()
    if current_time - last_cleanup_time < CLEANUP_INTERVAL:
        return
    
    last_cleanup_time = current_time
    
    # Remove users who haven't been active for more than 10 minutes
    cutoff_time = current_time - 600  # 10 minutes
    users_to_remove = []
    
    for user_id, user_data in user_attention_data.items():
        last_activity = user_data.get("last_activity", 0)
        if last_activity < cutoff_time:
            users_to_remove.append(user_id)
    
    # Remove old users
    for user_id in users_to_remove:
        del user_attention_data[user_id]
        if user_id in user_calibration:
            del user_calibration[user_id]
    
    # If still too many users, remove oldest ones
    if len(user_attention_data) > MAX_USERS:
        sorted_users = sorted(
            user_attention_data.items(),
            key=lambda x: x[1].get("last_activity", 0)
        )
        users_to_remove = [user_id for user_id, _ in sorted_users[:-MAX_USERS]]
        
        for user_id in users_to_remove:
            del user_attention_data[user_id]
            if user_id in user_calibration:
                del user_calibration[user_id]
    
    # Limit history entries for all users
    for user_id in user_attention_data:
        if "history" in user_attention_data[user_id] and len(user_attention_data[user_id]["history"]) > MAX_HISTORY_ENTRIES:
            user_attention_data[user_id]["history"] = user_attention_data[user_id]["history"][-MAX_HISTORY_ENTRIES:]
        
        # Limit measurements history
        if "measurements" in user_attention_data[user_id]:
            measurements = user_attention_data[user_id]["measurements"]
            if hasattr(measurements, '__len__') and len(measurements) > 10:
                user_attention_data[user_id]["measurements"] = list(measurements)[-10:]
    
    # Force garbage collection
    gc.collect()
    
    if users_to_remove:
        print(f"Cleaned up {len(users_to_remove)} inactive users. Current users: {len(user_attention_data)}")

def get_attention_state_confidence(measurements, current_state, user_id):
    """Calculate confidence score for attention state detection"""
    if len(measurements) < 1:
        return 0.7  # Default confidence for first measurement
    
    # Get the most recent measurement for immediate confidence
    latest_measurement = measurements[-1]
    
    # Calculate confidence based on current snapshot quality
    face_presence = latest_measurement.get('face_presence', 0)
    eye_openness = latest_measurement.get('eye_openness', 0)
    looking_score = latest_measurement.get('looking_score', 0)
    brightness = latest_measurement.get('brightness', 0)
    
    # Base confidence
    confidence = 0.5
    
    # Boost confidence for clear states
    if current_state == "darkness" and brightness < 15:
        confidence = 0.95  # Very confident for darkness
    elif current_state == "absent" and face_presence < 8:
        confidence = 0.9   # Very confident for absence

    elif current_state == "drowsy" and eye_openness < 12:
        confidence = 0.85  # Confident for drowsy
    elif current_state == "looking_away" and looking_score < 0.6:
        confidence = 0.8   # Confident for looking away
    elif current_state == "attentive" and face_presence > 25 and eye_openness > 25 and looking_score > 0.85:
        confidence = 0.9   # Very confident for attentive
    elif current_state == "active" and face_presence > 15 and eye_openness > 15 and looking_score > 0.7:
        confidence = 0.85  # Confident for active
    
    # Adjust confidence based on measurement quality
    if face_presence > 30:
        confidence += 0.1  # High face presence boosts confidence
    if eye_openness > 30:
        confidence += 0.1  # High eye openness boosts confidence
    if looking_score > 0.9:
        confidence += 0.1  # High looking score boosts confidence
    
    # Reduce confidence for edge cases
    if face_presence < 15 and current_state not in ["absent", "darkness"]:
        confidence -= 0.2  # Low face presence reduces confidence
    if eye_openness < 10 and current_state not in ["drowsy", "absent", "darkness"]:
        confidence -= 0.2  # Low eye openness reduces confidence
    
    return min(1.0, max(0.3, confidence))  # Clamp between 0.3 and 1.0

def update_attention_history(user_id, attention_state):
    """Update user attention history with new state"""
    current_time = int(time.time() * 1000)
    
    # Clean up old data periodically
    cleanup_old_data()
    
    if user_id not in user_attention_data:
        user_attention_data[user_id] = {
            "current_state": attention_state,
            "state_since": current_time,
            "history": [],
            "last_activity": time.time()
        }
    else:
        # Update last activity
        user_attention_data[user_id]["last_activity"] = time.time()
        
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
    
    # Limit history entries using the new constant
    if "history" in user_attention_data[user_id] and len(user_attention_data[user_id]["history"]) > MAX_HISTORY_ENTRIES:
        user_attention_data[user_id]["history"] = user_attention_data[user_id]["history"][-MAX_HISTORY_ENTRIES:]
    
    return user_attention_data[user_id]

def get_attention_percentage(attention_state):
    """Convert attention state to percentage score"""
    if attention_state == "attentive":
        return 95  # High attention
    elif attention_state == "active":
        return 75  # Moderate attention
    elif attention_state == "looking_away":
        return 40  # Low attention
    elif attention_state == "drowsy":
        return 25  # Very low attention

    elif attention_state == "absent":
        return 0   # No attention
    elif attention_state == "darkness":
        return 0   # No attention (darkness)
    else:
        return 0

def get_user_attention_data(user_id):
    """Get or create user attention data"""
    if user_id not in user_attention_data:
        user_attention_data[user_id] = {
            'measurements': [],
            'state_history': [],
            'calibration_images': [],
            'last_activity': time.time()
        }
    return user_attention_data[user_id]

def get_user_calibration(user_id):
    """Get user calibration data"""
    return user_calibration.get(user_id)

def set_user_calibration(user_id, calibration_data):
    """Set user calibration data"""
    user_calibration[user_id] = calibration_data 