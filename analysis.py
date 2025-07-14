import math
import numpy as np
import cv2
from PIL import Image, ImageStat
import mediapipe as mp

# MediaPipe initialization
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
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

def analyze_image_brightness(image):
    """Analyze image brightness using PIL"""
    gray_image = image.convert('L')
    stat = ImageStat.Stat(gray_image)
    brightness = stat.mean[0]
    
    return brightness

def analyze_image_contrast(image):
    """Analyze image contrast using PIL"""
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
    """Detect face using MediaPipe face detection"""
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
    """Detect face mesh using MediaPipe"""
    image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    
    return results

def detect_pose_mediapipe(cv_image):
    """Detect pose using MediaPipe"""
    image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    results = pose_detection.process(image_rgb)
    
    return results

def get_eye_landmarks(face_landmarks, face_oval_indices):
    """Extract eye landmarks from face mesh with improved accuracy"""
    # More comprehensive eye landmark indices for better detection
    left_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    right_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    
    left_eye_landmarks = [face_landmarks.landmark[i] for i in left_eye_indices]
    right_eye_landmarks = [face_landmarks.landmark[i] for i in right_eye_indices]
    
    return left_eye_landmarks, right_eye_landmarks

def calculate_eye_aspect_ratio(eye_landmarks, image_shape):
    """Calculate Eye Aspect Ratio (EAR) for drowsiness detection with improved accuracy"""
    if not eye_landmarks or len(eye_landmarks) < 6:
        return 0.0
    
    h, w = image_shape[0:2]
    landmarks_px = [(int(point.x * w), int(point.y * h)) for point in eye_landmarks]
    
    # Use more landmarks for better EAR calculation
    # Vertical distances (top to bottom)
    v1 = math.dist(landmarks_px[1], landmarks_px[5])  # Top to bottom
    v2 = math.dist(landmarks_px[2], landmarks_px[4])  # Top to bottom
    
    # Horizontal distance (left to right)
    h1 = math.dist(landmarks_px[0], landmarks_px[3])  # Left to right
    
    # Calculate EAR
    ear = (v1 + v2) / (2.0 * h1 + 1e-6)
    
    return ear

def detect_head_orientation(face_landmarks, image_shape):
    """Detect head orientation (yaw, pitch, roll) with improved accuracy"""
    if not face_landmarks:
        return 0.0, 0.0, 0.0
    
    h, w = image_shape[0:2]
    
    # Key facial landmarks for head orientation
    nose_tip = face_landmarks.landmark[4]
    left_eye_outer = face_landmarks.landmark[33]
    right_eye_outer = face_landmarks.landmark[263]
    left_ear = face_landmarks.landmark[234]
    right_ear = face_landmarks.landmark[454]
    forehead = face_landmarks.landmark[10]
    chin = face_landmarks.landmark[152]
    
    # Convert to pixel coordinates
    nose_tip_px = (int(nose_tip.x * w), int(nose_tip.y * h))
    left_eye_outer_px = (int(left_eye_outer.x * w), int(left_eye_outer.y * h))
    right_eye_outer_px = (int(right_eye_outer.x * w), int(right_eye_outer.y * h))
    left_ear_px = (int(left_ear.x * w), int(left_ear.y * h))
    right_ear_px = (int(right_ear.x * w), int(right_ear.y * h))
    forehead_px = (int(forehead.x * w), int(forehead.y * h))
    chin_px = (int(chin.x * w), int(chin.y * h))
    
    # Calculate YAW (left-right rotation)
    face_center_x = (left_ear_px[0] + right_ear_px[0]) / 2
    image_center_x = w / 2
    yaw = (face_center_x - image_center_x) / (w / 2)  # Normalized to [-1, 1]
    
    # Calculate PITCH (up-down rotation)
    face_center_y = (forehead_px[1] + chin_px[1]) / 2
    image_center_y = h / 2
    pitch = (face_center_y - image_center_y) / (h / 2)  # Normalized to [-1, 1]
    
    # Calculate ROLL (tilt)
    eye_line_angle = math.atan2(right_eye_outer_px[1] - left_eye_outer_px[1], 
                                right_eye_outer_px[0] - left_eye_outer_px[0])
    roll = math.degrees(eye_line_angle)
    
    return yaw, pitch, roll

def detect_sleeping_state(face_landmarks, image_shape, eye_openness):
    """Detect if user is sleeping based on eye closure and head position"""
    if not face_landmarks:
        return False, 0.0
    
    h, w = image_shape[0:2]
    
    # Get head orientation
    yaw, pitch, roll = detect_head_orientation(face_landmarks, image_shape)
    
    # Check for head dropping (forward tilt)
    head_dropping = pitch > 0.3  # Head tilted forward
    
    # Check for prolonged eye closure
    eyes_closed = eye_openness < 5  # Very low eye openness
    
    # Check for head tilt (common when sleeping)
    head_tilted = abs(roll) > 30  # Significant head tilt
    
    # Calculate sleeping probability
    sleeping_score = 0.0
    
    if eyes_closed:
        sleeping_score += 0.4
    
    if head_dropping:
        sleeping_score += 0.3
    
    if head_tilted:
        sleeping_score += 0.2
    
    # Additional check for stillness (would need temporal data)
    # For now, we'll use a combination of factors
    
    is_sleeping = sleeping_score > 0.6
    
    return is_sleeping, sleeping_score

def analyze_face_present(pil_image, cv_image):
    """Analyze face presence and position with improved detection for looking away scenarios"""
    face_bbox, confidence = detect_face_mediapipe(cv_image)
    
    # Also check face mesh detection as a backup
    face_mesh_results = detect_face_mesh_mediapipe(cv_image)
    mesh_confidence = 0.0
    
    if face_mesh_results.multi_face_landmarks:
        mesh_confidence = 0.8  # High confidence if face mesh is detected
    
    # Use the higher confidence between the two methods
    if face_bbox is None and mesh_confidence == 0.0:
        return 0
    
    # If face mesh is detected but bbox is not, still consider face present
    if face_bbox is None and mesh_confidence > 0:
        # Create a synthetic bbox based on face mesh landmarks
        face_landmarks = face_mesh_results.multi_face_landmarks[0]
        h, w, _ = cv_image.shape
        
        # Get bounding box from face mesh landmarks
        x_coords = [int(landmark.x * w) for landmark in face_landmarks.landmark]
        y_coords = [int(landmark.y * h) for landmark in face_landmarks.landmark]
        
        xmin, xmax = min(x_coords), max(x_coords)
        ymin, ymax = min(y_coords), max(y_coords)
        
        face_bbox = {
            'xmin': xmin,
            'ymin': ymin,
            'width': xmax - xmin,
            'height': ymax - ymin
        }
        confidence = mesh_confidence
    
    # Ensure face_bbox is not None before proceeding
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
    print(f"  Face detection confidence: {confidence:.2f}")
    
    adjusted_confidence = confidence
    
    # More tolerant face presence scoring for looking away scenarios
    if center_distance > 0.8:  # Increased from 0.7
        adjusted_confidence *= (1 - (center_distance - 0.8) / 0.2)  # More gradual reduction
    
    if face_size_ratio < 0.03:  # Reduced from 0.05
        adjusted_confidence *= (face_size_ratio / 0.03)
    
    if face_aspect_ratio < 0.6:  # Reduced from 0.7
        adjusted_confidence *= (face_aspect_ratio / 0.6)
    
    # Boost confidence if face mesh is also detected
    if mesh_confidence > 0:
        adjusted_confidence = max(adjusted_confidence, mesh_confidence * 0.8)
    
    return adjusted_confidence * 100

def analyze_eye_area(pil_image, cv_image):
    """Analyze eye openness and symmetry with improved drowsiness detection"""
    face_mesh_results = detect_face_mesh_mediapipe(cv_image)
    
    if not face_mesh_results.multi_face_landmarks:
        return 0
    
    face_landmarks = face_mesh_results.multi_face_landmarks[0]
    
    left_eye_landmarks, right_eye_landmarks = get_eye_landmarks(face_landmarks, [])
    
    left_ear = calculate_eye_aspect_ratio(left_eye_landmarks, cv_image.shape)
    right_ear = calculate_eye_aspect_ratio(right_eye_landmarks, cv_image.shape)
    
    eye_difference = abs(left_ear - right_ear)
    eye_difference_ratio = eye_difference / max(max(left_ear, right_ear), 0.01)

    avg_ear = (left_ear + right_ear) / 2
    
    # Improved eye openness scoring with better thresholds
    if avg_ear < 0.15:  # Completely closed eyes (sleeping)
        openness_score = avg_ear * 30
    elif avg_ear < 0.25:  # Partially closed eyes (drowsy)
        openness_score = 4 + ((avg_ear - 0.15) * 60)
    elif avg_ear < 0.35:  # Normal eyes
        openness_score = 10 + ((avg_ear - 0.25) * 100)
    else:  # Wide open eyes
        openness_score = 20 + ((avg_ear - 0.35) * 150)
    
    openness_score = min(100, max(0, openness_score))
    
    # Penalize asymmetric eyes (looking to the side)
    if eye_difference_ratio > 0.3:
        print("  Detected asymmetric eyes - possibly looking to the side")
        openness_score = max(0, openness_score * 0.8)
    
    print(f"  Left EAR: {left_ear:.3f}, Right EAR: {right_ear:.3f}, Avg: {avg_ear:.3f}")
    print(f"  Eye difference ratio: {eye_difference_ratio:.3f}")
    print(f"  Eye openness score: {openness_score:.1f}")
    
    return openness_score

def analyze_head_position(pil_image, cv_image):
    """Analyze head position and orientation with improved looking away detection"""
    face_mesh_results = detect_face_mesh_mediapipe(cv_image)
    
    if not face_mesh_results.multi_face_landmarks:
        return 0.0
    
    face_landmarks = face_mesh_results.multi_face_landmarks[0]
    
    yaw, pitch, roll = detect_head_orientation(face_landmarks, cv_image.shape)
    
    # Improved head position scoring with better thresholds for looking away detection
    # YAW factor (left-right rotation) - more sensitive to looking away
    yaw_abs = abs(yaw)
    if yaw_abs < 0.15:  # Looking straight (reduced from 0.2)
        yaw_factor = 1.0
    elif yaw_abs < 0.3:  # Slight turn (reduced from 0.4)
        yaw_factor = 0.7
    elif yaw_abs < 0.5:  # Moderate turn (reduced from 0.6)
        yaw_factor = 0.4
    else:  # Significant turn
        yaw_factor = 0.1
    
    # PITCH factor (up-down rotation) - more sensitive
    pitch_abs = abs(pitch)
    if pitch_abs < 0.15:  # Looking straight (reduced from 0.2)
        pitch_factor = 1.0
    elif pitch_abs < 0.3:  # Slight tilt (reduced from 0.4)
        pitch_factor = 0.6
    elif pitch_abs < 0.5:  # Moderate tilt (reduced from 0.6)
        pitch_factor = 0.3
    else:  # Significant tilt
        pitch_factor = 0.1
    
    # ROLL factor (head tilt) - more sensitive
    roll_abs = abs(roll)
    if roll_abs < 10:  # Straight head (reduced from 15)
        roll_factor = 1.0
    elif roll_abs < 20:  # Slight tilt (reduced from 30)
        roll_factor = 0.7
    elif roll_abs < 35:  # Moderate tilt (reduced from 45)
        roll_factor = 0.4
    else:  # Significant tilt
        roll_factor = 0.1
    
    # Weighted combination with emphasis on yaw (looking left/right)
    looking_score = (yaw_factor * 0.6) + (pitch_factor * 0.25) + (roll_factor * 0.15)
    
    # Additional penalty for extreme head positions
    if yaw_abs > 0.6 or pitch_abs > 0.6 or roll_abs > 45:
        looking_score *= 0.5  # Significant penalty for extreme positions
    
    print(f"  Head position - yaw: {yaw:.2f}, pitch: {pitch:.2f}, roll: {roll:.2f}")
    print(f"  Looking score: {looking_score:.2f}")
    print(f"  Yaw factor: {yaw_factor:.2f}, Pitch factor: {pitch_factor:.2f}, Roll factor: {roll_factor:.2f}")
    
    return looking_score

def analyze_drowsiness(pil_image, cv_image):
    """Enhanced drowsiness detection combining multiple factors"""
    face_mesh_results = detect_face_mesh_mediapipe(cv_image)
    
    if not face_mesh_results.multi_face_landmarks:
        return 0.0
    
    face_landmarks = face_mesh_results.multi_face_landmarks[0]
    
    # Get eye openness
    left_eye_landmarks, right_eye_landmarks = get_eye_landmarks(face_landmarks, [])
    left_ear = calculate_eye_aspect_ratio(left_eye_landmarks, cv_image.shape)
    right_ear = calculate_eye_aspect_ratio(right_eye_landmarks, cv_image.shape)
    avg_ear = (left_ear + right_ear) / 2
    
    # Get head orientation
    yaw, pitch, roll = detect_head_orientation(face_landmarks, cv_image.shape)
    
    # Calculate drowsiness score based on multiple factors
    drowsiness_score = 0.0
    
    # Eye closure factor (40% weight)
    if avg_ear < 0.15:
        eye_factor = 1.0
    elif avg_ear < 0.25:
        eye_factor = 0.7
    elif avg_ear < 0.35:
        eye_factor = 0.3
    else:
        eye_factor = 0.0
    
    drowsiness_score += eye_factor * 0.4
    
    # Head dropping factor (30% weight)
    if pitch > 0.3:  # Head tilted forward
        head_factor = 1.0
    elif pitch > 0.2:
        head_factor = 0.6
    elif pitch > 0.1:
        head_factor = 0.3
    else:
        head_factor = 0.0
    
    drowsiness_score += head_factor * 0.3
    
    # Head tilt factor (20% weight)
    if abs(roll) > 25:
        tilt_factor = 1.0
    elif abs(roll) > 15:
        tilt_factor = 0.5
    else:
        tilt_factor = 0.0
    
    drowsiness_score += tilt_factor * 0.2
    
    # Eye asymmetry factor (10% weight)
    eye_difference = abs(left_ear - right_ear)
    if eye_difference > 0.05:
        asymmetry_factor = 1.0
    elif eye_difference > 0.03:
        asymmetry_factor = 0.5
    else:
        asymmetry_factor = 0.0
    
    drowsiness_score += asymmetry_factor * 0.1
    
    print(f"  Drowsiness analysis - EAR: {avg_ear:.3f}, pitch: {pitch:.2f}, roll: {roll:.2f}")
    print(f"  Drowsiness score: {drowsiness_score:.2f}")
    
    return drowsiness_score * 100 