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
    """Extract eye landmarks from face mesh"""
    left_eye_indices = list(range(362, 374))
    right_eye_indices = list(range(33, 46))
    
    left_eye_landmarks = [face_landmarks.landmark[i] for i in left_eye_indices]
    right_eye_landmarks = [face_landmarks.landmark[i] for i in right_eye_indices]
    
    return left_eye_landmarks, right_eye_landmarks

def calculate_eye_aspect_ratio(eye_landmarks, image_shape):
    """Calculate Eye Aspect Ratio (EAR) for drowsiness detection"""
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
    """Detect head orientation (yaw, pitch, roll)"""
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
    """Analyze face presence and position"""
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
    """Analyze eye openness and symmetry"""
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
    """Analyze head position and orientation"""
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