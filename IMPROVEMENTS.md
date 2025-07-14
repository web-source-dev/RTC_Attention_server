# Attention Detection System Improvements

## Overview

This document outlines the significant improvements made to the RTC Attention Detection System, focusing on enhanced head posture detection, looking away detection, drowsy detection, and sleeping detection.

## Key Improvements

### 1. Enhanced Head Posture Detection

#### Improved Landmark Detection
- **More Comprehensive Eye Landmarks**: Updated eye landmark indices to include 16 points per eye instead of the previous 12, providing better accuracy for eye openness detection.
- **Better Head Orientation Calculation**: Improved the calculation of yaw, pitch, and roll angles using more precise facial landmarks.

#### Enhanced Head Position Scoring
- **YAW Factor (Left-Right Rotation)**:
  - Looking straight: 1.0
  - Slight turn: 0.8
  - Moderate turn: 0.5
  - Significant turn: 0.2

- **PITCH Factor (Up-Down Rotation)**:
  - Looking straight: 1.0
  - Slight tilt: 0.7
  - Moderate tilt: 0.4
  - Significant tilt: 0.1

- **ROLL Factor (Head Tilt)**:
  - Straight head: 1.0
  - Slight tilt: 0.8
  - Moderate tilt: 0.5
  - Significant tilt: 0.2

### 2. Improved Looking Away Detection

#### Enhanced Thresholds
- **Primary Threshold**: `looking_score < 0.5`
- **Secondary Threshold**: `face_presence > 15 and looking_score < 0.7`
- **Weighted Combination**: 50% yaw, 30% pitch, 20% roll

#### Better Asymmetric Eye Detection
- Detects when eyes are asymmetric (indicating looking to the side)
- Penalizes eye openness score when asymmetry is detected
- Threshold: `eye_difference_ratio > 0.3`

### 3. Enhanced Drowsy Detection

#### Multi-Factor Analysis
The new drowsiness detection combines multiple factors:

1. **Eye Closure Factor (40% weight)**:
   - EAR < 0.15: 1.0 (fully drowsy)
   - EAR < 0.25: 0.7 (moderately drowsy)
   - EAR < 0.35: 0.3 (slightly drowsy)
   - EAR >= 0.35: 0.0 (alert)

2. **Head Dropping Factor (30% weight)**:
   - Pitch > 0.3: 1.0 (head dropped)
   - Pitch > 0.2: 0.6 (moderate drop)
   - Pitch > 0.1: 0.3 (slight drop)
   - Pitch <= 0.1: 0.0 (upright)

3. **Head Tilt Factor (20% weight)**:
   - Roll > 25°: 1.0 (significant tilt)
   - Roll > 15°: 0.5 (moderate tilt)
   - Roll <= 15°: 0.0 (straight)

4. **Eye Asymmetry Factor (10% weight)**:
   - Difference > 0.05: 1.0
   - Difference > 0.03: 0.5
   - Difference <= 0.03: 0.0

#### Improved Thresholds
- **Drowsy Detection**: `drowsiness_score > 60 or eye_openness < 10`
- **Enhanced Eye Openness Scoring**:
  - Very closed eyes (EAR < 0.15): `score = EAR * 40`
  - Partially closed (EAR < 0.25): `score = 6 + (EAR - 0.15) * 80`
  - Normal eyes (EAR < 0.35): `score = 14 + (EAR - 0.25) * 120`
  - Wide open eyes (EAR >= 0.35): `score = 26 + (EAR - 0.35) * 150`

### 4. New Sleeping Detection

#### Comprehensive Sleep Detection
The new sleeping detection system analyzes multiple factors:

1. **Eye Closure**: `eye_openness < 5` (very low eye openness)
2. **Head Dropping**: `pitch > 0.3` (head tilted forward)
3. **Head Tilt**: `abs(roll) > 30°` (significant head tilt)

#### Sleeping Score Calculation
- **Eyes Closed**: +0.4 (40% weight)
- **Head Dropping**: +0.3 (30% weight)
- **Head Tilted**: +0.2 (20% weight)
- **Sleeping Threshold**: `sleeping_score > 0.6`

### 5. Enhanced State Detection Logic

#### Improved State Classification
1. **ABSENT**: `face_presence < 8`
2. **SLEEPING**: `is_sleeping or sleeping_score > 0.7`
3. **DROWSY**: `drowsiness_score > 60 or eye_openness < 10`
4. **LOOKING_AWAY**: `looking_score < 0.5 or (face_presence > 15 and looking_score < 0.7)`
5. **ATTENTIVE**: `face_presence > 30 and eye_openness > 30 and looking_score > 0.8 and drowsiness_score < 30`
6. **ACTIVE**: `face_presence > 20 and eye_openness > 20 and looking_score > 0.6 and drowsiness_score < 50`

#### Attention Percentage Mapping
- **ATTENTIVE**: 95% (high attention)
- **ACTIVE**: 75% (moderate attention)
- **LOOKING_AWAY**: 40% (low attention)
- **DROWSY**: 25% (very low attention)
- **SLEEPING**: 5% (minimal attention)
- **ABSENT/DARKNESS**: 0% (no attention)

### 6. New Attention Categories

#### Enhanced Category System
- **ATTENTIVE**: Fully engaged users
- **DISTRACTED**: Users looking away or drowsy
- **SLEEPING**: Users who are sleeping
- **INACTIVE**: Absent users or dark environment

## Technical Improvements

### 1. Better Eye Aspect Ratio (EAR) Calculation
- Uses more landmarks for improved accuracy
- Calculates multiple vertical and horizontal distances
- Formula: `EAR = (v1 + v2) / (2.0 * h1)`

### 2. Enhanced Head Orientation Detection
- Uses precise facial landmarks for yaw, pitch, and roll calculation
- Normalized coordinates for better accuracy
- Improved angle calculations using mathematical functions

### 3. Improved Face Presence Detection
- Better confidence adjustment based on face position
- Enhanced size and aspect ratio calculations
- Improved center distance calculations

### 4. New Measurement Fields
Added new fields to track additional metrics:
- `drowsiness_score`: Comprehensive drowsiness measurement
- `sleeping_score`: Sleeping probability score

## Performance Improvements

### 1. Optimized Processing
- Reduced unnecessary calculations
- Better threshold management
- Improved state transition logic

### 2. Enhanced Debugging
- More detailed logging of individual metrics
- Better state transition tracking
- Comprehensive measurement logging

## Usage

### Running the Improved System
```bash
# Start the attention detection server
python app_new.py

# Run tests
python test_improvements.py
```

### API Response Format
The improved system returns enhanced response objects:
```json
{
  "userId": "user123",
  "attentionState": "attentive",
  "attentionCategory": "attentive",
  "stateSince": 1640995200000,
  "attentionPercentage": 95,
  "confidence": 85.5,
  "timestamp": 1640995200000,
  "measurements": {
    "brightness": 128.5,
    "contrast": 45.2,
    "facePresence": 85.3,
    "eyeOpenness": 75.8,
    "lookingScore": 0.92,
    "drowsinessScore": 15.2,
    "sleepingScore": 0.1
  }
}
```

## Testing

### Test Scenarios
The system includes comprehensive tests for:
1. **Normal Attention**: User looking straight ahead
2. **Looking Away**: User looking to the side
3. **Drowsy**: User with partially closed eyes
4. **Sleeping**: User with closed eyes and head tilt
5. **Dark Environment**: Very low brightness
6. **Absent**: No face detected

### Performance Testing
- Measures processing time per frame
- Calculates FPS (frames per second)
- Validates threshold accuracy

## Future Enhancements

### Planned Improvements
1. **Temporal Analysis**: Track state changes over time
2. **Machine Learning**: Implement ML-based detection
3. **Calibration**: User-specific calibration system
4. **Real-time Adaptation**: Dynamic threshold adjustment

### Potential Additions
1. **Blink Detection**: Track blink frequency
2. **Micro-expressions**: Detect subtle facial expressions
3. **Gaze Tracking**: More precise eye movement detection
4. **Posture Analysis**: Full body posture detection

## Conclusion

These improvements significantly enhance the accuracy and reliability of the attention detection system, providing better detection of head posture, looking away behavior, drowsiness, and sleeping states. The system now offers more nuanced state classification and improved performance metrics. 