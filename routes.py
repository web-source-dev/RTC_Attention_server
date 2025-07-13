import os
import time
import sys
import gc
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. Memory monitoring will be limited.")

from utils import decode_base64_image, get_user_attention_data, get_user_calibration, set_user_calibration
from detection import process_attention_request, calibrate_user, get_room_attention_data
from models import AttentionResponse, RoomAttentionResponse, CalibrationResponse

app = Flask(__name__)
CORS(app)

@app.route('/api/detect_attention', methods=['POST'])
def api_detect_attention():
    """Main attention detection endpoint"""
    data = request.json
    
    if not data or 'image' not in data or 'userId' not in data:
        return jsonify({'error': 'Missing required data'}), 400
    
    try:
        pil_image, cv_image = decode_base64_image(data['image'])
        user_id = data['userId']
        
        result = process_attention_request(pil_image, cv_image, user_id)
        
        # Add attention category
        attention_category = "attentive"
        if result['attentionState'] in ["looking_away", "drowsy"]:
            attention_category = "distracted"
        elif result['attentionState'] in ["absent", "darkness"]:
            attention_category = "inactive"
        
        result['attentionCategory'] = attention_category
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in detect_attention: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/calibrate', methods=['POST'])
def api_calibrate():
    """User calibration endpoint"""
    data = request.json
    
    if not data or 'image' not in data or 'userId' not in data:
        return jsonify({'error': 'Missing required data'}), 400
    
    try:
        pil_image, cv_image = decode_base64_image(data['image'])
        user_id = data['userId']
        
        success = calibrate_user(pil_image, cv_image, user_id)
        current_timestamp = int(time.time() * 1000)
        
        response = CalibrationResponse(user_id, success, current_timestamp)
        return jsonify(response.to_dict())
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/room_attention', methods=['POST'])
def api_room_attention():
    """Room attention endpoint for multiple users"""
    data = request.json
    
    if not data or 'roomId' not in data or 'userIds' not in data:
        return jsonify({'error': 'Missing required data'}), 400
    
    room_id = data['roomId']
    user_ids = data['userIds']
    
    try:
        room_attention = get_room_attention_data(room_id, user_ids)
        current_timestamp = int(time.time() * 1000)
        
        response = RoomAttentionResponse(room_id, room_attention, current_timestamp)
        return jsonify(response.to_dict())
    
    except Exception as e:
        print(f"Error in room_attention: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/test', methods=['GET'])
def test_endpoint():
    """Test endpoint to verify server is running"""
    from utils import user_attention_data
    
    return jsonify({
        'status': 'ok',
        'message': 'Attention server is running',
        'timestamp': int(time.time() * 1000),
        'user_count': len(user_attention_data)
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint with memory monitoring"""
    current_timestamp = int(time.time() * 1000)
    
    # Get memory usage
    if PSUTIL_AVAILABLE:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # Force garbage collection
        gc.collect()
        
        # Get memory after GC
        memory_after_gc = process.memory_info().rss / 1024 / 1024
    else:
        memory_mb = 0
        memory_after_gc = 0
        # Force garbage collection even without psutil
        gc.collect()
    
    from utils import user_attention_data, user_calibration, last_cleanup_time
    
    return jsonify({
        'status': 'ok', 
        'timestamp': current_timestamp,
        'users_tracked': len(user_attention_data),
        'memory_usage_mb': round(memory_mb, 2) if PSUTIL_AVAILABLE else 'psutil_not_available',
        'memory_after_gc_mb': round(memory_after_gc, 2) if PSUTIL_AVAILABLE else 'psutil_not_available',
        'calibration_users': len(user_calibration),
        'last_cleanup': last_cleanup_time,
        'psutil_available': PSUTIL_AVAILABLE
    })

def create_app():
    """Factory function to create Flask app"""
    return app

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("Using Python version:", sys.version)
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True) 