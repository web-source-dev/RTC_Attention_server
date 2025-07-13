import os
import sys
import time
from routes import create_app

def main():
    """Main application entry point"""
    print("Starting Attention Detection Server...")
    print("Using Python version:", sys.version)
    
    app = create_app()
    
    port = int(os.environ.get('PORT', 5000))
    print(f"Server starting on port {port}")
    
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)

if __name__ == '__main__':
    main() 