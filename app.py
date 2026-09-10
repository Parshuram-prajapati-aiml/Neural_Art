import os
import sys

# Add NST_Code to Python path
NST_CODE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'NST_Code')
if NST_CODE_DIR not in sys.path:
    sys.path.insert(0, NST_CODE_DIR)

from NST_Code.app import app

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)

