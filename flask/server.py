from flask import Flask, render_template, request, jsonify, send_file, url_for
import os
from werkzeug.utils import secure_filename
import sys
import json
import cv2

# Add the queens_retry directory to Python path
queens_retry_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(queens_retry_path)

# Import from queens_retry's app.py
from main_logic.queens import detect_grid, Game, CellState

app = Flask(__name__)
app_root = os.path.dirname(os.path.abspath(__file__))
app.config['UPLOAD_FOLDER'] = os.path.join(app_root, 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists with correct permissions
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.chmod(app.config['UPLOAD_FOLDER'], 0o755)

# Allow iPhone formats too
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'heic', 'heif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in request'}), 400
    
    file = request.files['file']
    print("DEBUG - Incoming file:", file.filename, "MIME:", file.mimetype)

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            file.save(filepath)
            print("DEBUG - File saved at:", filepath)
        except Exception as e:
            return jsonify({'error': f'Failed to save file: {e}', 'path': filepath}), 500

        try:
            # Process the image using the existing Queens puzzle logic
            grid_size, clean_grid, normalized_colors, clusters = detect_grid(filepath, debug=True)
            
            # Convert clusters to a serializable format
            serializable_clusters = [list(cluster) for cluster in clusters]
            
            # Save the clean grid image
            clean_grid_path = os.path.join(app.config['UPLOAD_FOLDER'], 'clean_grid.png')
            saved = cv2.imwrite(clean_grid_path, clean_grid)
            print("DEBUG - Saved clean grid:", saved, "->", clean_grid_path)
            
            return jsonify({
                'success': True,
                'grid_size': grid_size,
                'colors': normalized_colors,
                'clusters': serializable_clusters,
                'clean_grid': url_for('static', filename='uploads/clean_grid.png')
            })
        except Exception as e:
            print("ERROR - Processing failed:", str(e))
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/save_settings', methods=['POST'])
def save_settings():
    data = request.get_json()
    settings_file = os.path.join(os.path.dirname(__file__), 'settings.json')
    try:
        with open(settings_file, 'w') as f:
            json.dump(data, f)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/load_settings', methods=['GET'])
def load_settings():
    settings_file = os.path.join(os.path.dirname(__file__), 'settings.json')
    try:
        if os.path.exists(settings_file):
            with open(settings_file, 'r') as f:
                settings = json.load(f)
        else:
            settings = {'auto_x': False}
        return jsonify(settings)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # 👇 Keep it open to other devices on your network
    app.run(host='0.0.0.0', port=5000, debug=True)

