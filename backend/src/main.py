import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, send_from_directory
from flask_cors import CORS # Import CORS

# Remove unused user model and blueprint if not needed for this app
# from src.models.user import db 
# from src.routes.user import user_bp

from src.routes.predict import predict_bp # Import the new predict blueprint

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist")) # Point to frontend build
CORS(app) # Enable CORS for all routes

app.config["SECRET_KEY"] = "your_very_secret_key_emotion_analyzer"

# Register the prediction blueprint
app.register_blueprint(predict_bp)

# Database setup is commented out as it is not used for this application
# app.config["SQLALCHEMY_DATABASE_URI"] = f\"mysql+pymysql://{os.getenv("DB_USERNAME", "root")}:{os.getenv("DB_PASSWORD", "password")}@{os.getenv("DB_HOST", "localhost")}:{os.getenv("DB_PORT", "3306")}/{os.getenv("DB_NAME", "mydb")}\"
# app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
# db.init_app(app)
# with app.app_context():
# db.create_all()

# Serve React App
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve(path):
    static_folder_path = app.static_folder
    if static_folder_path is None:
        # This case should ideally not happen if static_folder is configured correctly
        return "Static folder not configured", 404

    if path != "" and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    else:
        index_path = os.path.join(static_folder_path, "index.html")
        if os.path.exists(index_path):
            return send_from_directory(static_folder_path, "index.html")
        else:
            # Fallback if index.html is not found - useful for API-only or during development
            return jsonify({"message": "Welcome to the Emotion Analyzer API. Frontend not found or not built."}), 200

if __name__ == "__main__":
    # Make sure to run on 0.0.0.0 to be accessible externally if needed
    app.run(host="0.0.0.0", port=5001, debug=True) # Changed port to 5001 to avoid potential conflicts

