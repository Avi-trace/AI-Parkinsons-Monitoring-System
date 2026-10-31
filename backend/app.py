"""
NeuroTrack AI — Flask backend
Run:  python app.py
"""
import os
import sys
from pathlib import Path

from flask import Flask
from flask_cors import CORS

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from spiral_pd.routes.spiral import spiral_bp
from hand_extension.routes.tremor import tremor_bp
from Dashboard import dashboard_bp
from hand_tracking.routes.fingertap import fingertap_bp


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(
        app,
        origins="*",
        methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
        supports_credentials=False,
    )
    app.register_blueprint(spiral_bp)
    app.register_blueprint(tremor_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(fingertap_bp)
    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True, port=5000)