"""
NeuroTrack AI — Flask backend
Run:  python app.py
"""

from flask import Flask
from flask_cors import CORS

from hand_extension.routes.tremor import tremor_bp
from spiral_pd.routes.spiral import spiral_bp


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})  # tighten in production
    app.register_blueprint(tremor_bp)
    app.register_blueprint(spiral_bp)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, port=5000)