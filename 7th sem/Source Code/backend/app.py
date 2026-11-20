import os, sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from models.db_models import db, bcrypt
from routes.predict_routes import predict_bp
from routes.info_routes import info_bp
from routes.auth_routes import auth_bp
from routes.history_routes import history_bp


def create_app():
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    app = Flask(__name__, static_folder=os.path.join(BASE_DIR, '..', 'frontend'), static_url_path='')

    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret')
    app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'jwt-secret')
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/medpredict.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    db.init_app(app)
    bcrypt.init_app(app)
    JWTManager(app)
    CORS(app, resources={r'/api/*': {'origins': '*'}})

    app.register_blueprint(predict_bp)
    app.register_blueprint(info_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(history_bp)

    @app.route('/')
    def index():
        return send_from_directory(app.static_folder, 'index.html')

    @app.route('/admin/data')
    def admin_data():
        from models.db_models import User, PredictionHistory
        users = User.query.all()
        result = []
        for u in users:
            result.append({
                'id': u.id,
                'username': u.username,
                'email': u.email,
                'created_at': str(u.created_at),
                'predictions': [{
                    'id': p.id,
                    'disease': p.disease,
                    'risk_level': p.risk_level,
                    'probability': p.probability,
                    'created_at': str(p.created_at)
                } for p in u.predictions]
            })
        return jsonify(result)

    with app.app_context():
        db.create_all()

    return app


if __name__ == '__main__':
    create_app().run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))