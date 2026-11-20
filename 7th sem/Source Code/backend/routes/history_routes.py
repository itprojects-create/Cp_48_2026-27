"""
backend/routes/history_routes.py
Endpoints to save and retrieve prediction history per user
"""
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity, verify_jwt_in_request
from models.db_models import db, PredictionHistory
import json

history_bp = Blueprint('history', __name__, url_prefix='/api/history')


@history_bp.route('/', methods=['GET'])
@jwt_required()
def get_history():
    user_id = get_jwt_identity()
    records = (PredictionHistory.query
               .filter_by(user_id=user_id)
               .order_by(PredictionHistory.created_at.desc())
               .limit(50)
               .all())
    return jsonify({'history': [r.to_dict() for r in records]}), 200


@history_bp.route('/', methods=['DELETE'])
@jwt_required()
def clear_history():
    user_id = get_jwt_identity()
    PredictionHistory.query.filter_by(user_id=user_id).delete()
    db.session.commit()
    return jsonify({'message': 'History cleared'}), 200


# Called internally from predict_routes after a prediction
def save_prediction(disease, inputs_dict, risk_level, probability):
    """Save a prediction. user_id is optional (guests allowed)."""
    user_id = None
    try:
        verify_jwt_in_request(optional=True)
        from flask_jwt_extended import get_jwt_identity
        user_id = get_jwt_identity()
    except Exception:
        pass

    record = PredictionHistory(
        user_id      = user_id,
        disease      = disease,
        inputs_json  = json.dumps(inputs_dict),
        risk_level   = risk_level,
        probability  = probability
    )
    db.session.add(record)
    db.session.commit()
    return record
