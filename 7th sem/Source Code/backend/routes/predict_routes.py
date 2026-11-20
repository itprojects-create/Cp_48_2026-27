from flask import Blueprint, request, jsonify
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from services.predictor import predict

predict_bp = Blueprint('predict', __name__)


@predict_bp.route('/api/predict', methods=['POST'])
def predict_route():
    data = request.get_json(force=True)

    disease = data.get('disease', '').strip().lower()
    if not disease:
        return jsonify({'error': 'disease field is required'}), 400

    # Frontend sends inputs nested under 'inputs' key
    feature_dict = data.get('inputs', data)
    feature_dict = {k: v for k, v in feature_dict.items() if k not in ('disease', 'model')}

    model_key = data.get('model', 'ensemble')

    try:
        result = predict(disease, feature_dict, model_key=model_key)
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500


@predict_bp.route('/api/compare-models', methods=['POST'])
def compare_route():
    from services.predictor import compare_models
    data = request.get_json(force=True)
    disease = data.get('disease', '').strip().lower()
    feature_dict = data.get('inputs', data)
    feature_dict = {k: v for k, v in feature_dict.items() if k not in ('disease', 'model')}
    try:
        result = compare_models(disease, feature_dict)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500