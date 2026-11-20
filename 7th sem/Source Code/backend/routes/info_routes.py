# ─────────────────────────────────────────────────────────────
#  routes/info_routes.py
#  Blueprint: /api/diseases  and  /api/health
# ─────────────────────────────────────────────────────────────

from flask  import Blueprint, jsonify
from config import DISEASE_REGISTRY

info_bp = Blueprint("info", __name__, url_prefix="/api")


@info_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "2.0.0"})


@info_bp.route("/diseases", methods=["GET"])
def diseases():
    """Returns the disease registry so the frontend can build forms dynamically."""
    return jsonify({
        k: {
            "display_name": v["display_name"],
            "features":     v["features"],
        }
        for k, v in DISEASE_REGISTRY.items()
    })
