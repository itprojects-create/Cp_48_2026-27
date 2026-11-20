"""
backend/models/db_models.py
SQLAlchemy models for MedPredict v2
Tables: User, PredictionHistory
"""
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from datetime import datetime
import json

db = SQLAlchemy()
bcrypt = Bcrypt()


class User(db.Model):
    __tablename__ = 'users'

    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80),  unique=True, nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

    predictions   = db.relationship('PredictionHistory', backref='user',
                                    lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            'id':         self.id,
            'username':   self.username,
            'email':      self.email,
            'created_at': self.created_at.isoformat()
        }


class PredictionHistory(db.Model):
    __tablename__ = 'prediction_history'

    id            = db.Column(db.Integer, primary_key=True)
    user_id       = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    disease       = db.Column(db.String(50),  nullable=False)   # 'cardiovascular' | 'diabetes'
    inputs_json   = db.Column(db.Text,        nullable=False)   # JSON of feature inputs
    risk_level    = db.Column(db.String(20),  nullable=False)   # Low / Moderate / High / Very High
    probability   = db.Column(db.Float,       nullable=False)   # 0.0 – 1.0
    created_at    = db.Column(db.DateTime,    default=datetime.utcnow)

    def to_dict(self):
        return {
            'id':          self.id,
            'disease':     self.disease,
            'inputs':      json.loads(self.inputs_json),
            'risk_level':  self.risk_level,
            'probability': round(self.probability * 100, 1),
            'created_at':  self.created_at.isoformat()
        }
