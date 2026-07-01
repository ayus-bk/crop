from datetime import datetime
from database import db


# ===========================
# USERS TABLE
# ===========================
class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="farmer")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship
    farmer_profile = db.relationship(
        "FarmerProfile",
        backref="user",
        uselist=False,
        cascade="all, delete"
    )

    recommendations = db.relationship(
        "Recommendation",
        backref="user",
        cascade="all, delete"
    )


# ===========================
# FARMER PROFILE TABLE
# ===========================
class FarmerProfile(db.Model):
    __tablename__ = "farmer_profiles"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False,
        unique=True
    )

    phone = db.Column(db.String(20), nullable=False)
    district = db.Column(db.String(100), nullable=False)
    municipality = db.Column(db.String(100), nullable=False)
    ward = db.Column(db.Integer, nullable=False)
    farm_size = db.Column(db.Float, nullable=False)


# ===========================
# RECOMMENDATION HISTORY
# ===========================
class Recommendation(db.Model):
    __tablename__ = "recommendations"

    id = db.Column(db.Integer, primary_key=True)

    farmer_id = db.Column(
        db.Integer,
        db.ForeignKey("users.id"),
        nullable=False
    )

    nitrogen = db.Column(db.Float, nullable=False)
    phosphorus = db.Column(db.Float, nullable=False)
    potassium = db.Column(db.Float, nullable=False)

    temperature = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    ph = db.Column(db.Float, nullable=False)
    rainfall = db.Column(db.Float, nullable=False)

    recommended_crop = db.Column(db.String(100), nullable=False)

    prediction_time = db.Column(
        db.DateTime,
        default=datetime.utcnow
    )