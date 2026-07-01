from app import app
from database import db
from models import User

with app.app_context():

    admin = User.query.filter_by(username="admin").first()

    if admin:
        print("Admin already exists.")
    else:
        admin = User(
            full_name="System Administrator",
            username="admin",
            email="admin@crop.com",
            password="admin123",   # We'll hash later
            role="admin"
        )

        db.session.add(admin)
        db.session.commit()

        print("Admin created successfully!")