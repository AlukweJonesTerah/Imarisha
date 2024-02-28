from home import db, login, app
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from hashlib import md5
from sqlalchemy.exc import SQLAlchemyError
from flask import (
    jsonify, Blueprint
)


@login.user_loader
def load_user(id):
    return User.query.get(int(id))


@app.errorhandler(SQLAlchemyError)
def handle_database_error(e):
    # Log error
    app.logger.error(f'Database error: {str(e)}')
    # user-friendly response
    return jsonify({'error': 'A database error occurred'}), 500


class User(UserMixin, db.Model):
    __table_name__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), index=True, unique=True)
    email = db.Column(db.String(64), index=True, unique=True)
    password_hash = db.Column(db.String(128))
    about_me = db.Column(db.String(120))
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'User: {self.username}'

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def avatar(self, size):
        digest = md5(self.email.lower().encode('utf-8')).hexdigest()
        return f'https://www.gravatar.com/avatar/{digest}?d=identicon&s={size}'


class Conversation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    user_input = db.Column(db.String(1000))
    bot_response = db.Column(db.String(1000))

    def __repr__(self):
        return f"<Conversation {self.id}>"
