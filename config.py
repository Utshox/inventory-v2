import os

class Config:
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'csv'}
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-123'  # Default only used for development
    
    @staticmethod
    def init_app(app):
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)