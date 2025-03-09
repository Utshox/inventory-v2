import os

class Config:
    UPLOAD_FOLDER = 'uploads'
    ALLOWED_EXTENSIONS = {'csv'}
    GOOGLE_API_KEY = os.environ.get('AIzaSyDSi2TudRCVfBdC5IcJLaJsCv3NR4c--aY')
    SECRET_KEY = os.environ.get('4c76dd8203fa360b87a1929a8fa1ed92') or 'dev-key-123'
    
    @staticmethod
    def init_app(app):
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)