from flask import Flask, jsonify
from flask_cors import CORS
from routes.analysis import analysis_bp
from routes.user import user_bp
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    msg = """Welcome to Backend of 
Design of an Image Soil Analysis 
for Robotic Nutrient Mapping System""".replace('\n', '')
    return jsonify({"msg": msg}),200

app.register_blueprint(analysis_bp, url_prefix='/api/analysis')
app.register_blueprint(user_bp, url_prefix='/api/user')

if __name__ == '__main__':
    app.run(
        debug=True, 
        host = os.getenv('API_HOST', default='localhost'), 
        port=os.getenv("PORT", default=5000))
    print("APP RUNNING ...")