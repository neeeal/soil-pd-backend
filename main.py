from flask import Flask, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    msg = """Welcome to Backend of 
Design of an Image Soil Analysis 
for Robotic Nutrient Mapping System""".replace('\n', '')
    return jsonify({"msg": msg}),200


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
