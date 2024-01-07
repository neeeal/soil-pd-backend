from flask import jsonify, Blueprint, request
from controllers.analysis_methods import preprocess_image
import base64

analysis_bp = Blueprint('analysis',__name__)

@analysis_bp.route('/')
def index():
    msg = 'Welcome to Analysis Route'
    return jsonify({'msg': msg}),200

@analysis_bp.route('/soil_analysis', methods=["POST"])
def soil_analysis():
    DATA = request.get_json()
    msg = 'No Image Data Provided'
    if request.method == 'POST' and DATA:
        try: DATA['image'] = DATA['image'].split(',')[1]
        except: pass
        image = preprocess_image(DATA['image'])
        image64 = str(base64.b64encode(image))
        #### handle base64 passing of image as response
        msg = 'Successfully extracted soil properties from image'
        return jsonify({'msg':msg, 'processed_image':image64}), 200
    return jsonify({'msg':msg}), 401