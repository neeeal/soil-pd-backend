from flask import jsonify, Blueprint, request, send_file    
from controllers.analysis_methods import image_to_base64, get_acidity_moisture, get_type
from PIL import Image
import base64
from io import BytesIO

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
        image64 = image_to_base64(DATA['image'])
        acidity, moisture = get_acidity_moisture(image64)
        type_ = get_type(DATA['image'])
        nitrogen = -1
        phosporus = -1
        potassium = -1
        msg = 'Successfully extracted soil properties from image'
        response = jsonify({'msg':msg, 'image64':image64, 
            'soil_properties':{'acidity':acidity,
                               'moisture':moisture,
                               'type_':type_,
                               'nitrogen':nitrogen,
                               'phosporus':phosporus,
                               'potassium':potassium,}})
        response.headers['Content-Type'] = 'application/json'
        return response, 200
    return jsonify({'msg':msg}), 401