from flask import jsonify, Blueprint, request, send_file    
from controllers.analysis_methods import image_to_base64, get_acidity_moisture, get_type, base64_to_image, get_maps
from db import db
from cv2 import resize
from PIL import Image
import base64
from io import BytesIO

analysis_bp = Blueprint('analysis',__name__)

@analysis_bp.route('/')
def index():
    msg = 'Welcome to Analysis Route'
    return jsonify({'msg': msg}),200

@analysis_bp.route('/store', methods=["POST"])
def soil_analysis():
    DATA = request.get_json()
    msg = 'No Image Data Provided'
    if request.method == 'POST' and DATA:
        try: DATA['image'] = DATA['image'].split(',')[1]
        except: pass
        image = resize(base64_to_image(DATA['image']),(64,64)).tobytes()
        acidity, moisture = get_acidity_moisture(DATA['image'])
        type_ = get_type(DATA['image'])
        nitrogen = -1
        phosporus = -1
        potassium = -1
        latitude = 14.625983543082867
        longitude = 121.0617254517838
        # mapId = -1
        userId = -1
        robotId = -1
        db.ping(reconnect=True)
        with db.cursor() as cursor:
            # print("working")
            sql = """INSERT INTO `analysis` (`userId`, 
            `latitude`, `longitude`, `nitrogen`, `phosphorus`, 
            `potassium`, `moisture`, `acidity`, `type`, image) 
            VALUES (%s, %s, %s, %s, %s, 
                    %s, %s, %s, %s, %s
                    )"""
            cursor.execute(sql, (userId, latitude, 
                                 longitude, nitrogen, phosporus, 
                                 potassium, moisture, acidity, 
                                 type_, image))
        db.commit()
        msg = 'Successfully extracted soil properties from image'
        response = jsonify({'msg':msg, #'image64':image64, 
            'location':{'longitude':longitude, 'latitude':latitude},
            'soil_properties':{'acidity':acidity,
                               'moisture':moisture,
                               'type_':type_,
                               'nitrogen':nitrogen,
                               'phosporus':phosporus,
                               'potassium':potassium,}})
        response.headers['Content-Type'] = 'application/json'
        return response, 200
    return jsonify({'msg':msg}), 401

@analysis_bp.route('/get', methods=["GET"])
def get_analysis():
    userId = request.get_json()['userId']
    data = get_maps(userId)
    if data!=None:
        response = jsonify({"msg":"Succesfully Retrieved", "analysis":data})
        return response, 200
    return jsonify({"msg":"Client Error"}),400