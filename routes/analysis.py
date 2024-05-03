from flask import jsonify, Blueprint, request, send_file    
from controllers import analysis_methods #import image_to_base64, get_acidity_moisture, get_type, base64_to_image, get_maps
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

@analysis_bp.route('/remoteStore', methods=["POST"])
def remote_store():
    DATA = request.get_json()
    msg = 'No Image Data Provided'
    if request.method == 'POST' and DATA:
        data = analysis_methods.remote_store(DATA)
        msg = "Succesfully analyzed soil."
        response = jsonify({'msg':msg, "data":data})
        response.headers['Content-Type'] = 'application/json'
        return response, 200
    return jsonify({'msg':msg}), 401

@analysis_bp.route('/<userId>', methods=["GET"])
def get_analysis(userId):
    data = analysis_methods.get_maps(userId)
    print("here")
    if data is not None:
        print("i am")
        return jsonify({"msg": "Successfully Retrieved", "number_entries":len(data), "analysis": data, "ok": "true"}), 200
    else:
        print("playing")
        return jsonify({"msg": "No data found for the provided user ID", "ok": "false"}), 400
    
@analysis_bp.route('/store/<userId>', methods=["POST"])
def store(userId):
    try:
        ## function to add entry from client side
        userId = userId
        data = request.get_json()
        print(data)
        if data and userId:
            result = analysis_methods.store(userId, data)
            print(result)
            return jsonify({"msg":"Succesfully stored data","data":result}), 200
        else:
            return jsonify({"msg":"Missing input fields."}), 400
    except Exception as e:
        return jsonify({"msg":"Server error", "error":e}), 500

@analysis_bp.route('/update/<mapId>', methods=["PUT"])
def update(mapId):
    ## function to update entry from client side
    data = request.json
    result = analysis_methods.update(mapId, data)
    return jsonify({"msg":f"Succesfully updated entry mapId {mapId}", "updates":data})

@analysis_bp.route('/<mapId>', methods=["DELETE"])
def delete(mapId):
    ## function to delete entry from client side
    result = analysis_methods.delete(mapId)
    return jsonify({"msg":f"Sucessfully deleted entry mapId {mapId}"})