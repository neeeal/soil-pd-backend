from flask import jsonify, Blueprint, request, send_file    
from controllers.user_methods import signup_func, login_func, update_func, current_user_func, forgot_password_func
user_bp = Blueprint('user',__name__)

@user_bp.route('/')
def index():
    msg = 'Welcome to User Route'
    return jsonify({'msg': msg}),200

@user_bp.route('/signup', methods=['POST'])
def signup():
    msg='Sign up route'
    value = signup_func()
    response = jsonify({'msg':msg, 'value':value})
    return response, 200


@user_bp.route('/login', methods=['POST'])
def login():
    msg='Login route'
    value = login_func()
    response = jsonify({'msg':msg, 'value':value})
    return response, 200

@user_bp.route('/update', methods=['PUT'])
def update():
    msg='Update route'
    value = update_func()
    response = jsonify({'msg':msg, 'value':value})
    return response, 200

@user_bp.route('/current_user', methods=['GET'])
def current_user():
    msg='Current user credentials route'
    value = current_user_func()
    response = jsonify({'msg':msg, 'value':value})
    return response, 200

@user_bp.route('/forgot_password', methods=['POST'])
def forgot_password():
    msg='Forgot password route'
    value = forgot_password_func()
    response = jsonify({'msg':msg, 'value':value})
    return response, 200