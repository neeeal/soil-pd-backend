import cv2
# import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow import convert_to_tensor
import gdown
import os
import csv
from db import db
import requests
import datetime
BASE_PATH = "ipynb\\models\\model_v3"

GEOLOCATION_API_KEY = os.getenv("GEOLOCATION_API_KEY")

# if os.path.exists("my_models")==False:
#     LINK = os.getenv("DRIVE_LINK")
#     gdown.download_folder(LINK, quiet=True, use_cookies=False)
#     print("Downloading model")
# print(os.path.isdir("my_models/designB.h5")) 
# segmentation_model = load_model(filepath='my_models/designB.h5')
# type_model = load_model('my_models/type_model.h5')
# print("Loaded models")

def generatedId(lat,lon):
    return

def base64_to_image(string):
    file = string.strip()
    padding = len(file) % 4
    if padding:
        file += '=' * (4 - padding)
    image_data = base64.b64decode(file)
    image_stream = BytesIO(image_data)
    pil_image = Image.open(image_stream#.stream
                        ).convert('RGB')#.resize((300, 300))
    data = np.array(pil_image)
    return data

def get_mask(image):
    image = image.astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    binr = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    binr = np.invert(binr)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(binr, kernel, iterations=3)
    
    return mask

def preprocess_image(string):
    # Load image
    img = base64_to_image(string)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_img = img.copy()
    rgb_image = orig_img.astype(np.uint8)

    mask = get_mask(rgb_image)
    output = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)

    rgb_planes = cv2.split(output)
    result_planes = []
    for plane in rgb_planes:
        processed_image = cv2.medianBlur(plane, 3)
        result_planes.append(processed_image)
    result = cv2.merge(result_planes)
    result = cv2.resize(result, (32, 32))
    return result

def load_limits(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        MINPH = float(next(reader)[1])
        MAXPH = float(next(reader)[1])
        MINMOISTURE = float(next(reader)[1])
        MAXMOISTURE = float(next(reader)[1])
            
    return { 'MINPH':MINPH,  'MAXPH':MAXPH , 'MINMOISTURE':MINMOISTURE , 'MAXMOISTURE':MAXMOISTURE }

def unprocess_label(label, maxPh, minPh, maxMoisture, minMoisture):
    moisture = np.array(label[0::2], dtype=float) * (float(maxMoisture) - float(minMoisture)) + float(minMoisture)
    ph = np.array(label[1::2], dtype=float) * (float(maxPh) - float(minPh)) + float(minPh)
    output = [moisture, ph]
    return output

def image_to_base64(image):
    image = Image.fromarray(preprocess_image(image))
    
    ## Convert Image to passable base64 string
    image_io = BytesIO()
    image.save(image_io, format='JPEG')
    image_bytes = image_io.getvalue()
    image64 = base64.b64encode(image_bytes).decode("ascii")
    return image64

# MODEL = load_model(os.path.join(BASE_PATH,'model_v1.h5'))
# Correct way to unpack a dictionary
limits = load_limits(os.path.join(BASE_PATH,'limits.csv'))
MINPH = limits['MINPH']
MAXPH = limits['MAXPH']
MINMOISTURE = limits['MINMOISTURE']
MAXMOISTURE = limits['MAXMOISTURE']

segmentation_model = None
def init_model(path):
    global segmentation_model
    if segmentation_model is None:
        segmentation_model = load_model(filepath=path)
        return segmentation_model
    
def get_acidity_moisture(image64):
    global segmentation_model
    init_model(os.path.join(BASE_PATH,'model_v3.h5'))

    image = preprocess_image(image64).reshape((1,32,32,3))/255.
    print(image.shape)
    result = (segmentation_model.predict(image)[0])
    processed_result = unprocess_label(result,MINPH, MAXPH, MINMOISTURE, MAXMOISTURE)
    finalMoisture = np.mean(processed_result[0])
    finalPh = np.mean(processed_result[1])
    return {"moisture": finalMoisture, "acidity": finalPh}

# def get_type(image64):
#     # {'clay': 0, 'sand': 1, 'silt': 2}
#     classes = ['clay','sand','silt']
#     image = cv2.resize(base64_to_image(image64),(75,75)).reshape((1,75,75,3))/255.
#     print(image.shape)
#     type_ = type_model.predict(image)
#     type_ = classes[np.argmax(type_)]
#     print(type_)
#     return type_

def remote_store(DATA):
    try: DATA['image'] = DATA['image'].split(',')[1]
    except: pass
    image = cv2.resize(base64_to_image(DATA['image']),(64,64)).tobytes()
    result = get_acidity_moisture(DATA['image'])
    moisture = result['moisture']
    acidity = result['acidity']
    nitrogen = DATA['nitrogen']
    phosphorus = DATA['phosphorus']
    potassium = DATA['potassium']
    latitude = DATA['latitude']
    longitude = DATA['longitude']
    # mapId = -1
    userId = -1
    robotId = -1
    # mapId = generatedId(14.700407062019375,121.03216730474672)
    db.ping(reconnect=True)
    with db.cursor() as cursor:
        # print("working")
        sql = """INSERT INTO `analysis` (`userId`, 
        `latitude`, `longitude`, `nitrogen`, `phosphorus`, 
        `potassium`, `moisture`, `acidity`, image) 
        VALUES (%s, 
                %s, %s, %s, %s, 
                %s, %s, %s, %s
                )"""
        cursor.execute(sql, (userId, 
                             latitude, longitude, nitrogen, phosphorus, 
                             potassium, moisture, acidity, image
                             ))
        entry = cursor.execute("SELECT * FROM `analysis` WHERE `userId` = %s ORDER BY `mapId` DESC LIMIT 1" % userId)
    db.commit()
    print(entry)
    data = {
        "userId":userId, 
        "latitude":latitude, 
        "longitude":longitude, 
        "nitrogen":nitrogen, 
        "phosphorus":phosphorus, 
        "potassium":potassium, 
        "moisture":moisture, 
        "acidity":acidity, 
    }
    return data

def get_maps(userId, order_by=0, page=1, limit=10):
    db.ping(reconnect=True)
    offset = (page - 1) * limit
    ORDER_BY = 'DESC' if order_by == 0 else 'ASC'
    
    with db.cursor() as cursor:
        cursor.execute(f"""
            SELECT * FROM `analysis` 
            WHERE `userId` = %s AND `dateDeleted` IS NULL 
            ORDER BY `mapId` {ORDER_BY} 
            LIMIT %s OFFSET %s
        """, (userId, limit, offset))
        data = cursor.fetchall()
        
        # Convert numeric values to float with 3 decimal places
        for d in data:
            for key, value in d.items():
                if isinstance(value, (int, float)):
                    d[key] = round(float(value), 3)

    db.commit()

    for x in range(len(data)):
        if data[x]["image"] == b'none':
            data[x]['image'] = base64.b64encode(data[x]["image"]).decode("ascii")
            continue
        # Convert binary image data to base64 encoded string
        image = Image.frombytes("RGB", (64, 64), data[x]['image']) 
        image_io = BytesIO()
        image.save(image_io, format='JPEG')
        image_bytes = image_io.getvalue()
        data[x]['image'] = base64.b64encode(image_bytes).decode("ascii")
        temp = [x["formatted_address"] if type(x)==dict else None for x in requests.get("https://maps.googleapis.com/maps/api/geocode/json?latlng={},{}&key={}".format(
            data[x]['latitude'],
            data[x]['longitude'],
            GEOLOCATION_API_KEY
        )).json()["results"]]
        data[x]['address'] = temp[0] if len(temp)>0 else None
        interpretations = interpret(data[x])
        data[x]['interpretations'] = interpretations

    return data

def store(userId, data):
    ## function to add entry from client side
    # print(data['longitude'])
    print(data['mapId'])
    db.ping(reconnect=True)
    with db.cursor() as cursor:
        sql = """INSERT INTO `analysis` (`mapId`, `userId`, 
                `latitude`, `longitude`, 
                `nitrogen`, `phosphorus`, 
                `potassium`, `moisture`, 
                `acidity`, `image`) 
                VALUES (
                    %s, %s, 
                    %s, %s, 
                    %s, %s, 
                    %s, %s, 
                    %s, %s
                )"""
        cursor.execute(sql, (data['mapId'], userId,
                             data['latitude'], data['longitude'], 
                             data['nitrogen'], data['phosphorus'], 
                             data['potassium'], data['moisture'], 
                             data['acidity'], data['image']))
    db.commit()
    print(data)
    return data

def update(mapId, data):
    ## function to update entry from client side
    db.ping(reconnect=True)
    # Construct the SQL statement
    sql = "UPDATE `analysis` SET "
    placeholders = ", ".join([f"`{key}` = %s" for key in data])
    sql += placeholders + " WHERE `mapId` = %s"
    values = list(data.values()) + [mapId]
    with db.cursor() as cursor:
        cursor.execute(sql, values)
    db.commit()
    return True


def delete(mapId):
    ## function to delete entry from client side
    db.ping(reconnect=True)
    with db.cursor() as cursor:
        sql = "UPDATE `analysis` SET `dateDeleted` = %s WHERE `mapId` = %s"
        cursor.execute(sql, (datetime.datetime.now(), mapId))
    db.commit()
    return True

def interpret(data):
    interpretations = {}
    acidity = ['acidic', 'neutral', 'alkaline']
    nitrogen = ['high', 'normal', 'low']
    phosphorus = ['high', 'normal',' low']
    potassium = ['high', 'normal', 'low']
    moisture = ['high', 'normal', 'low']
    
    if data["acidity"] < 7:
        acidity_arg = 0
    elif data["acidity"] >= 7 and data["acidity"] < 8:
        acidity_arg = 1
    else:
        acidity_arg = 2
    
    if data["nitrogen"] < 7:
        nitrogen_arg = 2
    elif data["nitrogen"] == 7:
        nitrogen_arg = 1
    else:
        nitrogen_arg = 0
    
    if data["phosphorus"] < 7:
        phosphorus_arg = 2
    elif data["phosphorus"] == 7:
        phosphorus_arg = 1
    else:
        phosphorus_arg = 0
    
    if data["potassium"] < 7:
        potassium_arg = 2
    elif data["potassium"] == 7:
        potassium_arg = 1
    else:
        potassium_arg = 0
    
    if data["moisture"] < 0.25:
        moisture_arg = 0
    elif data["moisture"] >= 0.17 and data["moisture"] <= 0.22:
        moisture_arg = 1
    else:
        moisture_arg = 2
    
    interpretations ={
        "acidity": acidity[acidity_arg],
        "nitrogen": nitrogen[nitrogen_arg],
        "phosphorus": phosphorus[phosphorus_arg],
        "potassium": potassium[potassium_arg],
        "moisture": moisture[moisture_arg],
    }
    return interpretations
