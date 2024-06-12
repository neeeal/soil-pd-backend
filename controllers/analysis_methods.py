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
import matplotlib.pyplot as plt
import requests
import datetime
BASE_PATH = "ipynb\\models\\model_v8"

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
    binr = cv2.threshold(gray, 16, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    binr = np.invert(binr)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(binr, kernel, iterations=2)
    mask = (mask // 255).astype(np.uint8)
    return mask

def preprocess_image(image):
    # Load image
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_img = image.copy()
    rgb_image = orig_img.astype(np.uint8)

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    binr = cv2.threshold(gray, 16, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    binr = np.invert(binr)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(binr, kernel, iterations=2)
    mask = (mask // 255).astype(np.uint8)
    rgb_planes = cv2.split(rgb_image)
    result_planes = []
    for plane in rgb_planes:
        processed_image = cv2.medianBlur(plane, 3)
        processed_image = cv2.bitwise_and(processed_image, processed_image, mask=mask)
        result_planes.append(processed_image)
    result = cv2.merge(result_planes)
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

def mask_labels(mask, label):
    channel_0 = cv2.bitwise_and(label[:, :, 0], label[:, :, 0], mask=mask)
    channel_1 = cv2.bitwise_and(label[:, :, 1], label[:, :, 1], mask=mask)
    return [channel_0, channel_1]

def unprocess_label(label, maxPh, minPh, maxMoisture, minMoisture):
    # Convert label to float and flatten the arrays
    label = np.array(label, dtype=float)
    label_0_flat = label[0].flatten()
    label_1_flat = label[1].flatten()

    # Filter out zero values
    filtered_label_0 = label_0_flat[label_0_flat != 0]
    filtered_label_1 = label_1_flat[label_1_flat != 0]

    # Unnormalize the values
    moisture = filtered_label_0/10 #* (float(maxMoisture) - float(minMoisture)) + float(minMoisture)
    ph = filtered_label_1 #* (float(maxPh) - float(minPh)) + float(minPh)

    # Return the mean values of moisture and pH
    return [np.mean(moisture), np.mean(ph)]

def unprocess_label_wmask(image,label):
    def make_mask(image):
        image = np.array(image, dtype=np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = (gray > 0).astype(np.uint8)
        return mask

    unnormalized_image = np.array(image * 255).astype(np.uint8)
    
    mask = make_mask(unnormalized_image)

    masked_labels = mask_labels(mask, np.array(label))

    results = unprocess_label(masked_labels, MAXPH, MINPH, MAXMOISTURE, MINMOISTURE)
    return results

def image_to_base64(image):
    image = Image.fromarray(image)
    
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
    init_model(os.path.join(BASE_PATH,'designB_v8.h5'))

    image = cv2.resize(base64_to_image(image64),(64, 64))
    normalized_image = preprocess_image(image).reshape((1,64,64,3))/255.
    # for i in normalized_image[0]:
    #     print(i)
    result = (segmentation_model.predict(normalized_image)[0])
    print(result.shape)
    processed_result = unprocess_label_wmask(image, result)
    print(processed_result)
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
    image = cv2.resize(base64_to_image(DATA['image']),(128,128)).tobytes()
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
    
    if data["acidity"] < 7:
        acidity_arg = "Acidic"
    elif data["acidity"] >= 7 and data["acidity"] < 8:
        acidity_arg = "Neutral"
    else:
        acidity_arg = "Alkaline"
    
    if data["nitrogen"] >= 0 and data["nitrogen"] <= 127:
        phosphorus_arg = "Low"
    elif data["nitrogen"] >= 108 and data["nitrogen"] <= 214:
        phosphorus_arg = "Medium"
    elif data["nitrogen"] >= 215 and data["nitrogen"] <= 250:
        phosphorus_arg = "High"
    else:
        nitrogen_arg = "Very High"
    
    if data["phosphorus"] >= 0 and data["phosphorus"] <= 5:
        phosphorus_arg = "Low"
    elif data["phosphorus"] >= 6 and data["phosphorus"] <= 12:
        phosphorus_arg = "Moderately Low"
    elif data["phosphorus"] >= 13 and data["phosphorus"] <= 25:
        phosphorus_arg = "Moderately High"
    elif data["phosphorus"] >= 26 and data["phosphorus"] <= 50:
        phosphorus_arg = "High"
    else:
        phosphorus_arg = "Very High"
    
    if data["potassium"] >= 0 and data["potassium"] <= 75:
        potassium_arg = "Low"
    elif  data["potassium"] >= 76 and data["potassium"] <= 113:
        potassium_arg = "Sufficient"
    elif  data["potassium"] >= 114 and data["potassium"] <= 150:
        potassium_arg = "Sufficient+"
    elif  data["potassium"] >= 151 and data["potassium"] <= 250:
        potassium_arg = "Sufficient++"
    else:
        potassium_arg = "Sufficient+++"
    
    if data["moisture"] <= 0.25:
        texture_arg = "Sandy"
    elif data["moisture"] > 0.25 and data["moisture"] <= 0.45:
        texture_arg = "Loam"
    else:
        texture_arg = "Clay"
    
    interpretations ={
        "acidity": acidity_arg,
        "nitrogen": nitrogen_arg,
        "phosphorus": phosphorus_arg,
        "potassium": potassium_arg,
        "moisture": "",
        "Texture": textire_arg
    }
    return interpretations
