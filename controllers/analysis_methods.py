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
from db import db
import datetime

if os.path.exists("my_models")==False:
    LINK = os.getenv("DRIVE_LINK")
    gdown.download_folder(LINK, quiet=True, use_cookies=False)
    print("Downloading model")
print(os.path.isdir("my_models/designB.h5")) 
segmentation_model = load_model(filepath='my_models/designB.h5')
type_model = load_model('my_models/type_model.h5')
print("Loaded models")

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

def preprocess_image(string):
    # Load image
    img = base64_to_image(string)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_img = img.copy()

    gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    # Threshold Processing
    ret, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Noise removal
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=7)
    bin_img = cv2.dilate(bin_img, kernel, iterations=5)
    # Distance transform
    dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
    # Foreground area
    ret, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
    sure_fg = sure_fg.astype(np.uint8)

    new_image_size=32
    # Apply the mask to the original image to extract the region
    region = cv2.bitwise_and(orig_img, orig_img, mask=sure_fg)
    # Find the bounding box coordinates (non-zero pixels)
    non_zero_coords = np.argwhere(region > 0)
    min_y, min_x, _ = non_zero_coords.min(axis=0)
    max_y, max_x, _ = non_zero_coords.max(axis=0)
    # Crop the region to include only non-zero pixels
    cropped_region = region[min_y:max_y + 1, min_x:max_x + 1]

    rgb_planes = cv2.split(cropped_region)
    result_planes = []
    # Create a CLAHE object.
    clahe = cv2.createCLAHE(tileGridSize=(3,3),clipLimit=10)
    for plane in rgb_planes:
        processed_image = cv2.medianBlur(plane, 7)
        processed_image = clahe.apply(processed_image) 
        result_planes.append(processed_image)
    result = cv2.merge(result_planes)

    HSV = cv2.cvtColor(result,cv2.COLOR_RGB2HSV)
    HSV = cv2.resize(HSV, (new_image_size, new_image_size))
    H,S,V = cv2.split(HSV)
    V *= 0
    HS = cv2.merge([H,S,V])
    return HS

def image_to_base64(image):
    image = Image.fromarray(preprocess_image(image))
    
    ## Convert Image to passable base64 string
    image_io = BytesIO()
    image.save(image_io, format='JPEG')
    image_bytes = image_io.getvalue()
    image64 = base64.b64encode(image_bytes).decode("ascii")
    return image64

def get_acidity_moisture(image64):
    image = preprocess_image(image64).reshape((1,32,32,3))/255.
    print(image.shape)
    acidity = str(segmentation_model.predict(image)[0][0])
    print(acidity)
    # acidity = -1
    moisture = -1
    return acidity, moisture

def get_type(image64):
    # {'clay': 0, 'sand': 1, 'silt': 2}
    classes = ['clay','sand','silt']
    image = cv2.resize(base64_to_image(image64),(75,75)).reshape((1,75,75,3))/255.
    print(image.shape)
    type_ = type_model.predict(image)
    type_ = classes[np.argmax(type_)]
    print(type_)
    return type_

def remote_store(DATA):
    try: DATA['image'] = DATA['image'].split(',')[1]
    except: pass
    image = cv2.resize(base64_to_image(DATA['image']),(64,64)).tobytes()
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
    mapId = generatedId(14.700407062019375,121.03216730474672)
    db.ping(reconnect=True)
    with db.cursor() as cursor:
        # print("working")
        sql = """INSERT INTO `analysis` (`mapId`,`userId`, 
        `latitude`, `longitude`, `nitrogen`, `phosphorus`, 
        `potassium`, `moisture`, `acidity`, `type`, image) 
        VALUES (%s, %s, %s, %s, %s, 
                %s, %s, %s, %s, %s,
                %s
                )"""
        cursor.execute(sql, (mapId, userId, latitude, 
                                longitude, nitrogen, phosporus, 
                                potassium, moisture, acidity, 
                                type_, image))
    db.commit()
    msg = 'Successfully extracted soil properties from image'
    return msg

def get_maps(userId):
    db.ping(reconnect=True)
    with db.cursor() as cursor:
        cursor.execute(f"SELECT * FROM `analysis` WHERE `userId` = {userId} AND `dateDeleted` IS NULL")
        data = cursor.fetchall()
        # print(len(data))
    db.commit()
    for x in range(len(data)):
        # print(data[x]['image'])
        # print(data[x]['mapId'])
        if(data[x]["image"]==b'none'): 
            # print("wporking")
            data[x]['image'] = base64.b64encode(data[x]["image"]).decode("ascii")
            continue
        image = Image.frombytes("RGB", (64, 64), data[x]['image']) 
        image_io = BytesIO()
        image.save(image_io, format='JPEG')
        image_bytes = image_io.getvalue()
        data[x]['image'] = base64.b64encode(image_bytes).decode("ascii")
    # print("finished")
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
                `acidity`, `type`, 
                `image`) 
                VALUES (%s, %s, %s, %s, %s, 
                %s, %s, %s, %s, %s,
                %s
                )"""
        cursor.execute(sql, (data['mapId'], userId,
                             data['latitude'], data['longitude'], 
                             data['nitrogen'], data['phosphorus'], 
                             data['potassium'], data['moisture'], 
                             data['acidity'], data['soilType'],
                             data['image']))
    db.commit()
    print(data)
    return data

def update(recordId):
    ## function to update entry from client side
    return

def delete(mapId):
    ## function to delete entry from client side
    db.ping(reconnect=True)
    with db.cursor() as cursor:
        sql = "UPDATE `analysis` SET `dateDeleted` = '{}' WHERE `mapId` = '{}'".format(datetime.datetime.now(), mapId)
        cursor.execute(sql)
    db.commit()
    return True