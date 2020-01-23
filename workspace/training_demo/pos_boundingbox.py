# Improting Image class from PIL module 
from PIL import Image 
import xml.etree.ElementTree as ET
import xmltodict
import pprint
import json
import glob
import os 

x = 1
for filename in os.listdir("images/train_real_label"): 
    x = x+1
    
    with open('images/train_real/'+filename) as fd:
        doc = xmltodict.parse(fd.read())
        
    json1 = json.dumps(doc)
    loaded_json = json.loads(json1)
    print('LOADED JSON>>', loaded_json)
    filename=(loaded_json["annotation"]["filename"])
    path=('images/train_real/'+filename)
    out_file = 'images/train_cropped/'+ filename
    print('Path', path)

    # Opens a image in RGB mode 
    im = Image.open(path)     

    if 'object' in loaded_json["annotation"]:

        objects = loaded_json["annotation"]["object"]
        
    if objects:
        try:
            if (objects["name"]) :
                if (objects["name"] =="bucket" or  objects["name"] =="human" or objects["name"] =="truck") :
                    print("Objects :", objects["name"])
                    left = int(objects["bndbox"]["xmin"])
                    top = int(objects["bndbox"]["ymin"])
                    right = int(objects["bndbox"]["xmax"])
                    bottom = int(objects["bndbox"]["ymax"])

                    im1 = im.crop((left, top, right, bottom)) 
                    im1.save(out_file)
        
        except:
            for object1 in objects:
                if (object1["name"] and object1["name"]=="bucket"):
                    print("Object1 :", object1["name"])
                    left = int(object1["bndbox"]["xmin"])
                    top = int(object1["bndbox"]["ymin"])
                    right = int(object1["bndbox"]["xmax"])
                    bottom = int(object1["bndbox"]["ymax"])
                    print(left,top,right,bottom)

                    im1 = im.crop((left, top, right, bottom))
                    im1.save(out_file)