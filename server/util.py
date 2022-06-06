# @zizhen xian
# version 1.0 only dealing with one disease.

import json
import base64
import cv2
import numpy as np
from torchvision import models 
import torch

__class_name_to_number = {}
__class_number_to_name = {}

__model = None


def get_b64_image_test():
    # b64.txt is a file of the byte 64 format of one image.
    with open("b64.txt") as f:
        return f.read()


def load_saved_model():
    model_path = "/Users/zizhenxian/study/fall21/OCT_web_application/server/artifact/model_epoch_1.pth"
    print("loading saved artifacts.. ")

    global __class_name_to_number
    global __class_number_to_name

    with open("/Users/zizhenxian/study/fall21/OCT_web_application/server/artifact/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v: k for k, v in __class_name_to_number.items()}

    global __model
    
    __model = models.resnet101(pretrained=True)

    num_ftrs = __model.fc.in_features
    __model.fc = torch.nn.Linear(num_ftrs, 2)

    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    __model.load_state_dict(checkpoint['model'])
    
    print("done with loading saved models. ")


def get_cv2_image_from_base64_string(b64str):
    # credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv
    # -python-library

    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


# the input is either an image or a base64 string transformed.
def load_image(image_path, base64_image):
    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(base64_image)

    # chop right part of the image. 
    img = img[:496, img.shape[1]-768+128:img.shape[1]-128, :]
    img = torch.transpose(torch.FloatTensor(img),0,2)
    img = img.unsqueeze(0)
    return img


def class_number_to_name(class_num):
    return __class_number_to_name[class_num]


def classify(image_b64, file_path = None):
    img = load_image(file_path, image_b64)

    with torch.no_grad():
        __model.eval()
        logits = __model(img)
        _, pred = torch.max(logits,1)
        pred = int(pred) # tensor to int 
        
    return class_number_to_name(pred)


if __name__ == '__main__':
    load_saved_model()
    img = get_b64_image_test()
    print(classify(img, None))
