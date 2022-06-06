import streamlit as st
from torchvision import models, transforms
import numpy as np 
import torch 
import torch.nn as nn
from customized_transformation import Rescale, RightCrop, ToTensor, Normalize, Filt, ExtractRPE, Remove 
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model(label):
    BEST_MODEL_PATH = 'best_model_' + label + '.pth'
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft = model_ft.to(device)
    
    model_ft.load_state_dict(torch.load(BEST_MODEL_PATH))
    model_ft.eval()
    
    return model_ft

with st.spinner("Model is being loaded.."):
    model = load_model("HRF")

st.write("""
        # Auto Diagnosis 
        """)

file = st.file_uploader("Plase upload your OCT image", type = ["jpg"])
# file = "0000-0000L_1003.jpg"
if file is not None:
  my_img = Image.open(file)
  img = np.array(my_img)

import cv2 
from PIL import Image
import numpy as np 
st.set_option('deprecation.showfileUploaderEncoding', False)

def import_and_predict(image_data, model):
    preprocess = transforms.Compose([
                              RightCrop(),
                              Rescale((256,256)),
                              Remove(),
                              Normalize(),
                              ToTensor()
                              ])

    x = preprocess(image_data)
    x = x[None]
    x = x.type('torch.FloatTensor')

    outputs = model(x)
    output_array = np.array(outputs.argmax(1))
    return output_array

if file is None:
    st.text("Please upload an image file")
else:
    print(img.shape)
    predictions = import_and_predict(img,model)
    st.write(predictions[0])
    print("this image is " + str(predictions[0]))

