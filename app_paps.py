import io
import json
import os
import numpy as np

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from flask import Flask, jsonify, request

import cv2

from torchvision.models.detection.retinanet import retinanet_resnet50_fpn

app = Flask(__name__)
num_classes = 2
IMAGE_SIZE = 1024
model = retinanet_resnet50_fpn(pretrained=False, min_size=IMAGE_SIZE, max_size=IMAGE_SIZE, 
                               num_classes=2, nms_thresh=0.3)
pretrained = torch.load('trained_model/retinanet_resnet50_fpn_1/model_65.pth')
model.load_state_dict(pretrained['model'])
model.eval()
device='cpu'

# only one class now, abnormal


# Transform input into the form our model expects
def transform_image(filestr):
    npimg = np.fromstring(filestr, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = torch.from_numpy(image).permute(2,0,1)
    image = image/255.

    image.unsqueeze_(0)                                    
    return image


# Get a prediction
@torch.no_grad()
def get_prediction(images):
    outputs = model(images)
#     outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]    
    out_boxes = outputs[0]['boxes']
    out_scores = outputs[0]['scores']
    out_labels = outputs[0]['labels']
    
    pred_boxes = []
    pred_scores = []
    pred_labels = []
    for b, s, l in zip(out_boxes, out_scores, out_labels) :
        if s > 0.3 :
            pred_boxes.append(b.numpy())
            pred_scores.append(s.numpy())
            pred_labels.append(l.numpy())    
    return pred_labels, pred_scores, pred_boxes


@app.route('/', methods=['GET'])
def root():
    return jsonify({'msg' : 'Try POSTing to the /predict endpoint with an RGB image attachment'})


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        print('post is called')
        filestr = request.files['file'].read()
#         print('file', type(file))
        if filestr is not None:
            input_tensor = transform_image(filestr)
            labels, scores, boxes = get_prediction(list(input_tensor))
            print(labels)
            print(scores)
            print(boxes)
#             class_id, class_name = render_prediction(prediction_idx)
#             return jsonify({'class_id': class_id, 'class_name': class_name})


if __name__ == '__main__':
    app.run()