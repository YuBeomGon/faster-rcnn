# https://tutorials.pytorch.kr/intermediate/flask_rest_api_tutorial.html
# https://tutorials.pytorch.kr/recipes/deployment_with_flask.html
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

from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn, FastRCNNPredictor

image_mean = torch.tensor([0.485, 0.456, 0.406])
image_std = torch.tensor([0.229, 0.224, 0.225])

app = Flask(__name__)
num_classes = 2
model = fasterrcnn_resnet50_fpn(pretrained=True, min_size=1024, max_size=1024, box_nms_thresh=0.3)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
checkpoint = torch.load('trained_model/fasterrcnn_resnet50_fpn/checkpoint.pth')
model.load_state_dict(checkpoint['model'])
model.eval()                                              # autograd를 끄고

device='cpu'

IMAGE_SIZE = 1024


# only one class now, ASCUS


# Transform input into the form our model expects
def transform_image(filestr):
    npimg = np.fromstring(filestr, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
#     image = cv2.imread(str(path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = torch.from_numpy(image).permute(2,0,1)
    image = image/255.
    image = (image - image_mean[:, None, None]) / image_std[:, None, None]

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
        if s > 0.5 :
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