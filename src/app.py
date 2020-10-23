from flask import Flask, jsonify, request
import json
import base64
import os
import numpy as np
import cv2
from utils.image_reviser import ImageReviser

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/')
def index():
  return jsonify({
    "message": "Hello World!!"
  })

@app.route('/reply', methods=['POST'])
def reply():
  data = request.json['image']
  base_data = data['before_base']
  click = data['clicked_position']
  paper_size = data['paper_size']

  img_np = Base64ToNdarray(base_data)

  image_reviser = ImageReviser(img_np, max(paper_size), min(paper_size), click[0], click[1])
  image_reviser.run()

  mm_per_px = image_reviser.get_mm_per_px()
  img_array = image_reviser.get_img()

  revised_image = NdarrayToBase64(img_array)
  
  return jsonify({
    "status": 200,
    "data": {
      "mm_per_px": mm_per_px,
      "after_image": revised_image
    }
  })

def NdarrayToBase64(data):
  result, dst_data = cv2.imencode('.jpg', data)
  after_image = base64.b64encode(dst_data).decode()
  return after_image

def Base64ToNdarray(data):
  bin_data = base64.b64decode(data.encode(encoding='utf-8'))
  nparr = np.fromstring(bin_data, np.uint8)
  img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
  return img_np

if __name__ == '__main__':
  app.run()
