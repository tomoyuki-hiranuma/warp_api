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

  print(type(base_data.encode(encoding='utf-8')))
  bin_data = base64.b64decode(base_data.encode(encoding='utf-8'))
  with open('jeans.jpg', 'wb') as f:
    f.write(bin_data)

  nparr = np.fromstring(bin_data, np.uint8)
  img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR) 
  print(img_np.shape)
  # img_stream = base64.b64decode(base_data)
  # byte_img = bytearray(img_stream)
  # print(type(img_stream))
  # print(type(byte_img))
  # img_array = np.frombuffer(bin_data, dtype=np.uint8)
  # print(img_array.shape[0])
  # print(np.max(img_array))
  # print(np.min(img_array))
  # print(img_array)
  image_reviser = ImageReviser(img_np, max(paper_size), min(paper_size), click[0], click[1])
  image_reviser.run()
  mm_per_px = image_reviser.get_mm_per_px()
  after_img = image_reviser.get_img()
  result, dst_data = cv2.imencode('.jpg', after_img)
  after_image = base64.b64encode(dst_data).decode()

  return jsonify({
    "mm_per_px": mm_per_px,
    "after_image": after_image
  })


if __name__ == '__main__':
  app.run()
