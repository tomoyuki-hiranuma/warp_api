from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import base64
import os
import numpy as np
import cv2
from utils.image_reviser import ImageReviser

app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False

@app.route('/')
def index():
  return jsonify({
    "message": "Hello!!"
  })
  
@app.route('/same', methods=['GET', 'POST'])
def same():
  return request.get_data()

@app.route('/reply', methods=['POST'])
def reply():
  data = request.json['image']
  base_data = data['before_base']
  click = data['clicked_position']
  paper_size = data['paper_size']
  click_x = int(click[0])
  click_y = int(click[1])
  paper_long = int(paper_size[0])
  paper_short = int(paper_size[1])
  if paper_long < paper_short:
    paper_long, paper_short = paper_short, paper_long

  img_np = Base64ToNdarray(base_data)

  image_reviser = ImageReviser(img_np, paper_long, paper_short, click_x, click_y)
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
  app.debug = True
  app.run(host = '0.0.0.0')
  # app.run(port=int(os.environ.get("PORT", 5000)))
