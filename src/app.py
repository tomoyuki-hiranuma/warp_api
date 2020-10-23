from flask import Flask, jsonify, request
import json
import base64
import os
import numpy as np
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

  img_stream = base64.b64decode(base_data)
  img_array = np.asarray(bytearray(img_stream), dtype=np.uint8)
  image_reviser = ImageReviser(img_array, max(paper_size), min(paper_size), click[0], click[1])
  image_reviser.run()
  mm_per_px = image_reviser.get_mm_per_px()
  img = image_reviser.get_img() # ndarray

  return jsonify({
    "data": mm_per_px
  })


if __name__ == '__main__':
  app.run()
