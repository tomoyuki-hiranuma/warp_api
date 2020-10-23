from flask import Flask, jsonify, request
import json
import base64
import os

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
  user_id = data['user_id']

  bin_data = base64.b64decode(base_data.encode())
  path = "public/{0}/images/jeans.jpg".format(user_id)
  file_path = os.path.dirname(path)
  if not os.path.exists(file_path):
    os.makedirs(file_path)
  with open(path, 'bw') as f:
    f.write(bin_data)
  
  return jsonify({
    "data": paper_size
  })


if __name__ == '__main__':
  app.run()
