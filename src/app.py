from flask import Flask, jsonify, request
import json

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

@app.route('/')
def index():
  return jsonify({
    "message": "Hello World!!"
  })

@app.route('/reply')
def reply():
  return request.get_data()

@app.route('/warp', methods=["POST"])
def warp():
  return jsonify({
    "status": 200,
    "data": {
      "hello",
    },
  })

if __name__ == '__main__':
  app.run()
