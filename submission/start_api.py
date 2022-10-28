# common sys lib
import os
import sys

# setup gpu before import torch
GPU_ID = sys.argv[1]
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

# use flash and waitress to run api
from flask import Flask, request, jsonify
from waitress import serve

# import test function
from function import ShoesPredictor, classify_images

app = Flask(__name__)
@app.route('/api/classify_images', methods=['POST'])
def classify_images_api():
    """
    POST Request:
        json_data = {
            images: [<path1>, <url2>, ..., <err_url10>, ...]
        }
    Response:
        response_info = {
            result: [<label1>, <label2>, ..., <err_message10>, ...]
        }
    """
    if request.method == 'POST':
        print('-'*80)
        json_data = request.get_json()
        print('Request info: ', json_data)

        result = classify_images(json_data['images'])
        response_info = {'results': result}

        print('Response info:', response_info)
        response_body = jsonify(response_info)
        return response_body

def main(APP_PORT):
    serve(app, host="0.0.0.0", port=APP_PORT)

if __name__ == '__main__':
    """
    python -u start_api.py <GPUID> <APP_PORT>
    example:
    python -u start_api.py 0 8989 > api_log.txt 2>&1 &
    """
    
    # the port app run
    APP_PORT = sys.argv[2]
    
    # initial model for API
    # this will take a few seconds, initialize it before service start.
    ShoesPredictor.default_predictor()
    
    # start api service
    main(APP_PORT)
