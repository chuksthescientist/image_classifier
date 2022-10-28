import sys
import requests

def classify_images_api_request(image_paths, app_address, app_port):
    """
    util function to request classify_images api
    """
    # get api url
    func = 'api/classify_images'
    url = 'http://{0}:{1}/{2}'.format(app_address, app_port, func)
    
    # request json
    request_body = {'images':image_paths}
    
    # get response json
    response = requests.post(url=url, json=request_body).json()
    return response['results']

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='classifiy images')
    parser.add_argument('host', type=str, help='api host address')
    parser.add_argument('port', type=str, help='api port address')
    parser.add_argument('--paths', default=[], type=str, nargs='+',help='image paths')
    args = parser.parse_args()
    
    """
    python test_api.py <api_host> <api_port> --path <path_or_url_0>  <path_or_url_1> ...
    example:
    python test_api.py localhost 8989 --paths \
            ./demo_images/val_00150.jpg \
            http://localhost:8099/demo_images/val_00150.jpg \
            ./error_path \
            http://error
    """
    
    if len(args.paths) == 0:
        # run "python -m http.server 8099 > http.log 2>&1 &" before test
        # debug
        image_paths = [
            './demo_images/val_00150.jpg',
            'http://localhost:8099/demo_images/val_00150.jpg',
            'error_path',
            'http://error'
        ]
    else:
        image_paths = args.paths
    
    preds = classify_images_api_request(image_paths, args.host, args.port)
    for image_path, pred in zip(image_paths, preds):
        print(f'{image_path} -> {pred}')