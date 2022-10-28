# import common used libs
import os
import numpy as np
import glob

# import classify_images function
from function import classify_images

def prepare_test_paths():
    """
    helper function for debug. 
    Generate debug test list.
    Note: run "python -m http.server 8099 > http.log 2>&1 &" before test
    """
    test_root = './demo_images/'
    url_root = 'http://localhost:8099/demo_images'
    test_image_paths = []
    for file_path in glob.glob(os.path.join(test_root, '*')):
        test_image_paths.append(file_path)
        test_image_paths.append(os.path.join(url_root, os.path.relpath(file_path, test_root)))
    np.random.shuffle(test_image_paths)
    test_image_paths = test_image_paths[:20]
    test_image_paths.append('error_path')
    return test_image_paths

def main(image_paths):
    """
    predict and print results to stdout
    """
    preds = classify_images(image_paths)
    print('-'*80)
    print('inputs to classify_images():')
    print(image_paths)
    print('-'*50)
    print('outputs of classify_images():')
    print(preds)
    print('-'*80)
    for image_path, pred in zip(image_paths, preds):
        print(f'{image_path} -> {pred}')
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='classifiy images')
    parser.add_argument('--paths', default=[], type=str, nargs='+',help='image paths')
    args = parser.parse_args()
    
    """ 
    python test_function.py --paths <path_or_url_0>  <path_or_url_1> ...
    example: 
    python test_function.py --paths \
        ./demo_images/val_00150.jpg \
        http://localhost:8099/demo_images/val_00150.jpg \
        ./error_path \
        http://error
    """

    # run "python -m http.server 8099 > http.log 2>&1 &" before test
    if len(args.paths) == 0:
        # debug
        image_paths = prepare_test_paths()
        main(image_paths)
    else:
        main(args.paths)