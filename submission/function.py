# import common used libs
import os
import urllib.request

# cv libs
import cv2
import numpy as np

# deep learning libs
import torch
from torch.utils.data import DataLoader, Dataset
from efficientnet_pytorch import model as enet

import albumentations as A
from albumentations.pytorch import ToTensorV2

# for better visualization
from tqdm import tqdm

# helper functions
def load_image_url(image_url):
    """
    download the image, convert it to a NumPy array, and then read it into OpenCV format
    inputs:
        image_url: the image url to load (str)
    outputs:
        image: loaded image (numpy array)
    """ 
    
    with urllib.request.urlopen(image_url) as resp:
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image
    
def load_image_local(image_filepath):
    """
    load local image using OpenCV
    inputs:
        image_filepath: the path of image file to load (str)
    outputs:
        image: loaded image (numpy array)
    """ 
    
    image = cv2.imread(image_filepath)
    return image
    
    
# define model
class Swish(torch.autograd.Function):
    """
    Swish activation function.
    Reference: https://arxiv.org/abs/1710.05941v1
    """ 
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class Swish_Module(torch.nn.Module):
    """
    Wrapper of Swish activation function.
    """ 
    def forward(self, x):
        return Swish.apply(x)
    
class ClsModel(torch.nn.Module):
    """
    Classification Model.
    We use EfficientNet as backbone.
    Reference: https://arxiv.org/abs/1905.11946
    A classification head with Swish activation and Dropout are added after EfficientNet backbone.
    """ 
    def __init__(self, backbone, out_dim):
        super(ClsModel, self).__init__()
        self.enet = enet.EfficientNet.from_name(backbone)
        
        in_features = self.enet._fc.in_features
        self.enet._fc = torch.nn.Identity()
        self.activation = Swish_Module()
        self.head = torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features),
            self.activation,
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(in_features, out_dim),
        )
        
    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.head(x)
        return x
    
class ShoesDataset(Dataset):
    """
    Dataset for multiprocessing data loading, support local path and urls.
    """
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            if os.path.isfile(image_path):
                # if path is file, load as local image
                image = load_image_local(image_path)
            else:
                # load image from url
                image = load_image_url(image_path)
            # if load image success, error message is set to ""
            msg = ''
        except Exception as err:
            # if the path is wrong, return a dummy image and the error message.
            image = np.zeros((128,128,3), dtype=np.uint8)
            msg = repr(err)
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        sample = {
            'image': image, 
            'err': msg,
        }
        return sample

class ShoesPredictor:
    """
    Singleton Predictor.

    Get the predictor using: ShoesPredictor.default_predictor()
    This will make sure there are only one model is loaded to GPU. 
    This is useful when construct the API.
    """
    _predictor = None
    
    @staticmethod
    def default_predictor():
        """
        singleton wrapper.
        <batch_size> and <num_workers> can be redueced if not have enought GPU memory or CPU cores.
        """
        if ShoesPredictor._predictor is None:
            print('initialize model ...')
            base_size = 640
            backbone='efficientnet-b6'
            checkpoint = "./checkpoints/final.pth"
            batch_size = 16
            num_workers = 4
            ShoesPredictor._predictor = ShoesPredictor(base_size, backbone, checkpoint,
                                                       batch_size, num_workers)
            print('initialize model done.')
        return ShoesPredictor._predictor
    
    def __init__(self, base_size, backbone, checkpoint, 
                 batch_size, num_workers, verbose=True):
        # test data transformation
        self.transform = A.Compose(
            [
                A.LongestMaxSize(max_size=base_size),
                A.PadIfNeeded(p=1.0, min_height=base_size, min_width=base_size),
                A.CenterCrop(height=base_size, width=base_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        
        # build model
        model = ClsModel(backbone, out_dim=7)
        
        # load model from checkpoint
        model.load_state_dict(torch.load(checkpoint))
        
        # move model to gpu 
        self.model = model.cuda()
        self.model.eval()
        
        # class names, the order should be as same as in training
        self.classes = ["sneakers", "boots", "loafers", "sandals", "flip_flops", "soccer_shoes", "no_shoe_found"]
        
        # hyperparameters
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # control the stdout logger
        self.verbose = verbose
        
    def process_single_image(self, image_path):
        """
        Function for predicting single image. 
        This function is not used in classify_images.
        We left it here for following development.
        """
        pred_label = None
        try:
            if os.path.isfile(image_path):
                image = load_image_local(image_path)
            else:
                image = load_image_url(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = self.transform(image=image)["image"]
        
            prediction = self.model(image.unsqueeze(0).cuda())
            idx = torch.argmax(prediction, -1).detach().cpu().numpy()[0]
            pred_label = self.classes[idx]
        except Exception as err:
            pred_label = repr(err)
        return pred_label
        
    def process_batch_images(self, image_paths):
        """
        predict labels for images in batch manner.
        inputs:
            image_paths: [<path_or_url_0>, <path_or_url_1>, ...]
        outputs:
            pred_labels: [<label_0>, <label_1>, ...]
        """ 
        # build test dataset
        dataset = ShoesDataset(image_paths, self.transform)
        
        # multiprocessing data loader
        loader = DataLoader(dataset, shuffle=False, drop_last=False,
                            batch_size=self.batch_size,
                            num_workers=self.num_workers)
        
        pred_labels = []
        if self.verbose:
            pbar = tqdm(total=len(image_paths))
            
        # predict batch by batch
        for idx, batch in enumerate(loader):
            # forward the images to model and get the predicted results.
            with torch.no_grad():
                prediction = self.model(batch['image'].cuda())
                idxs = torch.argmax(prediction, -1).detach().cpu().numpy()
            
            # collect results
            for idx, err in zip(idxs, batch['err']):
                if not err:
                    pred_labels.append(self.classes[idx])
                else:
                    # if err message is not "", return the err message
                    pred_labels.append(err)
                if self.verbose:
                    pbar.update(1)
        if self.verbose:         
            pbar.close()

        return pred_labels

def classify_images(image_paths):
    """
    predict labels for images in batch manner.
    inputs:
        image_paths: [<path_or_url_0>, <path_or_url_1>, ...]
    outputs:
        preds: [<label_0>, <label_1>, ...]
    exception:
        If one of the path or URL in the list is wrong, the function will capture the Exception 
        and return the Exception string as the results at corresponding position.
        For example, if some inputs are wrong path and URL: 
            […, “error_path”, “http://error”, …]. 
        Then the function will output: 
            […, “ValueError(…)”, “URLError(…)”, …].
    """ 
    predictor = ShoesPredictor.default_predictor()
    
    print('process images ...')
    
    # process image batch by batch
    preds = predictor.process_batch_images(image_paths)
    
    # # process image one by one, left here for following development.
    # preds = []
    # for image_path in tqdm(image_paths):
    #     pred_label = predictor.process_single_image(image_path)
    #     preds.append(pred_label)
    return preds