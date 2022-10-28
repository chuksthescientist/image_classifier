# install miniconda
wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh 
bash miniconda.sh -b -p ./conda 
source ./conda/etc/profile.d/conda.sh
conda activate base

# install python reqirements
pip install --upgrade pip 
pip install tensorboard cmake   # cmake from apt-get is too old
pip install torch==1.9.0 torchvision==0.10.0 -f https://download.pytorch.org/whl/cu101/torch_stable.html
pip install ipython jupyter tqdm matplotlib \
                opencv-python scikit-learn scikit-image \
				albumentations==1.0.0 catalyst==21.5 \
				efficientnet_pytorch==0.7.1 pandas \
				flask waitress

chmod 777 download_train.sh
chmod 777 download_deploy.sh