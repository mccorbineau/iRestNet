# iRestNet
### Unfolded proximal interior point algorithm for image deblurring

* **License**            : GNU General Public License v3.0  
* **Author**             : Marie-Caroline Corbineau
* **Institution**        : Centre de Vision Numérique, CentraleSupélec, Inria, Université Paris-Saclay
* **Email**              : mariecaroline.corbineau@gmail.com 
* **Related publication**: https://arxiv.org/abs/1812.04276
* Please use the following citation:
     > Carla Bertocchi, Emilie Chouzenoux, Marie-Caroline Corbineau, Jean-Christophe Pesquet and Marco Prato, **`Deep unfolding of a proximal interior point method for image restoration'**. To appear in Inverse Problems, doi:10.1088/1361-6420/ab460a (2019).


### Installation
1. Install miniconda.
2. Create an environment with python version 3.6.5.
   ```sh
   $ conda create -n iRestNet_env python=3.6.5
   ```
3. Inside this environment install the following packages.
   ```
   $ conda activate iRestNet_env
   $ conda install pytorch=0.4.0 cuda80 -c pytorch
   $ pip install torchvision==0.2.1 matplotlib==3.0.2 numpy==1.16.0 jupyterlab opencv-python==4.0.0.21 scipy==1.2.0 tqdm==4.29.1 scikit-image==0.14.2
   ```
4. Use the demo notebook to test and train iRestNet models.

### Files organization
* `Datasets`   
   * `BSD500_COCO1000_train_val` 
      * `train`: contains 200 RGB training images from the Berkeley segmentation dataset (BSD500) and 1000 RGB images from the COCO dataset 
      * `val`: contains 100 RGB validation images from the Berkeley segmentation dataset (BSD500)
   * `Groundtruth`: contains groundtruth test images
      * `full`
          * `BSD500`: contains 200 groundtruth test images from the Berkeley segmentation dataset
          * `Flickr30`: contains 30 groundtruth test images from Flickr30
       * `cropped`: contains cropped test images (created by the function create_testset)
   * `Testsets`: contains a folder for each test configuration including associated blur kernels (kernel.mat)
* `Model_files`    : contains iRestNet files
    * `Deg3PolySolver.py`: contains the cardan class used to compute the proximity operator of the logarithmic barrier
    * `iRestNet.py`: contains iRestNet_class, which is the main class including train and test functions
    * `model.py`: includes the definition of the layers in iRestNet
    * `modules.py`: useful functions used in iRestNet
    * `tools.py`: contains some useful side functions
* `PyTorch_ssim`   : forward and backward functions to use SSIM as the training loss, we use the code from https://github.com/Po-Hsun-Su/pytorch-ssim
* `Trainings`      : contains the models used in https://arxiv.org/abs/1812.04276 for every test configuration. In the demo notebook, the results of the aforementioned models on the testsets are saved in this folder.


### Demo file
`demo.ipynb`: shows how to test and train iRestNet

###### Thank you for using our code, kindly report any suggestion to the corresponding author.
