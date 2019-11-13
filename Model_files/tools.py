"""
Tools.
Functions
---------
    my_compare_ssim      : computes the Structural SImilarity Measure (SSIM) of an RGB image
	switch_test_config   : provides the blur kernel and the noise level for each test configuration
	compute_ssim_results : computes the mean SSIM of the blurred and restored images
	create_testset       : creates blurred test images using the chosen blur kernel and Gaussian noise standard deviation
@author: Marie-Caroline Corbineau
@date: 03/10/2019
"""

from skimage.measure import compare_ssim
from tqdm import tqdm
import scipy.io as sio
import os
import sys
import numpy as np
from torch.autograd import Variable
import torch
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from Model_files.modules import MyConv2d, add_Gaussian_noise, MyDataset, OpenMat_transf
from PIL import Image
import time


def my_compare_ssim(x_true,x_est):
    """
    Computes the Structural SImilarity Measure (SSIM) of an RGB image. Excludes a 6-pixels-wide frame 
    to provide a faire comparison with other methods that do not estimate well the borders.
    Parameters
    ----------
        x_true (numpy array): groundtruth image, size h*w*c
        x_est  (numpy array): restored image, size h*w*c
    Returns
    -------
        (float): SSIM
    """
    return compare_ssim(x_true[6:250,6:250,:], x_est[6:250,6:250,:], gaussian_weights=True, data_range=1, multichannel=True)

def switch_test_config(test_config):
    """
    Provides the blur kernel and the noise level for each test configuration.
    Parameters
    ----------
        test_config (str): test configuration
    Returns
    -------
        (list): list of three elements containing the blur kernel name (str), the noise level name (str) and 
                the minimal and maximal values (floats) for the noise standard deviation
    """
    switcher = {
        'GaussianA': ['gaussian_1_6','_std_0008',[0.008,0.008]],
        'GaussianB': ['gaussian_1_6','_std_001_005',[0.01,0.05]],
        'GaussianC': ['gaussian_3','_std_004',[0.04,0.04]],
        'MotionA'  : ['motion8','_std_001',[0.01,0.01]],
        'MotionB'  : ['motion3','_std_001',[0.01,0.01]],
        'Square'   : ['square_7','_std_001',[0.01,0.01]]
    }
    return switcher.get(test_config, "Invalid test configuration")
    
def compute_ssim_results(test_config, dataset, path_iRestNet):
    """
    Computes the mean SSIM of the blurred and restored images.
    Parameters
    ----------
        test_config   (str): test configuration from {'GaussianA','GaussianB','GaussianC','MotionA','MotionB','Square'}
        dataset       (str): name of the test set
        path_iRestNet (str): path to images restored with iRestNet
    Results
    -------
        (float): average SSIM of the blurred images
        (float): average SSIM of the restored images
    """
    ### test configuration
    name_kernel, name_std, noise_std_range = switch_test_config(test_config)
    
    ### loss function
    loss_type        = 'SSIM'

    ### path to folders with blurred images, true ones, and results for the different methods
    path_testset     = os.path.join('Datasets','Testsets')
    path_blurred     = os.path.join(path_testset, name_kernel+name_std, dataset)
    path_true        = os.path.join('Datasets','Groundtruth','cropped', dataset)
    file_names       = os.listdir(path_true)
        
    file_list       = [[os.path.join(path_true, i),
                        os.path.join(path_blurred, i),
                        os.path.join(path_iRestNet,i)] for i in file_names]

    ### initialization of vectors containing the ssim results
    ssim_blurred, ssim_iRestNet = np.zeros(len(file_names)),np.zeros(len(file_names))
  
    for i in range(0,len(file_names)):
        ### load images
        x_true          = sio.loadmat(file_list[i][0])['image']
        x_blurred       = sio.loadmat(file_list[i][1])['image']
        x_iRestNet      = sio.loadmat(file_list[i][2])['image']
        
        ### compute ssim
        ssim_blurred[i]       = my_compare_ssim(x_true, x_blurred)
        ssim_iRestNet[i]      = my_compare_ssim(x_true, x_iRestNet)

    return np.mean(ssim_blurred), np.mean(ssim_iRestNet)

def create_testset(name_set,path_groundtruth,path_testset,name_kernel,name_std,std_range,im_size):
    """
    Creates blurred test images using the chosen blur kernel and Gaussian noise standard deviation.
    Parameters
    ----------
        name_set          (str): name of the test set
        path_groundtruth  (str): path to groundtruth images
        path_testset      (str): path to the test set
        name_kernel       (str): name of the blur kernel
        name_std          (str): name of the noise standard deviation
        std_range        (list): list of two elements, minimal and maximal pixel values
        im_size   (numpy array): number of rows and columns in the images, size 1*2
    """
    np.random.seed(0)
    torch.manual_seed(1)
    
    dtype       =  torch.FloatTensor
    chan        =  3 # nb of channels
    
    path_kernel = os.path.join(path_testset,name_kernel+name_std,'kernel.mat')
    path_save   = os.path.join(path_testset,name_kernel+name_std,name_set)
    if not os.path.exists(path_save):
        os.makedirs(path_save)
        
    ### blur kernel
    kernel      = sio.loadmat(path_kernel)
    kernel      = torch.from_numpy(kernel[name_kernel]).type(dtype)
    tens_kernel = torch.zeros(chan,chan,kernel.size(0),kernel.size(0))
    for i in range(chan):
        tens_kernel[i,i,:,:]=kernel
        
    ### transformations to be applied to the groundtruth images
    transf2 = transforms.Compose([MyConv2d(tens_kernel,mode='single').cuda(),add_Gaussian_noise(std_range)]) 
    if os.path.exists(os.path.join(path_groundtruth,'cropped',name_set)):
        transf1         = OpenMat_transf()
        already_cropped = 'yes'
        data            = MyDataset(folder=os.path.join(path_groundtruth,'cropped',name_set),transf1=transf1, transf2=transf2, need_names='yes')
    else:
        # center-crops the test images to match the input size
        transf1          = transforms.Compose([transforms.CenterCrop(im_size),transforms.ToTensor()]) 
        path_save_true   = os.path.join(path_groundtruth,'cropped',name_set)
        if not os.path.exists(path_save_true):
            os.makedirs(path_save_true)
        already_cropped  = 'no'
        data             = MyDataset(folder=os.path.join(path_groundtruth,'full',name_set),transf1=transf1, transf2=transf2, need_names='yes')
    
    loader  = DataLoader(data, batch_size=1, shuffle=False) 

    for minibatch in tqdm(loader,file=sys.stdout):
        [image_name,x,x_degraded] = minibatch
        file_name_degraded        = os.path.join(path_save,image_name[0])
        sio.savemat(file_name_degraded,{'image':x_degraded[0].permute(1,2,0).cpu().numpy().astype('float64')})
        if x[0].size(0)==3:
            # RGB images
            sio.savemat(file_name_degraded,{'image':x_degraded[0].permute(1,2,0).cpu().numpy().astype('float64')})
        elif x[0].size(0)==1:
            sio.savemat(file_name_degraded,{'image':x_degraded[0,0].cpu().numpy().astype('float64')})

        if already_cropped == 'no':
            file_name_true = os.path.join(path_save_true,image_name[0])
            if x[0].size(0)==3:
                # RGB images
                sio.savemat(file_name_true,{'image':x[0].permute(1,2,0).cpu().numpy().astype('float64')})
            elif x[0].size(0)==1:
                sio.savemat(file_name_true,{'image':x[0,0].cpu().numpy().astype('float64')})