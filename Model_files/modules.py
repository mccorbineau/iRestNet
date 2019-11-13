"""
Functions and classes used by the iRestNet model.
Functions
---------
	compute_PSNR         : computes the average peak signal-to-noise ratio of a list of images
	compute_PSNR_SSIM    : computes the average peak signal-to-noise ratio and structural similarity measure of a 
                           list of images scaled by the batch size over the total number of images
    OpenMat              : converts a numpy array loaded from a .mat file into a properly ordered tensor
	RGBToGray            : convert an RGB image into a gray image
    GrayToRGB            : convert a gray image into an RGB image by repeating the gray channel
	TensorFilter         : creates a list of tensors from a list of filters, such that the result of the 2-D 
                           convolution with one of a tensor and an image of size c*h*w is the convolution of 
                           each input channel with the filter (no combination of the channels)
	TransposeSquareFilter: returns Ht, the transpose of the 2-D convolution operator (H), and Ht*H
Classes
-------
    OpenMat_transf    : transforms an array into an ordered tensor
	CircularPadding   : performs circular padding on a batch of images for cyclic convolution
	MyConv2d          : performs circular convolution on images with a constant filter
	add_Gaussian_noise: adds Gaussian noise to images with a noise standard deviation randomly selected within a range
	MyDataset         : loads and transforms images before feeding it to the first layer L_0 of the network
	MyDataset_OneBlock: loads an image before feeding it to layer L_i, i>0
	MyTestset         : loads test images
@author: Marie-Caroline Corbineau
@date: 03/10/2019
"""

from torch.autograd import Variable
import torch
import torch.nn as nn
import numpy as np
from numpy.fft import fft2, fftshift, ifft2, ifftshift
import os
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
import PyTorch_ssim as compute_SSIM
import scipy.io as sio

def compute_PSNR(x_true, x):
    """
    Computes the average peak signal-to-noise ratio of a list of images.
    Parameters
    ----------
        x_true (torch.FloatTensor): ground-truth images, size n*c*h*w 
	    x      (torch.FloatTensor): restored images, size n*c*h*w 
    Returns
    -------
        (torch.FloatTensor): average PSNR of a list of images (x^(i))_{1<i<n} expressed in dB
            -10*sum_{i=1}^n(log_10(||x_true^(i)-x^(i)||^2/(c*h*w)))
    """
    return -10*torch.mean(np.log10(torch.mean(torch.mean(torch.mean((x_true-x)**2,1),1),1)))

def compute_PSNR_SSIM(x_true, x_before, x_after, size_set):
    """
    Computes the average peak signal-to-noise ratio and structural similarity measure of a 
    list of images scaled by the batch size over the total number of images.
    Parameters
    ----------
       	x_true   (Variable): ground-truth images, data of size n*c*h*w
	    x_before (Variable): blurred images, data of size n*c*h*w
	    x_after  (Variable): restored images, data of size n*c*h*w
	    size_set      (int): total number of images
    Returns
    -------
       	(numpy array): PSNR and SSIM values before and after restoration, size 2*2
    """
    size_batch  = x_true.data.size()[0]
    snr_before  = compute_PSNR(x_true.data, x_before.data)* size_batch/size_set
    snr_after   = compute_PSNR(x_true.data, x_after.data)* size_batch/size_set
    ssim_before = torch.Tensor.item(compute_SSIM.ssim(x_before, x_true))* size_batch/size_set
    ssim_after  = torch.Tensor.item(compute_SSIM.ssim(x_after, x_true))* size_batch/size_set
    return np.array((((snr_before),(snr_after)),((ssim_before),(ssim_after))))

def OpenMat(x):
    """
    Converts a numpy array loaded from a .mat file into a properly ordered tensor.
    Parameters
    ----------
        x (numpy array): image loaded from a .mat file, size h*w*c, c in {2,3}   
    Returns
    -------
        (torch.FloatTensor): size c*h*w
    """
    if len(x.shape)==3:
        return torch.from_numpy(x).permute(2,0,1).type(torch.FloatTensor)
    elif len(x.shape)==2:
        return torch.from_numpy(x).type(torch.FloatTensor).unsqueeze(0)
    
class OpenMat_transf(object):
    """
    Transforms an array into an ordered tensor.
    """
    def __init__(self):
        super(OpenMat_transf,self).__init__()
    def __call__(self,x):
        return OpenMat(x)
    
def RGBToGray(yes_no, x):
    """
    Convert an RGB image into a gray image.
    Parameters
    ----------
        yes_no          (str): 'yes' if x needs to be converted to gray, 'no' if it does not
	    x (torch.FloatTensor): RGB image, size c*h*w
    Returns
    -------
        (torch.FloatTensor): gray image, size 1*h*w 
    """
    if yes_no=='yes':
        return (0.2989 * x[0,:,:] + 0.5870 * x[1,:,:] + 0.1140 * x[2,:,:]).unsqueeze(0)
    elif yes_no=='no':
        return x

def GrayToRGB(x):
    """
    Convert a gray image into an RGB image by repeating the gray channel.
    Parameters
    ----------
	    x (torch.FloatTensor): RGB image, size c*h*w
    Returns
    -------
                      (str): 'yes' if x is a gray image, 'no' if it is RGB
        (torch.FloatTensor): RGB image, size 3*h*w
    """
    if x.size(0)<3:
        return 'yes', x.repeat(3,1,1)
    elif x.size(0)==3:
        return 'no', x

def TensorFilter(filt_list, c=3, dtype=torch.FloatTensor):
    """
    Creates a list of tensors from a list of filters, such that the result of the 2-D 
    convolution with one of a tensor and an image of size c*h*w is the convolution of 
    each input channel with the filter (no combination of the channels).
    Parameters
    ----------
        filt_list (list): list of numpy arrays, square filters of size h*h
        c          (int): number of channels
        dtype     (type): (default is torch.FloatTensor)
    Returns
    -------
        tens_filt_list (list): list of torch.FloatTensors, each element is of size c*c*h*h
    """
    tens_filt_list=[]
    for filt in filt_list:
        tens_filt = torch.zeros(c,c,filt.shape[0],filt.shape[0])
        for i in range(c):
            tens_filt[i,i,:,:] = torch.from_numpy(filt).type(dtype)
        tens_filt_list.append(tens_filt.clone())
    return tens_filt_list

def TransposeSquareFilter(filt, im_size, need_norm='no'):
    """
    Returns Ht, the transpose of the 2-D convolution operator (H), and Ht*H.
    Parameters
    ----------
        filt    (numpy array): square filter, with an odd number of rows
        im_size (numpy array): square array
        need_norm       (str): (default is 'no')
    Returns
    -------
        transpose_filter (numpy array): transpose of the 2-D convolution operator (Ht)
        square_filter    (numpy array): Ht*H
                               (float): only if need_norm = 'yes', spectral norm of Ht*H
    """
    #indices used for the zero-padding
    a = int((im_size[0]-filt.shape[0]+1)/2) 
    b = a+filt.shape[0]
    c = int(np.floor(filt.shape[0]/2))
    fourier_filt             = np.zeros(np.array(im_size))  #padding the kernel with zeros
    fourier_filt[a:b][:,a:b] = np.copy(filt)
    fourier_filt             = fft2(fftshift(fourier_filt)) #fourier transform 
    #fftshift because the blur kernel has odd dimensions  
    transpose_filter = ifftshift(ifft2(np.conj(fourier_filt))).real[a:b,a:b]
    square_filter    = ifftshift(ifft2(np.absolute(fourier_filt)**2 )).real[a-c:b+c,a-c:b+c]
    if need_norm=='yes':
        return transpose_filter, square_filter, np.max(np.absolute(fourier_filt)**2)
    else:
        return transpose_filter, square_filter

class CircularPadding(nn.Module):
    """
    Performs circular padding on a batch of images for cyclic convolution.
    Attributes
    ----------
        pad_size (int): padding size, same on all sides
    """
    def __init__(self, pad_size):
        super(CircularPadding, self).__init__()
        self.pad_size = pad_size
    
    def forward(self, batch):
        """
        Performs a circular padding on a batch of images (for circular convolution).
        Parameters
        ----------
            batch (torch.FloatTensor): batch of images, size n*c*h*w
        Returns
        -------
            (Variable): data type is torch.FloatTensor, size n*c*(h+2*pad_size)*(w+2*pad_size), padded images
        """
        h = batch.size()[2]
        w = batch.size()[3]
        z    = torch.cat((batch[:,:,:,w-self.pad_size:w],batch,batch[:,:,:,0:self.pad_size]),3)
        z    = torch.cat((z[:,:,h-self.pad_size:h,:],z,z[:,:,0:self.pad_size,:]),2)
        return Variable(z)
    
class MyConv2d(nn.Module):
    """
    Performs circular convolution on images with a constant filter.
    Attributes
    ----------
        kernel (torch.cuda.FloatTensor): size c*c*h*w filter
        mode                      (str): 'single' or 'batch'
        stride                    (int): dilation factor
        padding                        : instance of CircularPadding or torch.nn.ReplicationPad2d
    """
    def __init__(self, kernel, mode, pad_type = 'circular', padding=0, stride=1):
        """
        Parameters
        ----------
            gpu                  (str): gpu id
            kernel (torch.FloatTensor): convolution filter
            mode                 (str): indicates if the input is a single image of a batch of images
            pad_type             (str): padding type (default is 'circular')
            padding              (int): padding size (default is 0)
            stride               (int): dilation factor (default is 1)
        """
        super(MyConv2d, self).__init__()
        self.gpu      = 'cuda:0'
        self.kernel   = nn.Parameter(kernel,requires_grad=False)   
        self.mode     = mode #'single' or 'batch'
        self.stride   = stride
        if padding==0:
            size_padding = int((kernel[0,0].size(0)-1)/2)
        else:
            size_padding = padding
        if pad_type == 'replicate':
            self.padding = nn.ReplicationPad2d(size_padding)
        if pad_type == 'circular':
            self.padding = CircularPadding(size_padding)
            
    def forward(self, x): 
        """
        Performs a 2-D circular convolution.
        Parameters
        ----------
            x (torch.FloatTensor): image(s), size n*c*h*w 
        Returns
        -------
            (torch.FloatTensor): result of the convolution, size n*c*h*w if mode='single', size c*h*w if mode='batch'
        """
        if self.mode == 'single':
            return F.conv2d(self.padding(x.unsqueeze(0).cuda()), self.kernel, stride=self.stride).data[0]
        if self.mode == 'batch':
            return F.conv2d(self.padding(x.data), self.kernel, stride=self.stride)
    
class add_Gaussian_noise(object):
    """
    Adds Gaussian noise to images with a noise standard deviation randomly selected within a range.
    Parameters
    ----------
        std_min (double): minimal value for the noise standard deviation
        std_max (double): maximal value for the noise standard deviation
    """
    def __init__(self,std_range):
        super(add_Gaussian_noise,self).__init__()
        self.std_min = std_range[0]
        self.std_max = std_range[1]
    def __call__(self,x):
        """
        Adds Gaussian noise to images.
        Parameters
        ----------
            x (torch.FloatTensor): images, size n*c*h*w 
        Returns
        -------
            (torch.FloatTensor): noisy images, size n*c*h*w 
        """
        std = np.random.uniform(low=self.std_min,high=self.std_max)
        return x + torch.cuda.FloatTensor(x.size()).normal_(0,std)
    
class MyDataset(torch.utils.data.Dataset):
    """
    Loads and transforms images before feeding it to the first layer L_0 of the network.
    Attributes
    ----------
        folder      (str): path to the folder containing the images
        file_names (list): list of strings, list of names of images
        file_list  (list): list of strings, paths to images
        transf1    (list): list of Transform objects, classic transformation (cropp, to tensor, ...)
        transf2    (list): list of Transform objects, specific for each channel (blurr, noise, ...)
        need_names  (str): 'yes' for outputting image names, 'no' else
    """
    def __init__(self, folder='/path/to/folder/', transf1=None, transf2=None, need_names='no'):
        """
        Loads and transforms images before feeding it to the network.
        Parameters
        ----------
        folder     (str): path to the folder containing the images (default '/path/to/folder/')
        transf1   (list): list of Transform objects (default is None)
        transf2   (list): list of Transform objects (default is None)
        need_names (str): 'yes' for outputting image names, 'no' else (default is 'no')
        """
        super(MyDataset, self).__init__()
        self.folder     = folder
        self.file_names = os.listdir(self.folder)
        self.file_list  = [os.path.join(self.folder, i) for i in self.file_names]
        self.transf1    = transf1  
        self.transf2    = transf2 
        self.need_names = need_names
    def __getitem__(self, index):
        """
        Loads and transforms an image.
        Parameters
        ----------
            index (int): index of the image in the list of files, can point to a .mat, .jpg, .png.
                         If the image has just one channel the function will convert it to an RGB format by 
                         repeating the channel.
       Returns
       -------
                          (str): optional, image name without the extension
            (torch.FloatTensor): image before transformation, size c*h*w
            (torch.FloatTensor): image after transformation, size c*h*w
        """
        if os.path.splitext(self.file_names[index])[1] == '.mat':
            # if .mat file
            i = self.transf1(sio.loadmat(self.file_list[index])['image'])
        else:
            # if .jpg or .png file
            i = self.transf1(Image.open(self.file_list[index]))
        if i.size(0)<3:
            # if grayscale image
            j = self.transf2(i.repeat(3,1,1))[0].unsqueeze(0)
            j = j.repeat(3,1,1) # all three channels are the same
        else:
            # if RGB image
            j = self.transf2(i)
        if self.need_names=='no':
            return i, j
        elif self.need_names=='yes':
            return os.path.splitext(self.file_names[index])[0], i, j
    def __len__(self):
        return len(self.file_list)
    
class MyDataset_OneBlock(torch.utils.data.Dataset):
    """
    Loads an image before feeding it to layer L_i, i>0.
    Attributes
    ----------
        file_names         (list): list of strings, image names, length is n
        file_list          (list): list of strings, paths to: (i) ground-truth, (ii) output of previous layer, 
                                   (iii) result of the transposed convolution operator applied to the degraded image, 
                                   (iv) approximation of noise standard deviation), size n*4
        transf (Transform object): transforms an array into a tensor
    """
    def __init__(self, folder_true, folder_previous_block, folder_Htxblurred, folder_std_approx):
        super(MyDataset_OneBlock, self).__init__()
        self.file_names = os.listdir(folder_true)
        self.file_list  = [[os.path.join(folder_true,i),
                            os.path.join(folder_previous_block,i),
                            os.path.join(folder_Htxblurred,i),
                            os.path.join(folder_std_approx,os.path.splitext(i)[0]+".txt")] for i in self.file_names]
        self.transf = transforms.ToTensor()
    def __getitem__(self, index):
        """
        Loads and transforms an image.
        Parameters
        ----------
            index (int): index of the image in the list of files
        Returns
        -------
                          (str): image name without the extension
            (torch.FloatTensor): ground-truth image, size c*h*w 
            (torch.FloatTensor): output of previous layer, size c*h*w 
            (torch.FloatTensor): result oftranspose of convolution applied to degraded image, size c*h*w
            (torch.FloatTensor): approximation of the noise standard deviation, size 1
        """
        return os.path.splitext(self.file_names[index])[0], self.transf(sio.loadmat(self.file_list[index][0])['image']), self.transf(sio.loadmat(self.file_list[index][1])['image']), self.transf(sio.loadmat(self.file_list[index][2])['image']),torch.tensor(np.loadtxt(self.file_list[index][3]))
    #255* for version of python older than 3.6.5
    def __len__(self):
        return len(self.file_list)

class MyTestset(torch.utils.data.Dataset):
    """
    Loads test images.
    Attributes
    ----------
        file_names (list): list of strings, names of images, size n
        file_list  (list): list of strings, paths to images, size n
    """
    def __init__(self, folder):
        super(MyTestset, self).__init__()
        self.file_names      = os.listdir(folder)
        self.file_list       = [os.path.join(folder, i) for i in self.file_names]
    def __getitem__(self, index):
        """
        Loads a test image, if the image is gray, the channel is repeated to create an RGB image.
        Parameters
        ----------
            index (int): index of the image in the list of files
        Returns
        -------
                          (str): image name without the extension
            (torch.FloatTensor): test image, size c*h*w 
        """
        image_test = GrayToRGB(OpenMat(sio.loadmat(self.file_list[index])['image']))
        return self.file_names[index],image_test
    def __len__(self):
        return len(self.file_list)