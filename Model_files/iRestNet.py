import glob
import os
import gc
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import pickle
from IPython.display import clear_output
from Model_files.model import myModel, myLastLayer, SSIM_loss
import PyTorch_ssim
from Model_files.modules import MyConv2d, add_Gaussian_noise, MyDataset, MyDataset_OneBlock, CircularPadding, TensorFilter, RGBToGray, TransposeSquareFilter, compute_PSNR_SSIM, MyTestset, OpenMat
from PIL import Image
from math import ceil
from tqdm import tqdm
import sys
        
class iRestNet_class(nn.Module):
    """
    Includes the main training and testing methods of iRestNet.
    Attributes
    ----------
        name_kernel            (str): blur kernel name
        name_std               (str): noise standard deviation setting
        noise_std_range       (list): minimal and maximal pixel values
        im_size        (numpy array): image size
        path_test              (str): path to the folder containing the test sets
        path_train             (str): path to the training set folder 
        path_save              (str): path to the folder dedicated to saved models
        mode                   (str): 'first_layer' if training the first layer, 'greedy' if training the following layers one by one,
                                      'last_layers_lpp' if training the last 10 layers + lpp, 'test' if testing the model (default is 'first_layer')
        lr_first_layer        (list): list of two elements, first one is the initial learning rate to train the first layer, 
                                      second one is the number of epochs after which the learning rate is multiplied by 0.9 (default is [1e-2,5])    
        lr_greedy             (list): list of two elements, first one is the initial learning rate to train the layers, 
                                      second one is the number of epochs after which the learning rate is multiplied by 0.9 (default is [5e-3,5])         
        lr_last_layers_lpp    (list): list of two elements, first one is the initial learning rate to train the last 10 layers conjointly with lpp, 
                                      second one is the number of epochs after which the learning rate is multiplied by 0.9 (default is [1e-3,50])  
        nb_epochs             (list): list of three integers, number of epochs for training the first layer, the remaining layers, 
                                      and the last 10 layers + lpp, respectively (default is [40,40,600])          
        nb_blocks              (int): number of unfolded iterations (default is 40)    
        nb_greedy_blocks       (int): number of blocks trained in a greedy way (default is 30)
        batch_size            (list): list of three integers, number of images per batch for training, validation and testing, respectively (default is [10,10,1])                
        loss_type              (str): name of the training loss (default is 'SSIM')        
        kernel   (torch.FloatTensor): blurr kernel     
        kernel_2 (torch.FloatTensor): convolution kernel corresponding to the operator Ht*H            
        dtype            (data type): data type
        Ht                (MyConv2d): 2-D convolution operator corresponding to the operator Ht         
        model              (myModel): iRestNet layers    
        last_layer     (myLastLayer): post-processing layer
        sigmoid                     : sigmoid layer 
        train_loader    (DataLoader): loader for the training set
        val_loader      (DataLoader): loader for the validation set
        size_train             (int): number of images in the training set
        size_val               (int): number of images in the validation set
    """
    def __init__(self, test_conditions, folders, mode='first_layer', 
                 lr_first_layer=[1e-2,5], lr_greedy=[5e-3,5], lr_last_layers_lpp=[1e-3,50],
                 nb_epochs=[40,40,600], nb_blocks=40, nb_greedy_blocks=30, batch_size=[10,10,1], loss_type='SSIM'):
        """
        Parameters
        ----------
            test_conditions    (list): list of 5 elements, the name of the blur kernel (str), the noise level (str), the range of the noise standard deviation (list), 
                                       the image size (numpy array), minimal and maximal pixel values (list)
            folders            (list): list of str, paths to the folder containing (i) the test sets, (ii) the training, (iii) saved models
            mode                (str): 'first_layer' if training the first layer, 'greedy' if training the following layers one by one,
                                       'last_layers_lpp' if training the last 10 layers + lpp, 'test' if testing the model (default is 'first_layer')
            lr_first_layer     (list): list of two elements, first one is the initial learning rate to train the first layer, 
                                       second one is the number of epochs after which the learning rate is multiplied by 0.9 (default is [1e-2,5])    
            lr_greedy          (list): list of two elements, first one is the initial learning rate to train the layers, 
                                       second one is the number of epochs after which the learning rate is multiplied by 0.9 (default is [5e-3,5])
            lr_last_layers_lpp (list): list of two elements, first one is the initial learning rate to train the last 10 layers conjointly with lpp, 
                                       second one is the number of epochs after which the learning rate is multiplied by 0.9 (default is [1e-3,50])  
            nb_epochs          (list): list of three integers, number of epochs for training the first layer, the remaining layers, 
                                       and the last 10 layers + lpp, respectively (default is [40,40,600])      
            nb_blocks           (int): number of unfolded iterations (default is 40)    
            nb_greedy_blocks    (int): number of blocks trained in a greedy way (default is 30)
            batch_size         (list): list of three integers, number of images per batch for training, validation and testing, respectively (default is [10,10,1])                
            loss_type           (str): name of the training loss (default is 'SSIM')  
        """
        super(iRestNet_class, self).__init__()           
        # unpack information about test conditions and saving folders
        self.name_kernel, self.name_std, self.noise_std_range, self.im_size, im_range = test_conditions
        self.path_test, self.path_train, self.path_save                               = folders
        self.mode  = mode #'first_layer' or 'greedy' or 'last_layers_lpp' or 'test'
        # training information
        self.lr_first_layer     = lr_first_layer # learning rate, second nb indicates nb of epochs after which the learning rate is multiplied by 0.9
        self.lr_greedy          = lr_greedy
        self.lr_last_layers_lpp = lr_last_layers_lpp
        self.nb_epochs          = nb_epochs  # nb of epochs for the first layers, other layers trained in a greedy fashion, last layers+lpp
        self.nb_blocks          = nb_blocks
        self.nb_greedy_blocks   = nb_greedy_blocks
        self.batch_size         = batch_size # training set and validation set/test set 
        self.loss_type          = loss_type  # 'SSIM' or 'MSE'
        
        # load kernel and compute its transpose and its square (using the Fourier transform)
        self.kernel         = sio.loadmat(os.path.join(self.path_test,self.name_kernel+self.name_std,'kernel.mat'))
        kernel_t, kernel_2  = TransposeSquareFilter(self.kernel[self.name_kernel], self.im_size)
        self.kernel, kernel_t, self.kernel_2 = TensorFilter([self.kernel[self.name_kernel], kernel_t, kernel_2])

        if self.mode!='test':
            #definition of the loss function 
            if self.loss_type=='SSIM':
                self.loss_fun = SSIM_loss() 
            elif self.loss_type=='MSE':
                self.loss_fun = torch.nn.MSELoss(size_average=True)
          
        self.dtype        = torch.cuda.FloatTensor
        self.Ht           = MyConv2d(kernel_t, 'batch').cuda()
        self.model        = myModel(im_range,self.kernel_2,self.dtype,self.nb_blocks).cuda()
        self.last_layer   = myLastLayer().cuda()
        self.sigmoid      = nn.Sigmoid()
    
    def CreateLoader(self):
        """
        According to the mode, creates the appropriate loader for the training and validation sets.
        """
        if self.mode=='first_layer':
            #if the first layer is being trained, creates a loader loading pictures directly from the training set
            transf1     = transforms.Compose([transforms.RandomCrop(self.im_size),transforms.ToTensor()]) 
            transf2     = transforms.Compose([MyConv2d(self.kernel,'single').cuda(),add_Gaussian_noise(self.noise_std_range)])
            train_data  = MyDataset(folder=os.path.join(self.path_train,'train'), transf1=transf1, transf2=transf2, need_names='yes')
            val_data    = MyDataset(folder=os.path.join(self.path_train,'val'), transf1=transf1, transf2=transf2, need_names='yes')
        elif self.mode=='greedy' or self.mode=='last_layers_lpp':
            #else, creates a loader loading output of the previous layer
            folder_temp = os.path.join(self.path_save,'ImagesLastBlock')
            train_data  = MyDataset_OneBlock(
                folder_true           = os.path.join(folder_temp,'train','true'),                             
                folder_previous_block = os.path.join(folder_temp,'train','previous_block'),
                folder_Htxblurred     = os.path.join(folder_temp,'train','Htxblurred'),
                folder_std_approx     = os.path.join(folder_temp,'train','std_approx'))
            val_data  = MyDataset_OneBlock(
                folder_true           = os.path.join(folder_temp,'val','true'),                             
                folder_previous_block = os.path.join(folder_temp,'val','previous_block'),
                folder_Htxblurred     = os.path.join(folder_temp,'val','Htxblurred'),
                folder_std_approx     = os.path.join(folder_temp,'val','std_approx'))
        self.train_loader = DataLoader(train_data, batch_size=self.batch_size[0], shuffle=True)
        self.val_loader   = DataLoader(val_data, batch_size=self.batch_size[1], shuffle=False)
        self.size_train   = len([n for n in os.listdir(os.path.join(self.path_train,'train'))])
        self.size_val     = len([n for n in os.listdir(os.path.join(self.path_train,'val'))])
    
    def CreateFolders(self,block):
        """
        Creates directories for saving results.
        """
       
        if self.mode=='first_layer' or self.mode=='greedy':
            name = 'block_'+str(block)
            if not os.path.exists(os.path.join(self.path_save,name)):
                os.makedirs(os.path.join(self.path_save,name,'training'))
        elif self.mode=='last_layers_lpp':
            name = 'block_'+str(block)+'_'+str(self.nb_blocks-1)+'_lpp'
            if not os.path.exists(os.path.join(self.path_save,name)):
                os.makedirs(os.path.join(self.path_save,name,'training'))
        if self.mode!='test':
            folder = os.path.join(self.path_save,'ImagesLastBlock')
            if not os.path.exists(folder):
                subfolders    = ['train','val']
                subsubfolders = ['true','previous_block','Htxblurred','std_approx']
                paths         = [os.path.join(folder, sub, subsub) for sub in subfolders for subsub in subsubfolders]
                for path in paths:
                    os.makedirs(path)
        
    def train(self,block=0):
        """
        Trains iRestNet.
        Parameters
        ----------
            block (int): number of the layer to be trained, numbering starts at 0 (default is 0)
        """     
        if self.mode=='first_layer':
            # trains the first layer
            print('=================== Block number %d ==================='%(0))
            # to store results
            loss_epochs       =  np.zeros(self.nb_epochs[0])
            psnr_ssim_train   =  np.zeros((2,2,self.nb_epochs[0]))
            psnr_ssim_val     =  np.zeros((2,2,self.nb_epochs[0]))
            loss_min_val      =  float('Inf')
            self.CreateFolders(0)
            folder = os.path.join(self.path_save,'block_'+str(0))
            self.CreateLoader()
            # defines the optimizer
            lr        = self.lr_first_layer[0]
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,self.parameters()),lr=lr)
            #==========================================================================================================
            # trains for several epochs
            for epoch in range(0,self.nb_epochs[0]): 
                # sets training mode
                self.model.Layers[0].train()
                gc.collect()
                # modifies learning rate
                if epoch%self.lr_first_layer[1]==0 and epoch>0:
                    lr        = lr*0.9 
                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,self.parameters()), lr=lr)
                # goes through all minibatches
                for i,minibatch in enumerate(self.train_loader,0):
                    [names, x_true, x_blurred] = minibatch            # get the minibatch
                    x_true       = Variable(x_true.type(self.dtype),requires_grad=False)
                    x_blurred    = Variable(x_blurred.type(self.dtype),requires_grad=False)
                    Ht_x_blurred = self.Ht(x_blurred).detach()        # do not compute gradient
                    x_pred       = self.model(x_blurred,Ht_x_blurred,self.mode) 
                    
                    # Computes and prints loss
                    loss                = self.loss_fun(x_pred, x_true)
                    loss_epochs[epoch] += torch.Tensor.item(loss)
                    sys.stdout.write('\r(%d, %3d) minibatch loss: %5.4f '%(epoch,i,torch.Tensor.item(loss)))
                    
                    # sets the gradients to zero, performs a backward pass, and updates the weights.
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                    # for statistics
                    psnr_ssim_train[:,:,epoch] += compute_PSNR_SSIM(x_true, x_blurred, x_pred, self.size_train)

                # saves images and model state  
                if epoch%20==0:
                    utils.save_image(x_pred.data,os.path.join(
                        folder,'training',str(epoch)+'_restored_images.png'),normalize=True)
                torch.save(self.last_layer.state_dict(),os.path.join(folder,'trained_post-processing.pt'))
                torch.save(self.model.state_dict(),os.path.join(folder,'trained_model.pt'))

                # tests on validation set
                self.model.eval()      # evaluation mode
                self.last_layer.eval() # evaluation mode
                psnr_ssim = np.zeros((2,2))
                nb_im            = 0
                loss_current_val = 0
                for minibatch in self.val_loader:
                    [names, x_true, x_blurred] = minibatch            # gets the minibatch
                    x_true       = Variable(x_true.type(self.dtype),requires_grad=False)
                    x_blurred    = Variable(x_blurred.type(self.dtype),requires_grad=False)
                    Ht_x_blurred = self.Ht(x_blurred).detach()        # does not compute gradient
                    x_pred = self.model(x_blurred,Ht_x_blurred,self.mode) 
                    
                    # computes loss on validation set
                    loss_current_val += torch.Tensor.item(self.loss_fun(x_pred, x_true))
                    
                    #for statistics
                    psnr_ssim += compute_PSNR_SSIM(x_true, x_blurred, x_pred, self.size_val) # compute PSNR an SSIM
                if loss_min_val>loss_current_val:
                    torch.save(self.last_layer.state_dict(),os.path.join(folder,'trained_post-processing_MinLossOnVal.pt'))
                    torch.save(self.model.state_dict(),os.path.join(folder,'trained_model_MinLossOnVal.pt'))
                    loss_min_val = loss_current_val
                psnr_ssim_val[:,:,epoch] = psnr_ssim
                # prints statistics
                self.PrintStatistics(psnr_ssim_train[:,:,epoch], psnr_ssim_val[:,:,epoch], epoch, loss_epochs[epoch],lr)
                self.SaveLoss_PSNR_SSIM(epoch, loss_epochs, psnr_ssim_train, psnr_ssim_val,self.mode)
            #==========================================================================================================
            # training is finished
            print('-----------------------------------------------------------------')
            print('Training of Block 0 is done.')
            self.SaveLoss_PSNR_SSIM(epoch,loss_epochs, psnr_ssim_train, psnr_ssim_val, self.mode)
            self.save_OneBlock()
            print('-----------------------------------------------------------------')
            
            # calls the same function to start training of the next layer
            self.mode = 'greedy'
            self.train(block=1)
            
        ###############################################################################################################
        
        elif self.mode=='greedy':
            # trains the next layer
            print('=================== Block number %d ==================='%(block))
            # to store results
            loss_epochs       =  np.zeros(self.nb_epochs[1])
            psnr_ssim_train   =  np.zeros((2,2,self.nb_epochs[1]))
            psnr_ssim_val     =  np.zeros((2,2,self.nb_epochs[1]))
            loss_min_val      =  float('Inf')
            self.CreateFolders(block)
            folder = os.path.join(self.path_save,'block_'+str(block))
            self.CreateLoader()
            # puts first blocks in evaluation mode: gradient is not computed
            self.model.GradFalse(block,self.mode) 
            # defines the optimizer
            lr        = self.lr_greedy[0]
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,self.parameters()),lr=lr)
            #==========================================================================================================
            # trains for several epochs
            for epoch in range(0,self.nb_epochs[1]):
                self.model.Layers[block].train() # training mode
                gc.collect()
                # modifies learning rate
                if epoch%self.lr_greedy[1]==0 and epoch>0:
                    lr        = lr*0.9 
                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,self.parameters()), lr=lr)
                # goes through all minibatches
                for i,minibatch in enumerate(self.train_loader,0):
                    [names, x_true, x_blurred, Ht_x_blurred, std_approx] = minibatch           # gets the minibatch
                    x_true       = Variable(x_true.type(self.dtype),requires_grad=False)
                    x_blurred    = Variable(x_blurred.type(self.dtype),requires_grad=False)
                    Ht_x_blurred = Variable(Ht_x_blurred.type(self.dtype),requires_grad=False) # does not compute gradient
                    std_approx   = std_approx.type(self.dtype)
                    x_pred       = self.model(x_blurred,Ht_x_blurred,self.mode,block=block,std_approx=std_approx) 

                    # Computes and prints loss
                    loss                = self.loss_fun(x_pred, x_true)
                    loss_epochs[epoch] += torch.Tensor.item(loss)
                    sys.stdout.write('\r(%d, %3d) minibatch loss: %5.4f '%(epoch,i,torch.Tensor.item(loss)))
                    
                    # sets the gradients to zero, performs a backward pass, and updates the weights.
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                    # for statistics
                    psnr_ssim_train[:,:,epoch] += compute_PSNR_SSIM(x_true, x_blurred, x_pred, self.size_train)

                # saves images and model state  
                if epoch%20==0:
                    utils.save_image(x_pred.data,os.path.join(
                        folder,'training',str(epoch)+'_restored_images.png'),normalize=True)
                torch.save(self.last_layer.state_dict(),os.path.join(folder,'trained_post-processing.pt'))
                torch.save(self.model.state_dict(),os.path.join(folder,'trained_model.pt'))

                # tests on validation set
                self.model.eval()      # evaluation mode
                self.last_layer.eval() # evaluation mode
                psnr_ssim = np.zeros((2,2))
                nb_im            = 0
                loss_current_val = 0
                for minibatch in self.val_loader:
                    [names, x_true, x_blurred, Ht_x_blurred,std_approx] = minibatch            # gets the minibatch
                    x_true       = Variable(x_true.type(self.dtype),requires_grad=False)
                    x_blurred    = Variable(x_blurred.type(self.dtype),requires_grad=False)
                    Ht_x_blurred = Variable(Ht_x_blurred.type(self.dtype),requires_grad=False) # does not compute gradient
                    std_approx   = std_approx.type(self.dtype)
                    x_pred       = self.model(x_blurred,Ht_x_blurred,self.mode,block=block,std_approx=std_approx) 
                
                    # computes loss on validation set
                    loss_current_val += torch.Tensor.item(self.loss_fun(x_pred, x_true))
                    # for statistics
                    psnr_ssim += compute_PSNR_SSIM(x_true, x_blurred, x_pred, self.size_val) # computes PSNR an SSIM
                if loss_min_val>loss_current_val:
                    torch.save(self.last_layer.state_dict(),os.path.join(folder,'trained_post-processing_MinLossOnVal.pt'))
                    torch.save(self.model.state_dict(),os.path.join(folder,'trained_model_MinLossOnVal.pt'))
                    loss_min_val = loss_current_val
                psnr_ssim_val[:,:,epoch] = psnr_ssim
                # prints statistics
                self.PrintStatistics(psnr_ssim_train[:,:,epoch], psnr_ssim_val[:,:,epoch], epoch, loss_epochs[epoch],lr)
                self.SaveLoss_PSNR_SSIM(epoch, loss_epochs, psnr_ssim_train, psnr_ssim_val,self.mode,block=block)
            #==========================================================================================================
            # training is finished
            print('-----------------------------------------------------------------')
            print('Training of Block ' + str(block) + ' is done.')
            self.SaveLoss_PSNR_SSIM(epoch,loss_epochs, psnr_ssim_train, psnr_ssim_val,self.mode,block=block)
            self.save_OneBlock(block=block)
            print('-----------------------------------------------------------------')
            
            # calls the same function to start training of next block 
            if block==self.nb_greedy_blocks-1:
                self.mode = 'last_layers_lpp'
                self.train()
            else:   
                self.train(block=block+1)
        
        ###############################################################################################################
        
        elif self.mode=='last_layers_lpp':
            # trains the last 10 layers and the post-processing layer
            print('=================== Blocks %d to %d and lpp ==================='%(self.nb_greedy_blocks,self.nb_blocks-1))
            # to store results
            loss_epochs       =  np.zeros(self.nb_epochs[2])
            psnr_ssim_train   =  np.zeros((2,2,self.nb_epochs[2]))
            psnr_ssim_val     =  np.zeros((2,2,self.nb_epochs[2]))
            loss_min_val      =  float('Inf')
            self.CreateFolders(self.nb_greedy_blocks)
            folder = os.path.join(self.path_save,'block_'+str(self.nb_greedy_blocks)+'_'+str(self.nb_blocks-1)+'_lpp')
            self.CreateLoader()
            # puts first blocks in evaluation mode: gradient is not computed
            self.model.GradFalse(self.nb_greedy_blocks,self.mode) 
            # defines the optimizer
            lr        = self.lr_last_layers_lpp[0]
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,self.parameters()),lr=lr)
            #==============================================================================================
            # trains for several epochs
            for epoch in range(0,self.nb_epochs[2]):
                for k in range(self.nb_greedy_blocks,self.nb_blocks):
                    self.model.Layers[k].train() #training mode
                self.last_layer.train() #training mode
                gc.collect()
                # modifies learning rate
                if epoch%self.lr_last_layers_lpp[1]==0 and epoch>0:
                    lr = lr*0.9 
                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,self.parameters()), lr=lr)
                # goes through all minibatches
                for i,minibatch in enumerate(self.train_loader,0):
                    [names, x_true, x_blurred, Ht_x_blurred, std_approx] = minibatch           # gets the minibatch
                    x_true       = Variable(x_true.type(self.dtype),requires_grad=False)
                    x_blurred    = Variable(x_blurred.type(self.dtype),requires_grad=False)
                    Ht_x_blurred = Variable(Ht_x_blurred.type(self.dtype),requires_grad=False) # does not compute gradient
                    std_approx   = std_approx.type(self.dtype)
                    x_last_block = self.model(x_blurred,Ht_x_blurred,self.mode,block=self.nb_greedy_blocks,std_approx=std_approx) 
                    x_pred       = self.sigmoid(x_last_block + self.last_layer(x_last_block))  # Forward: just last layer
                
                    # Computes and prints loss
                    loss                = self.loss_fun(x_pred, x_true)
                    loss_epochs[epoch] += torch.Tensor.item(loss)
                    sys.stdout.write('\r(%d, %3d) minibatch loss: %5.4f '%(epoch,i,torch.Tensor.item(loss)))
                    
                    # sets the gradients to zero, performs a backward pass, and updates the weights.
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                    # for statistics
                    psnr_ssim_train[:,:,epoch] += compute_PSNR_SSIM(x_true, x_blurred, x_pred, self.size_train)

                # saves images and model state  
                if epoch%20==0:
                    utils.save_image(x_pred.data,os.path.join(
                        folder,'training',str(epoch)+'_restored_images.png'),normalize=True)
                torch.save(self.last_layer.state_dict(),os.path.join(folder,'trained_post-processing.pt'))
                torch.save(self.model.state_dict(),os.path.join(folder,'trained_model.pt'))

                # tests on validation set
                self.model.eval()      # evaluation mode
                self.last_layer.eval() # evaluation mode
                psnr_ssim = np.zeros((2,2))
                nb_im            = 0
                loss_current_val = 0
                for minibatch in self.val_loader:
                    [names, x_true, x_blurred, Ht_x_blurred,std_approx] = minibatch            # gets the minibatch
                    x_true       = Variable(x_true.type(self.dtype),requires_grad=False)
                    x_blurred    = Variable(x_blurred.type(self.dtype),requires_grad=False)
                    Ht_x_blurred = Variable(Ht_x_blurred.type(self.dtype),requires_grad=False) # does not compute gradient
                    std_approx   = std_approx.type(self.dtype)
                    x_last_block = self.model(x_blurred,Ht_x_blurred,self.mode,block=self.nb_greedy_blocks,std_approx=std_approx) 
                    x_pred       = self.sigmoid(x_last_block + self.last_layer(x_last_block))  # post-processing layer
                
                    # computes loss on validation set
                    loss_current_val += torch.Tensor.item(self.loss_fun(x_pred, x_true))
                    
                    # for statistics
                    psnr_ssim += compute_PSNR_SSIM(x_true, x_blurred, x_pred, self.size_val) # computes PSNR an SSIM
                if loss_min_val>loss_current_val:
                    torch.save(self.last_layer.state_dict(),os.path.join(folder,'trained_post-processing_MinLossOnVal.pt'))
                    torch.save(self.model.state_dict(),os.path.join(folder,'trained_model_MinLossOnVal.pt'))
                    loss_min_val = loss_current_val
                psnr_ssim_val[:,:,epoch] = psnr_ssim
                # print statistics
                self.PrintStatistics(psnr_ssim_train[:,:,epoch], psnr_ssim_val[:,:,epoch], epoch, loss_epochs[epoch],lr)
                self.SaveLoss_PSNR_SSIM(epoch, loss_epochs, psnr_ssim_train, psnr_ssim_val,self.mode)
        
            # training is finished
            print('-----------------------------------------------------------------')
            print('Training of last layers and lpp is done.')
            self.SaveLoss_PSNR_SSIM(epoch,loss_epochs, psnr_ssim_train, psnr_ssim_val,self.mode)
            print('-----------------------------------------------------------------')
            return

    def save_OneBlock(self,block=0): 
        """
        Saves the outputs of the current layer.
        Parameters
        ----------
            block (int): number of the layer to be trained, numbering starts at 0 (default is 0)    
        """
        self.model.eval() #evaluation mode   
        # Haar filter for estimating noise standard deviation
        Haar_filt = TensorFilter([np.array(((0.5,-0.5),(-0.5,0.5)))])[0]
        Haar      = MyConv2d(Haar_filt,'batch',pad_type='circular',padding=1,stride=2).cuda()
        folder    = os.path.join(self.path_save,'ImagesLastBlock')
           
        for minibatch in self.train_loader:
            if self.mode=='first_layer':
                [names, x_true, x_blurred] = minibatch     # gets the minibatch
                x_true       = Variable(x_true.type(self.dtype),requires_grad=False)
                x_blurred    = Variable(x_blurred.type(self.dtype),requires_grad=False)
                Ht_x_blurred = self.Ht(x_blurred).detach() # does not compute gradient
                x_pred       = self.model(x_blurred,Ht_x_blurred,self.mode) # Forward
                # computes approximation of noise std
                y          = torch.abs(Haar(x_blurred)).view(x_blurred.data.shape[0],-1).data/0.6745
                std_approx = torch.topk(y,ceil(y.shape[1]/2),1)[0][:,-1]
                for kk in range(len(names)):
                    sio.savemat(os.path.join(folder,'train','true',names[kk]),{
                         'image':x_true.data[kk].permute(1,2,0).cpu().numpy().astype('float64')})
                    sio.savemat(os.path.join(folder,'train','previous_block',names[kk]),{
                         'image':x_pred.data[kk].permute(1,2,0).cpu().numpy().astype('float64')})
                    sio.savemat(os.path.join(folder,'train','Htxblurred',names[kk]),{
                         'image':Ht_x_blurred.data[kk].permute(1,2,0).cpu().numpy().astype('float64')})
                    np.savetxt(os.path.join(folder,'train','std_approx',names[kk]+'.txt'),np.array([std_approx[kk].cpu().numpy()]))
            else:
                [names, x_true, x_blurred, Ht_x_blurred, std_approx] = minibatch                              # gets the minibatch
                x_true       = Variable(x_true.type(self.dtype),requires_grad=False)
                x_blurred    = Variable(x_blurred.type(self.dtype),requires_grad=False)
                Ht_x_blurred = Variable(Ht_x_blurred.type(self.dtype),requires_grad=False) 
                std_approx   = std_approx.type(self.dtype)
                x_pred       = self.model(x_blurred,Ht_x_blurred,self.mode,block=block,std_approx=std_approx) # Forward
                for kk in range(len(names)):
                     sio.savemat(os.path.join(folder,'train','previous_block',names[kk]),{
                         'image':x_pred.data[kk].permute(1,2,0).cpu().numpy().astype('float64')})
        # validation set
        for minibatch in self.val_loader:
            if self.mode=='first_layer':
                [names, x_true, x_blurred] = minibatch                      # gets the minibatch
                x_true       = Variable(x_true.type(self.dtype),requires_grad=False)
                x_blurred    = Variable(x_blurred.type(self.dtype),requires_grad=False)
                Ht_x_blurred = self.Ht(x_blurred).detach()                  # does not compute gradient
                x_pred       = self.model(x_blurred,Ht_x_blurred,self.mode) # Forward
                # computes aproximation of the noise standard deviation
                y          = torch.abs(Haar(x_blurred)).view(x_blurred.data.shape[0],-1).data/0.6745
                std_approx = torch.topk(y,ceil(y.shape[1]/2),1)[0][:,-1]
                for kk in range(len(names)):
                    sio.savemat(os.path.join(folder,'val','true',names[kk]),{
                         'image':x_true.data[kk].permute(1,2,0).cpu().numpy().astype('float64')})
                    sio.savemat(os.path.join(folder,'val','previous_block',names[kk]),{
                         'image':x_pred.data[kk].permute(1,2,0).cpu().numpy().astype('float64')})
                    sio.savemat(os.path.join(folder,'val','Htxblurred',names[kk]),{
                         'image':Ht_x_blurred.data[kk].permute(1,2,0).cpu().numpy().astype('float64')})
                    np.savetxt(os.path.join(folder,'val','std_approx',names[kk]+'.txt'),np.array([std_approx[kk].cpu().numpy()]))
            else:
                [names, x_true, x_blurred, Ht_x_blurred, std_approx] = minibatch # gets the minibatch
                x_true       = Variable(x_true.type(self.dtype),requires_grad=False)
                x_blurred    = Variable(x_blurred.type(self.dtype),requires_grad=False)
                Ht_x_blurred = Variable(Ht_x_blurred.type(self.dtype),requires_grad=False)    
                std_approx   = std_approx.type(self.dtype)
                x_pred       = self.model(x_blurred,Ht_x_blurred,self.mode,block=block,std_approx=std_approx) # Forward
                for kk in range(len(names)):
                     sio.savemat(os.path.join(folder,'val','previous_block',names[kk]),{
                         'image':x_pred.data[kk].permute(1,2,0).cpu().numpy().astype('float64')})
    
    def test(self, dataset, save_gamma_mu_lambda='no'):    
        """
        Parameters
        ----------
            dataset        (str): name of the test set
            save_gamma_mu_lambda: indicates if the user wants to save the values of the estimated hyperparameters (default is 'no')
        """
        # for RGB and gray images        
        path_savetest                   = os.path.join(self.path_save,'Results_on_Testsets',dataset)
        path_dataset                    = os.path.join(self.path_test, self.name_kernel+self.name_std, dataset)
        if save_gamma_mu_lambda=='no':
            print('Saving restaured images in %s ...'%(path_savetest),flush=True)
            # creates directory for saving results
            if not os.path.exists(path_savetest):
                os.makedirs(path_savetest)
            data          = MyTestset(folder=path_dataset)
            loader        = DataLoader(data, batch_size=self.batch_size[2], shuffle=False)
            # evaluation mode
            self.model.eval() 
            self.last_layer.eval()
            for minibatch in tqdm(loader,file=sys.stdout):
                [im_names, [yes_no, x_blurred]] = minibatch # gets the minibatch
                x_blurred     = Variable(x_blurred.type(self.dtype), requires_grad=False)
                Ht_x_blurred  = self.Ht(x_blurred)       
                x_pred        = self.model(x_blurred,Ht_x_blurred.detach(),self.mode)
                x_pred        = x_pred.detach()
                x_pred        = self.sigmoid(x_pred + self.last_layer(x_pred))
                # saves restored images
                for j in range(len(im_names)):
                    sio.savemat(os.path.join(path_savetest,im_names[j]),{'image':RGBToGray(
                        yes_no[j],x_pred.data[j]).permute(1,2,0).cpu().numpy().astype('float64')})
        else:
            print('Saving restaured images in %s ...'%(path_savetest),flush=True)
            print('Saving stepsize, barrier parameter and regularization parameter in %s ...'%(save_gamma_mu_lambda),flush=True)
            data          = MyTestset(folder=path_dataset)
            loader        = DataLoader(data, batch_size=1, shuffle=False)
            # evaluation mode
            self.model.eval() 
            self.last_layer.eval()
            for minibatch in tqdm(loader,file=sys.stdout):
                [im_names, [yes_no, x_blurred]] = minibatch # gets the minibatch
                path_gamma_mu_lambda = os.path.join(save_gamma_mu_lambda,im_names[0][0:-4])
                if not os.path.exists(path_gamma_mu_lambda):
                    os.makedirs(path_gamma_mu_lambda)
                x_blurred     = Variable(x_blurred.type(self.dtype), requires_grad=False)
                Ht_x_blurred  = self.Ht(x_blurred)       
                x_pred        = self.model(x_blurred,Ht_x_blurred.detach(),self.mode,save_gamma_mu_lambda=path_gamma_mu_lambda)
                
        
    def PrintStatistics(self, train, val, epoch, loss, lr):
        """
        Prints information about the training.
        Parameters
        ----------
        train (list): size 2*2, average PSNR and SSIM on the training set and on the deblurred training images
        val   (list): size 2*2, average PSNR and SSIM on the validation set and on the deblurred validation images
        epoch  (int): epoch number
        loss (float): value of the training loss function
        lr   (float): learning rate
        """
        print('-----------------------------------------------------------------')
        print('[%d]'%(epoch),'average', self.loss_type,': %5.5f'%(loss), 'lr %.2E'%(lr))
        print('     Training set:') 
        print('         PSNR blurred = %2.3f, PSNR pred = %2.3f'%(train[0,0],train[0,1]))
        print('         SSIM blurred = %2.3f, SSIM pred = %2.3f'%(train[1,0],train[1,1]))
        print('     Validation set:') 
        print('         PSNR blurred = %2.3f, PSNR pred = %2.3f'%(val[0,0],val[0,1]))
        print('         SSIM blurred = %2.3f, SSIM pred = %2.3f'%(val[1,0],val[1,1]))
    
    def SaveLoss_PSNR_SSIM(self, epoch, loss_epochs, psnr_ssim_train, psnr_ssim_val, mode, block=0):
        """
        Plots and saves training results.
        Parameters
        ----------
        epoch            (int): epoch number
        loss_epochs     (list): value of the loss function at each epoch
        psnr_ssim_train (list): average PSNR and SSIM on the blurred training set and on the deblurred training images, size 2*2*epoch 
        psnr_ssim_val   (list): average PSNR and SSIM on the blurred validation set and on the deblurred validation images, size 2*2*epoch 
        mode             (str): 'first_layer' if training the first layer, 'greedy' if training the following layers one by one,
                                'last_layers_lpp' if training the last 10 layers + lpp, 'test' if testing the model 
        block            (int): number of the layer to be trained, numbering starts at 0 (default is 0)
        """
        # plot and save loss for all epochs
        fig,(ax_loss) = plt.subplots(1,1,figsize=(4, 4))
        ax_loss.plot(loss_epochs[0:epoch+1])
        ax_loss.set_title('Minimal loss\n'+ "%5.2f" % np.min(loss_epochs))
        if mode=='first_layer' or mode=='greedy':
            name_temp = 'block_'+str(block)
        elif mode=='last_layers_lpp':
            name_temp = 'block_'+str(self.nb_greedy_blocks)+'_'+str(self.nb_blocks-1)+'_lpp'
        fig_name = os.path.join(self.path_save,name_temp,'training',"loss.png")
        plt.savefig(fig_name)
        plt.close(fig)
        # plot and save PSNR on training set for all epochs
        self.MyPlot('max PSNR', psnr_ssim_train[0,:,0:epoch+1],'psnr_training_set.png',block,mode)
        self.MyPlot_JustRestored('max PSNR', psnr_ssim_train[0,1,0:epoch+1], 'psnr_training_set_zoom.png',block,mode)
        # plot and save PSNR on validation set for all epochs
        self.MyPlot('Max PSNR', psnr_ssim_val[0,:,0:epoch+1],'psnr_validation_set.png',block,mode)
        self.MyPlot_JustRestored('Max PSNR', psnr_ssim_val[0,1,0:epoch+1],'psnr_validation_set_zoom.png',block,mode)
        # plot and save SSIM on training set for all epochs
        self.MyPlot('Max SSIM', psnr_ssim_train[1,:,0:epoch+1],'ssim_training_set.png',block,mode)
        self.MyPlot_JustRestored('Max SSIM', psnr_ssim_train[1,1,0:epoch+1],'ssim_training_set_zoom.png',block,mode,psnr=0)
        # plot and save SSIM on validation set for all epochs
        self.MyPlot('Max SSIM', psnr_ssim_val[1,:,0:epoch+1],'ssim_validation_set.png',block,mode)
        self.MyPlot_JustRestored('Max SSIM', psnr_ssim_val[1,1,0:epoch+1],'ssim_validation_set_zoom.png',block,mode,psnr=0)
        print('Plots of the training loss, PSNR and SSIM during training are saved.')
    
    def MyPlot(self, title, vec, name,block,mode):
        """
        Plots and save the SSIM or PSNR during training before and after deblurring.
        Parameters
        ----------
        title  (str): figure title
        vec   (list): average PSNR or SSIM on the training or validation set and on the deblurred images, size 2*epoch
        name   (str): figure name
        block  (int): number of the layer to be trained, numbering starts at 0 (default is 0)
        mode   (str): 'first_layer' if training the first layer, 'greedy' if training the following layers one by one,
                      'last_layers_lpp' if training the last 10 layers + lpp, 'test' if testing the model 
        """
        fig, ax = plt.figure(), plt.subplot(111)
        ax.plot(vec[0,:],'b',label='Blurred')
        ax.plot(vec[1,:],'g',label='Restored')
        ax.set_title(title + "%3.3f" %np.max(vec[1,:]))
        if mode=='first_layer' or mode=='greedy':
            name_temp = 'block_'+str(block)
        elif mode=='last_layers_lpp':
            name_temp = 'block_'+str(self.nb_greedy_blocks)+'_'+str(self.nb_blocks-1)+'_lpp'
        fig_name        = os.path.join(self.path_save,name_temp,'training',name)
        handles, labels = ax.get_legend_handles_labels()
        lgd             = ax.legend(handles, labels, loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        fig.savefig(fig_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close(fig)
        
    def MyPlot_JustRestored(self, title, vec, name,block,mode,psnr=1):
        """
        Plots and save the SSIM or PSNR during training after deblurring.
        Parameters
        ----------
        title  (str): figure title
        vec   (list): average PSNR or SSIM of deblurred images, length is number of epochs
        name   (str): figure name
        block  (int): number of the layer to be trained, numbering starts at 0 (default is 0)
        mode   (str): 'first_layer' if training the first layer, 'greedy' if training the following layers one by one,
                      'last_layers_lpp' if training the last 10 layers + lpp, 'test' if testing the model 
        psnr   (int): indicates if vec is the list of average PSNR (psnr=1) or SSIM (psnr=0) values (default is 1)
        """
        fig, ax = plt.figure(), plt.subplot(111)
        ax.plot(vec,'g',label='Restored')
        ax.set_title(title + "%3.3f" %np.max(vec))
        if mode=='first_layer' or mode=='greedy':
            name_temp = 'block_'+str(block)
        elif mode=='last_layers_lpp':
            if psnr==1:
                ax.set_ylim([30,31.65])
            else:
                ax.set_ylim([0.89,0.91])
            name_temp = 'block_'+str(self.nb_greedy_blocks)+'_'+str(self.nb_blocks-1)+'_lpp'
        fig_name        = os.path.join(self.path_save,name_temp,'training',name)
        handles, labels = ax.get_legend_handles_labels()
        lgd             = ax.legend(handles, labels, loc=2, bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        fig.savefig(fig_name, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close(fig)