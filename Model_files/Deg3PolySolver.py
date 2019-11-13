from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import sys

class cardan(torch.autograd.Function):  

    @staticmethod
    def forward(ctx,gamma_mu,xtilde,im_range,mode_training=True):
        """
	    Finds the solution of the cubic equation involved in the computation of the proximity operator of the 
        logarithmic barrier of the hyperslab constraints (xmin<x<xmax) using the Cardano formula: ax^3+bx^2+cx+d=0 
        is rewritten as x^3+px+q=0. Selects the solution x such that x-a/3 is real and belongs to ]xmin,xmax[.
        Parameters
        ----------
           gamma_mu (torch.FloatTensor): product of the barrier parameter and the stepsize, size n
           xtilde (torch.FloatTensor): point at which the proximity operator is applied, size n
           im_range (list): minimal and maximal pixel values
           mode_training (bool): indicates if the model is in training (True) or testing (False) (default is True)
        Returns
        -------
           sol (torch.FloatTensor): proximity operator of gamma_mu*barrier at xtilde, size n 
        """
        #initialize variables
        dtype             = torch.cuda.FloatTensor
        size              = xtilde.size()
        x1,x2,x3          = torch.zeros(size).type(dtype),torch.zeros(size).type(dtype),torch.zeros(size).type(dtype)   
        crit,crit_compare = torch.zeros(size).type(dtype),torch.zeros(size).type(dtype)
        sol               = torch.zeros(size).type(dtype),
        torch_one         = torch.ones(size).type(dtype)
        xmin,xmax         = im_range
        #set coefficients
        a     = -(xmin+xmax+xtilde)
        b     = xmin*xmax + xtilde*(xmin+xmax) - 2*gamma_mu
        c     = gamma_mu*(xmin+xmax) - xtilde*xmin*xmax
        p     = b - (a**2)/3
        q     = c - a*b/3 + 2*(a**3)/27
        delta = (p/3)**3 + (q/2)**2  

        #three cases depending on the sign of delta
        #########################################################################
        #when delta is positive
        ind     = delta>0
        z1      = -q[ind]/2
        z2      = torch.sqrt(delta[ind])
        u       = (z1+z2).sign() * torch.pow((z1+z2).abs(),1/3)
        v       = (z1-z2).sign() * torch.pow((z1-z2).abs(),1/3) 
        x1[ind] = u+v   
        x2[ind] = -(u + v)/2 ; #real part of the complex solution
        x3[ind] = -(u + v)/2 ; #real part of the complex solution
        #########################################################################
        #when delta is 0
        ind     = delta==0
        x1[ind] = 3 *q[ind] / p[ind]  
        x2[ind] = -1.5 * q[ind] / p[ind] 
        x3[ind] = -1.5 * q[ind] / p[ind] 
        #########################################################################
        #when delta is negative
        ind         = delta<0
        cos         = (-q[ind]/2) * ((27 / torch.pow(p[ind],3)).abs()).sqrt() 
        cos[cos<-1] = 0*cos[cos<-1]-1
        cos[cos>1]  = 0*cos[cos>1]+1
        phi         = torch.acos(cos)
        tau         = 2 * ((p[ind]/3).abs()).sqrt() 
        x1[ind]     = tau * torch.cos(phi/3) 
        x2[ind]     = -tau * torch.cos((phi + np.pi)/3)
        x3[ind]     = -tau * torch.cos((phi - np.pi)/3)
        #########################################################################
        x1   = x1-a/3
        x2   = x2-a/3
        x3   = x3-a/3
        # when gamma_mu is very small there might be some numerical instabilities
        # in case there are nan values, we set the corresponding pixels equal to 2*xmax
        # these values will be replaced by valid values at least once (line 99 or 107 or 114 or 119 or 129)
        if (x1!=x1).any():
            x1[x1!=x1]=2*xmax
        if (x2!=x2).any():
            x2[x2!=x2]=2*xmax
        if (x3!=x3).any():
            x3[x3!=x3]=2*xmax
        sol  = x1
        #########################################################################
        #take x1
        x_ok         = (x1>xmin)&(x1<xmax)
        crit[1-x_ok] = np.inf
        crit[x_ok]   = -(torch.log(x1[x_ok]-xmin)+torch.log(xmax-x1[x_ok]))
        crit         = 0.5*(x1-xtilde)**2+gamma_mu*crit
        #########################################################################
        #test x2
        x_ok                     = (x2>xmin)&(x2<xmax)
        crit_compare[1-x_ok]     = np.inf
        crit_compare[x_ok]       = -(torch.log(x2[x_ok]-xmin)+torch.log(xmax-x2[x_ok]))
        crit_compare             = 0.5*(x2-xtilde)**2+gamma_mu*crit_compare
        sol[crit_compare<=crit]  = x2[crit_compare<=crit]
        crit[crit_compare<=crit] = crit_compare[crit_compare<=crit]
        #########################################################################
        #test x3
        x_ok                     = (x3>xmin)&(x3<xmax)
        crit_compare[1-x_ok]     = np.inf
        crit_compare[x_ok]       = -(torch.log(x3[x_ok]-xmin)+torch.log(xmax-x3[x_ok]))
        crit_compare             = 0.5*(x3-xtilde)**2+gamma_mu*crit_compare
        sol[crit_compare<=crit]  = x3[crit_compare<=crit]
        crit[crit_compare<=crit] = crit_compare[crit_compare<=crit]
        #########################################################################
        #test xmin+1e-10
        crit_compare             = (0.5*(xmin+1e-10-xtilde)**2)*torch_one-gamma_mu*(
            torch.log(1e-10*torch_one)+torch.log((xmax-xmin-1e-10)*torch_one))
        sol[crit_compare<=crit]  = 0*sol[crit_compare<=crit]+(xmin+1e-10)
        crit[crit_compare<=crit] = crit_compare[crit_compare<=crit]
        #########################################################################
        #test xmax-1e-10
        crit_compare             = (0.5*(xmax-1e-10-xtilde)**2)*torch_one-gamma_mu*(
            torch.log(1e-10*torch_one)+torch.log((xmax-xmin-1e-10)*torch_one))
        sol[crit_compare<=crit]  = 0*sol[crit_compare<=crit]+(xmax-1e-10)
        crit[crit_compare<=crit] = crit_compare[crit_compare<=crit]
        #########################################################################
        # when gamma_mu is very small and xtilde is very close to one of the bounds,
        # the solution of the cubic equation is not very well estimated -> test xtilde
        #denom       = (sol-xmin)*(sol-xmax)-2*gamma_mu -(sol-xtilde)*(xmin+xmax-2*sol)
        xtilde_ok                 = (xtilde>xmin)&(xtilde<xmax)
        crit_compare[1-xtilde_ok] = np.inf
        crit_compare[xtilde_ok]   = -(torch.log(xmax-xtilde[xtilde_ok])+torch.log(xtilde[xtilde_ok]-xmin))
        crit_compare              = gamma_mu*crit_compare
        sol[crit_compare<crit]    = xtilde[crit_compare<crit]
        
        if mode_training==True:
            ctx.save_for_backward(gamma_mu,xtilde,sol)
        return sol

    @staticmethod
    def backward(ctx, grad_output_var):
        """
        Computes the first derivatives of the proximity operator of the log barrier with respect to x and gamma_mu.
            This method is automatically called by the backward method of the loss function.
        Parameters
        ----------
           ctx (list): list of torch.FloatTensors, variable saved during the forward operation
           grad_output_var (torch.FloatTensor): gradient of the loss wrt the output of cardan
        Returns
        -------
           grad_input_gamma_mu (torch.FloatTensor): gradient of the prox wrt gamma_m 
           grad_input_u (torch.FloatTensor): gradient of the prox wrt x
           None: no gradient wrt the image range
           None: no gradient wrt the mode
        """
        xmin           = 0
        xmax           = 1
        dtype          = torch.cuda.FloatTensor
        grad_output    = grad_output_var.data
        gamma_mu,u,x   = ctx.saved_tensors
        denom          = (x-xmin)*(x-xmax)-2*gamma_mu -(x-u)*(xmin+xmax-2*x)
        
        idx                 = denom.abs()>1e-7
        denom[1-idx]        = denom[1-idx]+1
        grad_input_gamma_mu = (2*x-(xmin+xmax))/denom
        grad_input_u        = ((x**2-x*(xmin+xmax)+xmin*xmax))/denom
        # if denom is very small, it means that gamma_mu is very small and u is very close to one of the bounds,
        # there is a discontinuity when gamma_mu tends to zero, if 0<u<1 the derivative wrt x is approximately equal to 
        # 1 and the derivative wrt gamma_mu is approximated by 10^5 times the sign of 2*x[1-idx]-(xmin+xmax)
        grad_input_gamma_mu[1-idx] = 0*grad_input_gamma_mu[1-idx]+1e5*torch.sign(2*x[1-idx]-(xmin+xmax))
        grad_input_u[1-idx]        = 0*grad_input_u[1-idx]+1
        
        grad_input_gamma_mu = (grad_input_gamma_mu*grad_output).sum(1).sum(1).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        grad_input_u        = grad_input_u*grad_output
        
        # safety check for numerical instabilities
        if (grad_input_gamma_mu!=grad_input_gamma_mu).any():
            print('there is a nan in grad_input_gamma_mu')
            if (x!=x).any():
                print('there is a nan in x')
            sys.exit()
        if (grad_input_u!=grad_input_u).any():
            print('there is a nan in grad_input_u')
            sys.exit()
        
        grad_input_gamma_mu = Variable(grad_input_gamma_mu.type(dtype),requires_grad=True)
        grad_input_u        = Variable(grad_input_u.type(dtype),requires_grad=True)
        
        return grad_input_gamma_mu, grad_input_u, None, None