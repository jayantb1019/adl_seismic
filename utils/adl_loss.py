import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

MAE_obj = nn.L1Loss()

def Loss_L1(x_gt, yhat) :
    return MAE_obj.__call__(x_gt, yhat)

def Loss_PYR(x_gt, yhat, levels=3) : 
    '''Pyramid Loss
    A pyramidal loss function using ATW. 
    ATW is a stationary Wavelet transform that decomposes an image into several levels 
    by a cubic spline filter and then subtracts any two successive layers to obtain 
    fine images with edges and texture. 
    The low-pass filters in ATW alleviate the side effects of noise so it enables us to 
    include texture information of input images in the loss function.
    
    In short, it decomposes an input image into J levels by a cubic spline finite impulse
    response that is denoted by h(1). Unlike the nonstationary multiscale transforms that
    downscale the images and then apply a filter, ATW upscales the kernel by inserting 2**(j-1) - 1 
    zeros between each pair of adjacent elements in h(1), where j denotes the j-th 
    decomposition level. Fine images with texture and edges are derived via subtraction 
    of any two successive filtered images
    
    '''
    # print(x_gt)
    
    def atw_kernel(ker_base, image_dtype, Cin, level=1) : # level = J 
        zeros_len = - 1 + 2**(level-1) # 2^(J-1) -1 zeros 
        ker_len = zeros_len * 4 + len(ker_base)
        kernel_1d = np.zeros((ker_len,)) 
        kernel_1d[::zeros_len+1] = ker_base/np.sum(ker_base)
        kernel_2d = np.tensordot(kernel_1d, np.transpose(kernel_1d), axes=0)  # kernel_2d = kernel_1d dot kernel_1d
        
        # convert to tensor 
        kernel_size = kernel_2d.shape[0]
        kernel_torch = torch.tensor(kernel_2d, dtype=image_dtype).unsqueeze(0).expand(Cin, 1, kernel_size, kernel_size)
        return  kernel_torch, kernel_size
    
    
    def _convolve(image, ker_base, level): # 1x1 convolution
        # get filter
        Cin = image.size(1)
        kernel_torch, pad_sz = atw_kernel(ker_base, image.dtype, Cin, level)

        # apply convolution
        output = F.conv2d(image, kernel_torch.to(image.get_device()), stride=1, padding=int(pad_sz/2), groups=Cin)
        return output
        
    B = x_gt.size()[0] # batch size
    
    per_batch_loss = torch.Tensor([0.])
    ker_base = [0.002566, 0.1655, 0.6638, 0.1655, 0.002566]
    
    x_blur = _convolve(x_gt, ker_base, 1)
    y_blur = _convolve(yhat, ker_base, 1)
    
    ker_base = [1., 4., 6., 4., 1.]
    
    for i in range(1, levels+1):
        x_blur_cur = _convolve(x_blur, ker_base, i)
        y_blur_cur = _convolve(y_blur, ker_base, i)

        # get detail
        Di_x = x_blur - x_blur_cur
        Di_y = y_blur - y_blur_cur

        # update x, y
        x_blur = x_blur_cur
        y_blur = y_blur_cur

        x_ravel = torch.reshape(Di_x, [B,-1])
        y_ravel = torch.reshape(Di_y, [B,-1])
        per_batch_loss = MAE_obj(x_ravel, y_ravel)

    return per_batch_loss
    
    
def log_cosh_torch(x_gt, y_pred):
    
    def _logcosh(x):
        return x + torch.nn.functional.softplus(-2.*x, beta=1, threshold=10) - torch.log(torch.tensor(2.)).to(x.dtype)
           
    return torch.mean(_logcosh(x_gt - y_pred))

def Loss_Hist(x_gt, y_pred):
    '''
    The histogram loss assures that the histograms of G(y) and x are close to each other. 
    This term maintains the global structure of G(y) with respect to x since the added edges 
    and texture information (by ATW) may change the overall histogram of the denoised instance.
    To compute this loss, we first compute the histogram of both G(y) and x denoted by H. 
    Then, we use an strictly increasing function that computes the loss between H [G(y)] and H [x].
    
    Histogram loss preserves the global structure of the image.
    '''

    B = x_gt.shape[0]
    n_channels = x_gt.shape[1]
    N_pixels = x_gt.shape[2]*x_gt.shape[3]
    
    x_gt = torch.clamp(x_gt,-1,1)  # changed from 0,1 , for seismic signals
    y_pred = torch.clamp(y_pred,-1,1)  # changed from 0,1 , for seismic signals
    
    #change it to a function
    loss_all = 0.
    for b in range(B):
        for i in range(n_channels):
            x_hist = torch.histc(x_gt[b,i,:,:], bins=500, min=-1, max=1.)  # increased the bins from 256 to 500 , changed min and max to -1,1
            y_hist = torch.histc(y_pred[b,i,:,:], bins=500, min=-1, max=1.) # increased the bins from 256 to 500 , changed min and max to -1,1
            loss_all += log_cosh_torch(x_hist, y_hist)
             
    return  loss_all/torch.prod(torch.tensor(x_gt.size()))


        