import torch

from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure


class MetricEval(object) : 
    '''
    PNSR : Replaced original with skimage version 
    '''
    
    @staticmethod
    def psnr(gt, y_pred, device = torch.device('cpu')) -> torch.Tensor : 
        
        B = gt.shape[0]
        return  peak_signal_noise_ratio(y_pred, gt) / B
    
    @staticmethod
    def ssim(gt, y_pred , device, data_range) -> torch.Tensor : 
        
        return  structural_similarity_index_measure(y_pred, gt, reduction='elementwise_mean', kernel_size=11,)
         