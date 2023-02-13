
import numpy as np


def fspectra(data, dt = 4) : 
        ''''
        Calculating the average real amplitude frequency spectrum of the noisy and denoised data

        Input Parameters
        ===============
        data : ndarray 
                noisy, clean, denoised data for which average amplitude spectrum to be calculated
        dt   : sampling interval 
                in milliseconds
        
        Output Parameters 
        =================
        f    : Frequency ticks
        a    : real amplitude fourier coefficients
        
        '''

        # Absolute values of fourier coefficients 
        fc = np.abs(np.fft.rfft(data, axis=-1))
        
        a = np.mean(fc, axis=0) # mean

        dts = dt / 1000 # sampling freq in seconds

        length = data.shape[-1]

        f = np.fft.rfftfreq(length, d = dts)

        return f, a