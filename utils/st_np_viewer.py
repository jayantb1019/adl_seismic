import streamlit as st 
import matplotlib.pyplot as plt 
from argparse import ArgumentParser 

import numpy as np 


parser = ArgumentParser()

parser.add_argument('-file', type=str, default=None)
pargs = parser.parse_args()

st.title('Numpy Array Viewer')

if pargs.file : 
    nparray = np.load(pargs.file)

    if len(nparray.shape) > 2 : 
        dims = nparray.shape 
        if dims == 3 : 
            nparray = nparray[0,:,:]
        if dims ==4 : 
            nparray = nparray[0,0,:,:]

    fig = plt.figure(figsize=(40,10))

    plot_args = dict(
        cmap='seismi',
        vmin = -1, 
        vmax = 1, 
        aspect = 'auto'
    )

    plt.imshow(nparray, **plot_args)
    st.pyplot(fig)