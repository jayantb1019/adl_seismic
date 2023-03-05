
import streamlit as st
import matplotlib.pyplot as plt

import numpy as np

#create your figure and get the figure object returned
st.title('Dataset Quality Explorer')

fig = plt.figure() 
plt.plot([1, 2, 3, 4, 5]) 

INTERPRETATION_DATA_PATH = '/Users/jayanthboddu/Desktop/data_science/upgrad/MSDS/experiments_feb/data/f3_interpretation/inline_vol.npy'
INTERPRETATION_LABEL_PATH = '/Users/jayanthboddu/Desktop/data_science/upgrad/MSDS/experiments_feb/data/f3_interpretation/inline_label.npy'

FACIESMARK_DATA_PATH = '/Users/jayanthboddu/Desktop/data_science/upgrad/MSDS/experiments_feb/data/faciesmark/raw/seismic_entire_volume.npy'
FACIESMARK_LABEL_PATH = '/Users/jayanthboddu/Desktop/data_science/upgrad/MSDS/experiments_feb/data/faciesmark/raw/labels_entire_volume.npy'

STDATA12_DATA_PATH = '/Users/jayanthboddu/Desktop/data_science/upgrad/MSDS/experiments_feb/data/stdata12/stdata_12_amplitude.npy'
STDATA12_LABEL_PATH = '/Users/jayanthboddu/Desktop/data_science/upgrad/MSDS/experiments_feb/data/stdata12/stdata_12_labels.npy'


dataset = st.sidebar.selectbox('Select Dataset' , options = ['interpretation', 'faciesmark', 'stdata12'])

data = None
labels = None 
inlines = None
vmin = None 
vmax = None 

figsize = (40,120)

colorbar_args = dict(fraction=0.010, pad=0.04)


if dataset == 'interpretation' : 
    
    data = np.load(INTERPRETATION_DATA_PATH)
    labels = np.load(INTERPRETATION_LABEL_PATH)
    
    st.write('dataset_shape', data.shape)
    inlines = data.shape[0]
    inline_list = list(range(inlines))
    
    vmin = np.min(data)
    vmax = np.max(data)
    
    plot_args = dict(
    cmap='seismic', 
    vmin = vmin, 
    vmax = vmax
    )
    
    selected_inline = st.radio('select inline', options=inline_list, horizontal=True, format_func= lambda x : x + 100 )
    st.markdown(f'## Selected inline : {100 + selected_inline} ')
    
    # plot data 
    fig = plt.figure(figsize=figsize)   
    plt.imshow(data[selected_inline].T, **plot_args)
    plt.colorbar(**colorbar_args)
    # plt.title(f'Interpretation Dataset : data in inline # {selected_inline + 100}')
    
    st.pyplot(fig)
    
    # # plot labels 
    
    fig = plt.figure(figsize=(20,60))
    plt.imshow(labels[selected_inline].T)
    plt.colorbar(**colorbar_args)
    plt.tight_layout()
    # plt.axis('off')
    # plt.title(f'Interpretation Dataset : labels in inline # {selected_inline + 1}')
    st.pyplot(fig)
    
if dataset == 'faciesmark' : 
    
    data = np.load(FACIESMARK_DATA_PATH)
    labels = np.load(FACIESMARK_LABEL_PATH)
    
    st.write('dataset_shape', data.shape)
    
    inlines = data.shape[0]
    
    vmin = np.min(data)
    vmax = np.max(data)
    
    plot_args = dict(
    cmap='seismic', 
    vmin = vmin, 
    vmax = vmax
    )
    
    selected_inline = st.radio('select inline', options=list(range(inlines)), horizontal=True)
    st.markdown(f'## Selected inline : {selected_inline} ')
    # plot data 
    fig = plt.figure(figsize=figsize)   
    plt.imshow(data[selected_inline].T, **plot_args)
    plt.colorbar(**colorbar_args)
    # plt.title(f'FaciesMark Dataset : data in inline # {selected_inline}')
    
    st.pyplot(fig)
    
    # # plot labels 
    
    fig = plt.figure(figsize=(20,60))
    plt.imshow(labels[selected_inline].T)
    plt.colorbar(**colorbar_args)
    plt.tight_layout()
    # plt.axis('off')
    # plt.title(f'FaciesMark Dataset : labels in inline # {selected_inline}')
    st.pyplot(fig)
    
if dataset == 'stdata12' : 
    data = np.load(STDATA12_DATA_PATH)
    labels = np.load(STDATA12_LABEL_PATH)
    
    st.write('dataset_shape', data.shape)
    
    inlines = data.shape[0]
    
    vmin = np.min(data)
    vmax = np.max(data)
    
    plot_args = dict(
    cmap='seismic', 
    vmin = vmin, 
    vmax = vmax, 
    # aspect='auto'
    )
    
    selected_inline = st.radio('select inline', options=list(range(inlines)), horizontal=True, format_func= lambda x : 190 + x*100 )
    
    st.markdown(f'## Selected inline : {190 + 100 * selected_inline} ')
    
    # plot data 
    fig = plt.figure(figsize=(20,60))   
    plt.imshow(data[selected_inline].T, **plot_args)
    plt.colorbar(**colorbar_args)
    # plt.title(f'Stdata-12 Dataset : data in inline # {selected_inline}')
    
    st.pyplot(fig)
    
    # # plot labels 
    
    fig = plt.figure(figsize=(20,60))
    plt.imshow(labels[selected_inline].T)
    plt.colorbar(**colorbar_args)
    plt.tight_layout()
    # plt.axis('off')
    # plt.title(f'Stdata-12 Dataset : labels in inline # {selected_inline}')
    st.pyplot(fig)