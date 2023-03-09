# # commands for colab 

# from google.colab import drive
# drive.mount('/content/drive')

# !pip install -qq rich pytorch-lightning==1.7.7 scikit-image==0.19.2
# !git clone https://jayantb1019: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@github.com/jayantb1019/adl_seismic.git
# !mkdir -p /content/adl_seismic/data
# !unzip /content/drive/MyDrive/faciesmark.zip -d /content/adl_seismic/data/.
# !cp /content/drive/MyDrive/denoiser_20230213_epoch=49-step=27600.ckpt /content/denoiser_20230213_epoch=49-step=27600.ckpt
# !cp /content/drive/MyDrive/discriminator_20230213_epoch=49-step=27600.ckpt /content/discriminator_20230213_epoch=49-step=27600.ckpt

# %load_ext tensorboard
# %tensorboard --logdir=/content/adl_seismic/lightning_logs

# # save logs to gdrive
# # save checkpoints to drive 
# !cp -r /content/adl_seismic/lightning_logs/ /content/drive/MyDrive/adl_seismic/.