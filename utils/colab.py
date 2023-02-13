# commands for colab 

from google.colab import drive
drive.mount('/content/drive')

!pip install -qq rich pytorch-lightning==1.7.7 scikit-image==0.19.2
!git clone https://github.com/jayantb1019/adl_seismic.git
!mkdir -p /content/adl_seismic/data
!unzip /content/drive/MyDrive/faciesmark.zip -d /content/adl_seismic/data/.

