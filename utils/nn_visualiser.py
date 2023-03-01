from mirror import mirror
from mirror.visualisations.web import *
from PIL import Image
import imageio
from torchvision.models import resnet101, resnet18, vgg16, alexnet
from torchvision.transforms import ToTensor, Resize, Compose

# create a model
model = vgg16(pretrained=True)
# open some images

cat = Image.open("/Users/jayanthboddu/Desktop/data_science/upgrad/MSDS/experiments_feb/data/nn_visualiser/1.jpg")
dog_and_cat = Image.open("/Users/jayanthboddu/Desktop/data_science/upgrad/MSDS/experiments_feb/data/nn_visualiser/2.jpg")

# print(cat.shape , dog_and_cat.shape)
# resize the image and make it a tensor
to_input = Compose([Resize((224, 224)), ToTensor()])
# call mirror with the inputs and the model
mirror([to_input(cat), to_input(dog_and_cat)], model, visualisations=[BackProp, GradCam, DeepDream])