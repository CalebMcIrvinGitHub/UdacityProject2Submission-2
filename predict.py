import sys

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import json

import argparse

image_path_checking = sys.argv[1]
checkpoint_dir=sys.argv[2]

parser = argparse.ArgumentParser()

parser.add_argument("--gpu", dest='gpu', nargs='*')
parser.add_argument("--category_names", dest='category_names')
parser.add_argument("--top_k", dest='top_k')

args, unknown = parser.parse_known_args()

if not args.gpu == []:
    device = ("cuda")
else:
    device = ("cpu")


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc=checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    learning_rate = checkpoint['learning_rate']
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)    
    optimizer.load_state_dict(checkpoint['optimizer_dict'])
    
    return optimizer, model

optimizer, model = load_checkpoint(checkpoint_dir)
model.to(device)       
            
def process_image(image):
    
    image.thumbnail((256, 256))
    
    width, height = image.size

    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2

    image = image.crop((left, top, right, bottom))
    
    np_image = np.array(image)
    
    np_image = np.true_divide(np_image, 255)
    
    mean=[.485, .456, .406]
    std=[.229, .224, .225]
    
    np_image = (np_image-mean)/std
    
    np.transpose(np_image,(2,0,1))
    
    return torch.tensor(np_image)      
            
            
            
            
            
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 0, 2))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
            
            
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    
    im = PIL.Image.open(image_path)
    im = process_image(im)
    im = im.numpy()
    im.resize(3,224,224)
    im = torch.tensor(im).float()
    
    
    im.unsqueeze_(0)
    model = model.to("cpu")
    topk_var = model(im)
    topk_var = torch.exp(topk_var)
    
    return topk_var.topk(topk)

model.eval()

# TODO: Display an image along with the top 5 classes
plt.figure(figsize=(6,10))
ax=plt.subplot(2,1,1)
model.eval()

cat_to_name=''

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)


im = PIL.Image.open(image_path_checking)
im = process_image(im)
imshow(im, ax)

top_k_values, top_k_indices = predict(image_path_checking, model, int(args.top_k))

list_show=[]
for i in range(len(top_k_values[0])):
    list_show.append(top_k_values[0][i].item())
tuple_show = tuple(list_show)


plt.subplot(2,1,2)

y = []

for i in top_k_indices:
    for iterator in i:
        y.append(cat_to_name[str(iterator.item())])




energy = (tuple_show)


y_pos = [i for i, _ in enumerate(y)]

plt.barh(y_pos, energy, color='green')

plt.yticks(y_pos, y)
ax.invert_yaxis()
plt.xticks(x_pos, x)                
