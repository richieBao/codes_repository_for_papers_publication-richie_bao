# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:32:49 2021

@author: richie bao -Spatial structure index value distribution of urban streetscape
ref:https://towardsdatascience.com/image-feature-extraction-using-pytorch-e3b327c3607a
"""
import torch
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.transforms import Compose, CenterCrop, Normalize, Scale, Resize
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import glob,os 
from torch.autograd import Variable
from torch import optim, nn
from collections import Counter

NUM_CHANNELS = 3

image_transform=ToPILImage()
input_transform_cityscapes=Compose([
    Resize((512,1024*1),Image.BILINEAR),
    ToTensor(),
])

class imgs_dataset(Dataset):
    def __init__(self, root, transforms=None, labels=[], limit=None,file_type='jpg',shuffle=False):
        self.root = Path(root)
        self.file_type=file_type
        self.image_paths = glob.glob(os.path.join(img_path,'*.{}'.format(file_type)))
        # print(self.image_paths)
        if limit:
            self.image_paths = self.image_paths[:limit]
        self.labels = labels
        self.transforms = transforms
        # self.classes = set([path.parts[-2] for path in self.image_paths])
        self.shuffle=shuffle        
        # print(self.image_paths)
        
    def __getitem__(self, index):
        # print("_"*50)
        image_path = self.image_paths[index]
        label = self.labels[index] if self.labels else 0
        if self.file_type=='npy':
            image_array=np.load(image_path)    
            image=Image.fromarray((image_array*255/np.max(image_array)).astype('uint8'))
            #image=np.load(image_path)
        else:
            image=Image.open(image_path)
        if self.transforms:
            return self.transforms(image), label        
        return image, label
            
    def __len__(self):
        return len(self.image_paths)  

import re
import torch
import torch.nn as nn
from torchvision import models
def insert_module(model, indices, modules):
    indices = indices if isinstance(indices, list) else [indices]
    modules = modules if isinstance(modules, list) else [modules]
    assert len(indices) == len(modules)

    layers_name = [name for name, _ in model.named_modules()][1:]
    for index, module in zip(indices, modules):
        layer_name = re.sub(r'(.)(\d)', r'[\2]', layers_name[index])
        exec("model.{name} = nn.Sequential(model.{name}, module)".format(name = layer_name))

def feature_extraction(img_path,ERFNet_fn,limit_images=10,batch_size=1,out_num=256):
    from tqdm import tqdm
    import numpy as np
    
    ERFNet=torch.load(ERFNet_fn)
    # print(ERFNet)
    ERFNet_encoder=ERFNet.module.encoder
    
    ERFNet_encoder.eval()
    ERFNet_encoder.cuda()
    # print(ERFNet_encoder) 
    
    limit_images=limit_images
    batch_size=batch_size
    raw_dataset=imgs_dataset(root=img_path, transforms=input_transform_cityscapes, limit=limit_images,file_type='jpg',shuffle=True) #shuffle=True
    loader=DataLoader(raw_dataset, batch_size=batch_size)
    features=[]
    
    out_num=out_num
    out_layers=nn.Sequential(
        # nn.AdaptiveAvgPool2d(output_size=(7, 7)),
        nn.Flatten(),   
        nn.Linear(128*64*128, out_num) #512,
        ).cuda() 
    
    for img, _ in tqdm(loader):
        # print(img.shape)
        feature=ERFNet_encoder(Variable(img).cuda())
        # print(feature.shape)         
        out_feature=out_layers(feature)
        features.append(out_feature.cpu().detach().numpy().reshape(-1))
    features=np.array(features)    
    print(features.shape)
    return features,raw_dataset

def cluster_KMeans(featurs,n_clusters=5,pca_dim=10):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import IncrementalPCA
    
    pca_dim=pca_dim
    pca=IncrementalPCA(n_components=pca_dim, batch_size=512, whiten=True)
    reduced=pca.fit_transform(features)
    print(features.shape,reduced.shape)
    
    model_km=KMeans(n_clusters=n_clusters, random_state=42)
    model_km.fit(reduced)
    pseudo_labels=model_km.labels_
    return pseudo_labels

def show_cluster(cluster, labels, dataset, limit=32,figsize=(15, 10)):
    import numpy as np
    from torchvision.utils import make_grid
    import matplotlib.pyplot as plt
    
    images = []
    labels = np.array(labels)
    indices = np.where(labels==cluster)[0]
    
    if not indices.size:
        print(f'cluster: {cluster} is empty.')
        return None
    
    for i in indices[:limit]:
        image, _ = dataset[i]
        images.append(image)
        
    gridded = make_grid(images)
    plt.figure(figsize=figsize)
    plt.title(f'cluster: {cluster}')
    plt.imshow(gridded.permute(1, 2, 0))
    plt.axis('off')

if __name__=="__main__":
    # img_path=r'./data/panoramic imgs valid'
    img_path=r'C:\Users\richi\omen_richiebao\omen_github\codes_repository_for_papers_publication-richie_bao\quantification of visual perception changes based on spatial data of streets in the driverless city\data\panoramic imgs valid'
    ERFNet_fn='./model/ERFNet.pth'
    limit_images=100
    features,raw_dataset=feature_extraction(img_path,ERFNet_fn,limit_images=limit_images,batch_size=1,out_num=256)
    
    pseudo_labels=cluster_KMeans(features,n_clusters=5,pca_dim=100) #pca_dim<=batch number of samples
    counts=Counter(pseudo_labels)
    show_cluster(counts.most_common()[3][0], pseudo_labels, raw_dataset,limit=limit_images,figsize=(50,50))
