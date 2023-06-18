import os
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import math
import json

import torch
import torch.optim as optim
import requests
from torchvision import transforms, models
from torchsummary import summary
import random


class NeuralStyleTransfer:
    
    def __init__(self, artist, device=torch.device('cpu')):
        self.artist = artist
        print("Style Transfer Network for", self.artist)
        
        with open('style_weights.json', 'r') as f:
            style_weights = json.load(f)
            self.style_weights = style_weights[artist]

        self.device = device

        self.vgg = models.vgg19(weights='IMAGENET1K_V1').features
        for param in self.vgg.parameters():
            param.requires_grad_(False)
        
    def set_seed(self, seed=143):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        print(f"Seed for torch, GPU, and backend set to: {seed}")

    def im_convert(self, tensor):    
        image = tensor.to("cpu").clone().detach()
        image = image.numpy().squeeze()
        image = image.transpose(1,2,0)
        image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
        image = image.clip(0, 1)

        return image

    def get_features(self, image, layers=None):
        if layers is None:
            layers = {'0': 'conv1_1',
                      '5': 'conv2_1', 
                      '10': 'conv3_1', 
                      '19': 'conv4_1',
                      '30': 'conv5_2', #content
                      '28': 'conv5_1'}

        features = {}
        x = image
        for name, layer in self.vgg._modules.items():
            layer = layer.to(self.device)
            x = layer(x)
            if name in layers:
                features[layers[name]] = x

        return features
    
    def gram_matrix(self, tensor):
        _, d, h, w = tensor.size()
        tensor = tensor.view(d, h * w)
        gram = torch.mm(tensor, tensor.t())

        return gram
    
    def generate_style_transfer(self, target, content, style,
                                show=1000, steps=7000,
                                save_interval=1000):
        
        content_features = self.get_features(content)
        style_features = self.get_features(style)

        style_grams = {layer: self.gram_matrix(style_features[layer]) for layer in style_features}
        content_weight = 1e-2
        style_weight = 1e9
        save_dir = f"saved_images_{self.artist}"
        os.makedirs(save_dir, exist_ok=True)
        
        optimizer = optim.Adam([target], lr=0.05)
        best_loss = math.inf
        best_target = None
        for ii in range(1, steps+1):
            target_features = self.get_features(target)

            content_loss = torch.mean((target_features['conv5_2'] - content_features['conv5_2'])**2)

            style_loss = 0
            for layer in self.style_weights:
                target_feature = target_features[layer]
                target_gram = self.gram_matrix(target_feature)
                _, d, h, w = target_feature.shape
                style_gram = style_grams[layer]
                layer_style_loss = self.style_weights[layer] * torch.mean((target_gram - style_gram)**2)
                style_loss += layer_style_loss / (d * h * w)

            total_loss = content_weight * content_loss + style_weight * style_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if ii % show == 0:
                print('Epoch: ', ii)
                print('Total loss: ', total_loss.item())
                plt.axis('off')
                plt.imshow(self.im_convert(target))
                plt.show()

            if total_loss < best_loss:
                best_loss = total_loss
                best_target = target

            if ii % save_interval == 0:
                save_path = os.path.join(save_dir, f'epoch_{ii}.png')
                plt.imshow(self.im_convert(target))
                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                
        return best_loss, best_target