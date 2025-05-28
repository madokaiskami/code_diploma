import torch
import torch.nn as nn
from torchvision import transforms
import timm
import torch.nn.functional as F

import os
import cv2
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

crop_size = 224
input_image_size = 160

class CFG:
    batch_size = 16
    num_workers = 4
    head_lr = 1e-3
    satellite_encoder_lr = 1e-4
    drone_encoder_lr = 1e-4
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'resnet50'
    image_embedding = 2048

    pretrained = True
    trainable = True
    temperature = 1.0

    # image size
    size = input_image_size

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 256 
    dropout = 0.1

class SatelliteEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    model_name = ResNet50, pretrained=True, trainable=True
    """

    def __init__(self, model_name='resnet50',
                 pretrained=True, trainable=True,
                 image_embedding=2048):
        super().__init__()
        self.resnet = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.resnet.parameters():
            p.requires_grad = trainable
        self.satellite_projection = ProjectionHead(embedding_dim=image_embedding)
        self.resnet.layer4.register_forward_hook(self._get_features)

    def forward(self, x):
        x = self.resnet(x)
        return self.satellite_projection(x), self._avgpool_features
    
    @property
    def last_feature(self):
        """It works after 'forward' only"""
        return self._avgpool_features
    
    def _get_features(self, module, inputs, output):
        self._avgpool_features = output
class DroneEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(self, model_name='resnet50',
                 pretrained=True, trainable=True,
                 image_embedding=2048):
        super().__init__()
        self.resnet = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.resnet.parameters():
            p.requires_grad = trainable
        self.drone_projection = ProjectionHead(embedding_dim=image_embedding)
        self.resnet.layer4.register_forward_hook(self._get_features)

    def forward(self, x):
        x = self.resnet(x)
        return self.drone_projection(x), self._avgpool_features
    
    @property
    def last_feature(self):
        """It works after 'forward' only"""
        return self._avgpool_features
    
    def _get_features(self, module, inputs, output):
        self._avgpool_features = output

class ProjectionHead(nn.Module):
    
    def __init__(self, embedding_dim, projection_dim=CFG.projection_dim,dropout=0.1):
        super().__init__()
        self.projection_dim = projection_dim
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

# class ProjectionHead(nn.Module):
    
#     def init(self, embedding_dim, projection_dim=256,dropout=0.1):
#         super().init()
#         self.projection = nn.Linear(embedding_dim, projection_dim)
#         self.gelu = nn.GELU()
#         self.fc = nn.Linear(projection_dim, projection_dim)
#         self.dropout = nn.Dropout(dropout)
#         self.layer_norm = nn.LayerNorm(projection_dim)
    
#     def forward(self, x):
#         projected = self.projection(x)
#         x = self.gelu(projected)
#         x = self.fc(x)
#         x = self.dropout(x)
#         x = x + projected
#         x = self.layer_norm(x)
#         return x

class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
    ):
        super().__init__()
        self.satellite_encoder = SatelliteEncoder()
        self.drone_encoder = DroneEncoder()
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Embeddings (with same dimension)
        satellite_embeddings, _ = self.satellite_encoder(batch["satellite"])
        drone_embeddings, _ = self.drone_encoder(batch["drone"])

        # Calculating the Loss
        logits = (drone_embeddings @ satellite_embeddings.T) / self.temperature
        satellite_similarity = satellite_embeddings @ satellite_embeddings.T
        drone_similarity = drone_embeddings @ drone_embeddings.T
        targets = F.softmax(
            (satellite_similarity + drone_similarity) / 2 * self.temperature, dim=-1
        )
        drone_loss = cross_entropy(logits, targets, reduction='none')
        satellite_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (satellite_loss + drone_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

import numpy as np
import os
import json
from datetime import datetime
import cv2

from PIL import Image
import config as cfg


def read_image_and_meta(path):
    
    im = Image.open(path)
    img = np.asarray(im)
    metastr = im.info['metadata'].replace('\'', "\"")
    try:
        metadata = json.loads(metastr)
        
        return img, metadata['PixHawk_data']
    except ValueError:
        print('JSONDecodeError')
        
        return img, None

    
def from_meta_back(meta, packettype, datatypes):
    
    data = []
    for d in meta:
        if packettype == d['mavpackettype']:
            for t in datatypes:
                data.append(d[t]) 
            break

    return np.asarray(data)


def from_meta(meta, packettype, datatypes):
    
    data = []
    if type(meta)==list:
        for d in meta:
            if packettype == d['mavpackettype']:
                for t in datatypes:
                    data.append(d[t]) 
                break
    elif type(meta)==dict:
        for t in datatypes:
            data.append(meta[packettype][t])

    return np.asarray(data)


def pt2h(abs_pressure, P0, temperature):
    
    return (1 - abs_pressure/P0) * 8.3144598 * (273.15 + temperature/100) / 9.80665 / 0.0289644


def read_pos_data(path : str, P0 : float) -> dict:
    """
    args:
        path to png with meta
        P0 - pressure at drone start
    
    """
    frame, meta = read_image_and_meta(path)
    
    timestamp = from_meta(meta, 'SYSTEM_TIME', ['time_boot_ms'])
    
    lat, lon, rel_alt, vx, vy, vz = from_meta(meta, 'GLOBAL_POSITION_INT', ['lat', 'lon', 'relative_alt', 'vx', 'vy', 'vz'])
    
    press_abs, temperature = from_meta(meta, 'SCALED_PRESSURE', ['press_abs', 'temperature'])
    altitude = pt2h(press_abs, P0, temperature)

    angles = from_meta(meta, 'ATTITUDE', ['roll', 'pitch', 'yaw'])
            
    heading = from_meta(meta, 'VFR_HUD', ['heading']) / 180 * np.pi
    
    return dict(
                timestamp=float(timestamp)/1000,
                image=frame,
                lat=lat/10**7,
                lon=lon/10**7,
                rel_alt=rel_alt,
                vels = [vx, vy, vz],
                pressure=press_abs,
                temperature=temperature,
                altitude=altitude,
                roll=float(angles[0]),
                pitch=float(angles[1]),
                yaw=float(angles[2]),
                heading=float(heading),
                #dpp=cfg.TEST_DPP - angles[:2]*cfg.focal,
               )


def to_homo(arr):
    
    if len(arr.shape)==1:
        
        return np.hstack((arr, 1))
    else:
        
        homo = np.ones((len(arr),1), dtype=arr.dtype)
        
        return np.hstack((arr, homo))

def undistortion(img: list, crop: bool = True):
    # w, h = img.size
    # camera_matrix = np.array([[447.27399414, 0, 320],
    #                           [0, 447.2436399, 180],
    #                           [0, 0, 1]])
    # camera_matrix = np.array([[447.27399414, 0, w//2],
    #                           [0, 447.2436399, h//2],
    #                           [0, 0, 1]])
    camera_matrix = cfg.CaM
    dist_coefs = np.array([-0.35328536, 0.2090826, 0,  0,  -0.06274637])
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
    dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    x, y, w, h = roi
    if crop:
        dst = dst[y:y+h,x:x+w]
    # dst = Image.fromarray(dst)
    return dst

# class SatelliteEncoder(nn.Module):
#     """
#     Encode images to a fixed size vector
#     model_name = ResNet50, pretrained=True, trainable=True
#     """

#     def init(self, model_name='resnet50',
#                  pretrained=True, trainable=True,
#                  image_embedding=2048):
#         super().init()
#         self.model = timm.create_model(
#             model_name, pretrained, num_classes=0, global_pool="avg"
#         )
#         for p in self.model.parameters():
#             p.requires_grad = trainable
#         self.satellite_projection = ProjectionHead(embedding_dim=image_embedding)
#         self.model.layer4.register_forward_hook(self._get_features)

#     def forward(self, x):
#         x = self.model(x)
#         return self.satellite_projection(x), self._avgpool_features
    
#     @property
#     def last_feature(self):
#         """It works after 'forward' only"""
#         return self._avgpool_features
    
#     def _get_features(self, module, inputs, output):
#         self._avgpool_features = output #output.data.cpu().numpy()
        

    
# class DroneEncoder(nn.Module):
#     """
#     Encode images to a fixed size vector
#     """

#     def init(self, model_name='resnet50',
#                  pretrained=True, trainable=True,
#                  image_embedding=2048):
#         super().init()
#         self.model = timm.create_model(
#             model_name, pretrained, num_classes=0, global_pool="avg"
#         )
#         for p in self.model.parameters():
#             p.requires_grad = trainable
#         self.drone_projection = ProjectionHead(embedding_dim=image_embedding)
#         self.model.layer4.register_forward_hook(self._get_features)

#     def forward(self, x):
#         x = self.model(x)
#         return self.drone_projection(x), self._avgpool_features
    
#     @property
#     def last_feature(self):
#         """It works after 'forward' only"""
#         return self._avgpool_features
    
#     def _get_features(self, module, inputs, output):
#         self._avgpool_features = output #output.data.cpu().numpy()
        
# class CLIPModel(nn.Module):
    
#     def init(self, temperature=1.0):
#         super().init()
#         self.satellite_encoder = SatelliteEncoder()
#         self.drone_encoder = DroneEncoder()
#         self.temperature = temperature

#     def encode_satellite(self, satellite_imgs):
#         satellite_embeddings, _ = self.satellite_encoder(satellite_imgs)
#         return satellite_embeddings
    
#     def encode_drone(self, drone_imgs):
#         drone_embeddings, _ = self.drone_encoder(drone_imgs)
#         return drone_embeddings
    
#     def forward(self, batch):
#         # Getting Image and Text Embeddings (with same dimension)
#         satellite_embeddings = self.satellite_encoder(batch["satellite"])
#         drone_embeddings = self.drone_encoder(batch["drone"])










