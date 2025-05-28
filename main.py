import numpy as np
import cv2
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from utils_winter import CLIPModel

class DronePos:
#     a_shape = (448, 448) # area_half_shape px
#     f_shape = (14, 14) # features shape 
    a_shape = (20*32, 20*32) # area_half_shape px
    f_shape = (20, 20) # features shape 
    
    def __init__(self, init_coord, features_map):
        self.y_x = init_coord # px
        self.sat_features_map = features_map
        self.area = None
        self.update_area()
    
    def update_area(self):
#         print('=====',self.y_x)
        y, x = self.y_x
        y, x = y//32, x//32
#         self.area = self.sat_features_map[:,:, y-self.f_shape[0]:y+self.f_shape[0], x-self.f_shape[1]:x+self.f_shape[1]]
        self.area = self.sat_features_map
    def update_pos(self, y_x): # px
        dy, dx = y_x
        new_y_x = (self.y_x[0] + dy - self.a_shape[0], self.y_x[1] + dx - self.a_shape[1])
#         print('NEW', new_y_x)
#         self.y_x = new_y_x
#         self.update_area()
    
    def get_area(self):
        return self.area

# class DronePos:
# #     a_shape = (448, 448) # area_half_shape px
# #     f_shape = (14, 14) # features shape 
#     a_shape = (50*32, 50*32) # area_half_shape px
#     f_shape = (100, 100) # features shape 
    
#     def __init__(self, init_coord, features_map):
#         self.y_x = init_coord # px
#         self.sat_features_map = features_map
#         self.area = None
#         self.update_area()
    
#     def update_area(self):
#         y, x = self.y_x
#         y, x = y//32, x//32
#         self.area = self.sat_features_map[:, y-self.f_shape[0]:y+self.f_shape[0], x-self.f_shape[1]:x+self.f_shape[1]]
    
#     def update_pos(self, new_y_x): # px
#         #new_y_x = self.y_x[0] + dy - self.a_shape[0], self.y_x[1] + dx - self.a_shape[1]
#         self.y_x = new_y_x
# #         self.update_area()
    
#     def get_area(self):
#         return self.area

#Preprocess
transform = transforms.Compose([
        # Convert image to tensor with image values in [0, 1]
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
        ])

def pre_proc(img):
    return transform(img)

# Model
crop_size = 224
# input_image_size = 160

input_image_size = 224

# device = "cuda"
device = 'cpu'
model = CLIPModel().to(device)

# model_path = f"best_{crop_size}_{input_image_size}.pt"
model_path = f'best_autumn_margin_1.pt'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

d_encoder = model.drone_encoder
s_encoder = model.satellite_encoder

del(model)
# f"best_cropsize{crop_size}_inputsize{input_image_size}.pt"

def get_features_map(model, input_tensor, n = 1):
    #global conv_features
    step = (32*n)
    print('step', step)
    n_col = (input_tensor.shape[3])// step 
    n_str = (input_tensor.shape[2])// step  
    print('n_col', n_col)
    print('n_str', n_str)
    if n_str <= 0 or n_col <= 0:
        features_map, _ = model(input_tensor)
        features_map = features_map.permute(1, 0).cpu().detach()
        return F.normalize(features_map, p=1, dim=0)
    for i in range(n_str):
        for j in range(n_col):
            crop = input_tensor[:,:,i*step:i*step+crop_size, j*step:j*step+crop_size]
            feature, _ = model(crop)
            feature = feature.permute(1, 0)
            feature = feature.cpu().detach()
            if j == 0:
                line = feature
                continue
            line = torch.cat((line, feature), 1) #torch.hstack((line, feature))
            del feature
        
        if i == 0:
            features_map = line[:,None, :]
            continue
        features_map = torch.cat((features_map, line[:,None,:]), 1) #torch.vstack((features_map, line))
    features_map = features_map.cpu().detach()
    return F.normalize(features_map, p=1, dim=0)

# def get_drone_features(drone_img):
#     img_shape = drone_img.shape[:2]
#     up, left = img_shape[0]//2, img_shape[1]//2
#     input_crop =  drone_img[up:up+crop_size, left:left+crop_size, :]
# #     input_img = cv2.resize(input_crop, (input_image_size, input_image_size)) 
#     input_img = input_crop
    
#     drone_tensor = pre_proc(input_img)
#     frame_features, _ = d_encoder(drone_tensor[None,:,:,:].to(device))
#     frame_features =  frame_features.permute(1, 0).cpu().detach() 
    
#     frame_features = F.normalize(frame_features, p=2, dim=0)
#     frame_features[frame_features<=frame_features.mean()] = 0
#     return frame_features

def get_drone_features(drone_img):
    img_shape = drone_img.shape[:2]
    up, left = img_shape[0]//2, img_shape[1]//2
    input_crop =  drone_img[up:up+crop_size, left:left+crop_size, :]
    input_img = cv2.resize(input_crop, (input_image_size, input_image_size)) 
    drone_tensor = pre_proc(input_img)
    _, frame_features = d_encoder(drone_tensor[None,:,:,:].to(device))
    frame_features =  frame_features.cpu().detach() 
    #frame_features_n = F.normalize(frame_features, p=1, dim=0)
    return F.normalize(frame_features)

# def get_heat_map(sat_features, drone_features):
#     tresh = 200
    

    
#     heat_map = F.conv2d(sat_features[None,:,:,:], drone_features[None,:,:,None]).cpu().detach().squeeze().numpy() #padding – 'same'
    
#     heat_map = heat_map/heat_map.max() * 255.
#     heat_map = heat_map.astype(np.uint8)
#     # clip
#     heat_map_c = heat_map.copy()
#     heat_map_c[heat_map_c<tresh] = tresh
#     return heat_map_c #heat_map_c heat_map

def hm_vote(heat_maps):
    y_x_list = []
    for heat_map in heat_maps:
        y, x = np.where(heat_map==heat_map.max())
        y_x_list.append(int(y))
        y_x_list.append(int(x))
    max_count = 0
    max_n = 0
    for n, el in enumerate(y_x_list):
        if y_x_list.count(el) > max_count:
            max_count = y_x_list.count(el)
            max_n = n
    return max_n//2

# def get_heat_map(sat_features, drone_features):
#     tresh = 200
#     heat_map = F.conv2d(sat_features, drone_features[None,:,:,None]).cpu().detach().squeeze().numpy() #padding – 'same'
#     #z, y, x = np.where(heat_map==heat_map.max()) 
#     max_ind = hm_vote(heat_map)
#     heat_map = heat_map[max_ind]
#     heat_map = heat_map/heat_map.max() * 255.
#     heat_map = heat_map.astype(np.uint8)
#     # clip
#     heat_map_c = heat_map.copy()
#     heat_map_c[heat_map_c<tresh] = tresh
#     return heat_map_c

def get_heat_map(sat_features, drone_features):
    tresh = 100
#     print('SAT', sat_features.shape, 'DRONE', drone_features.shape)
    heat_map = F.conv2d(sat_features, drone_features).cpu().detach().squeeze().numpy() #padding – 'same'
    #z, y, x = np.where(heat_map==heat_map.max()) 
    #max_ind = hm_vote(heat_map)
    heat_map -= np.mean(heat_map)#.unsqueeze(1)
    heat_map /= np.std(heat_map)#.unsqueeze(1)
    heat_map[heat_map<0]=0
    heat_map = heat_map/heat_map.max() * 255.
    heat_map = heat_map.astype(np.uint8)
    # clip
    heat_map_c = heat_map.copy()
    heat_map_c[heat_map_c<tresh] = tresh
    return heat_map_c

def get_pixel(heat_map):
    y, x = np.where(heat_map==heat_map.max())
    y, x = int(y), int(x)

#     print('------------', y, x)
    y, x = 32*y + 16, 32*x + 16
    return x, y

# def get_pixel(heat_map):
#     y, x = np.where(heat_map==heat_map.max())
# #     print(y, x)
#     y, x = int(y[0]), int(x[0])
# #     print(y, x)
# #     dy, dx = 32*y-112, 32*x-112
# #     dy, dx = 32*y+112, 32*x+112
#     dy, dx = 32*y, 32*x
#     return dy, dx
def get_pixel(y_x):
#     print(y_x)
#     y, x = 32*y_x[0] + 16, 32*y_x[1] + 16
    y, x = 32*y_x[0], 32*y_x[1]
    return y, x

# def get_max_idxs(heat_map, n = 2):
#     kernel = np.ones((n, n), np.uint8) 
#     heat_map[0,:] = 0
#     heat_map[-1,:] = 0
#     heat_map[:,0] = 0
#     heat_map[:,-1] = 0
#     image_erode = cv2.erode(heat_map, kernel)
    
# #     print('ERODE', image_erode)
    
#     y_arr, x_arr = np.where(image_erode==image_erode.max())
    
# #     print('========',y_arr, x_arr)
    
#     if len(y_arr) > 1:
#         if n>5:
#             return image_erode, (0,0)
#         n = n+1 
#         return get_max_idxs(image_erode, n)
#     return image_erode, (int(y_arr[0]),int(x_arr[0]))

def get_max_idxs(heat_map):
    y,x = np.where(heat_map==heat_map.max()) #  y, x 
    return heat_map, (int(y[0]), int(x[0]))

# def main(frame, dron_position):
# #     satellite_features_map = torch.from_numpy(satellite_features_map)
# #     dron_position = DronePos((1000, 6800), satellite_features_map)
# #     print(dron_position.y_x)
#     while True:
#         #ret, frame = cap.read()
#         # frame = cv2.imread('../../../CLIP_train/winter_dataset/crops/310124_42.png')
#         # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         drone_features = get_drone_features(frame)
#         sat_features = dron_position.get_area()
# #         print('SHAPES', drone_features.shape, sat_features.shape)
# #         try:
#         hm = get_heat_map(sat_features, drone_features)
# #             print('HM')
#         y_x = get_pixel(hm)
# #         print('Y_X', type(y_x), y_x)
#         dron_position.update_pos(y_x)
# #         print('UPDATE')
# #         except:
# #             pass
# #         y, x = dron_position.y_x
#         return dron_position, hm

def main(frame, dron_position):
#     dron_position = DronePos((1000, 6800), satellite_features_map)
#     print(dron_position.y_x)
#     while True:
        #ret, frame = cap.read()
#         frame = cv2.imread('../../../CLIP_train/winter_dataset/crops/310124_42.png')
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
    drone_features = get_drone_features(frame)
    sat_features = dron_position.get_area()
    hm = get_heat_map(sat_features, drone_features)
    hm, y_x = get_max_idxs(hm)
    dy, dx = get_pixel(y_x)
    dy_dx = (dy, dx)
    dron_position.update_pos(dy_dx)
    
    
#     print(dron_position.y_x)
    
    return dron_position, hm

#shift_step = 56

# class DronePos:
#     a_shape = (14*32, 14*32) # area_half_shape px
#     f_shape = (14, 14) # features shape 
    
#     def __init__(self, init_coord, features_map):
#         self.y_x = init_coord # px
#         self.sat_features_map = features_map
#         self.area = None
#         self.update_area()
    
#     def update_area(self):
#         y, x = self.y_x
#         y, x = round(y/32), round(x/32)
#         self.area = self.sat_features_map[:,:, y-self.f_shape[0]:y+self.f_shape[0], x-self.f_shape[1]:x+self.f_shape[1]]
#         print('area', self.area.shape)
    
#     def update_pos(self, dy, dx): # px
#         new_y_x = self.y_x[0] + dy - self.a_shape[0], self.y_x[1] + dx - self.a_shape[1]
#         self.y_x = new_y_x
#         self.update_area()
    
#     def get_area(self):
#         return self.area

# def get_drone_features(drone_img):
#     img_shape = drone_img.shape[:2]
#     up, left = img_shape[0]//2, img_shape[1]//2
#     input_crop =  drone_img[up:up+crop_size, left:left+crop_size, :]
#     input_img = cv2.resize(input_crop, (input_image_size, input_image_size)) 
#     drone_tensor = pre_proc(input_img)
#     _, frame_features = d_encoder(drone_tensor[None,:,:,:].to(device))
#     frame_features =  frame_features.cpu().detach() 
#     #frame_features_n = F.normalize(frame_features, p=1, dim=0)
#     return F.normalize(frame_features)

# def get_heat_map(sat_features, drone_features):
#     tresh = 200
#     heat_map = F.conv2d(sat_features, drone_features).cpu().detach().squeeze().numpy() #padding – 'same'
#     #z, y, x = np.where(heat_map==heat_map.max()) 
#     #max_ind = hm_vote(heat_map)
#     heat_map -= np.mean(heat_map)#.unsqueeze(1)
#     heat_map /= np.std(heat_map)#.unsqueeze(1)
#     heat_map[heat_map<0]=0
#     heat_map = heat_map/heat_map.max() * 255.
#     heat_map = heat_map.astype(np.uint8)
#     # clip
#     heat_map_c = heat_map.copy()
#     heat_map_c[heat_map_c<tresh] = tresh
#     return heat_map_c

# def get_max_idx(heat_map):
#     y,x = np.where(heat_map==heat_map.max()) #  y, x 
#     return int(y[0]), int(x[0])

# def get_pixel(y_x):
#     y, x = 32*y_x[0] + 16, 32*y_x[1] + 16
#     return x, y

# map_img = cv2.imread('output19.tif')
# map_img = map_img[640:-640, 640:-640, :]
# map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)

# with open(f'sat_features_{input_image_size}.npy', 'rb') as f:
#     satellite_features_map = np.load(f)
#     satellite_features_map = torch.from_numpy(satellite_features_map)
    

# def main():
#     dron_position = DronePos((1000, 4800), satellite_features_map)
#     print(dron_position.y_x)
#     while True:
#         #ret, frame = cap.read()
#         frame = cv2.imread('../../CLIP_train/winter_dataset/crops/310124_42.png')
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         drone_features = get_drone_features(frame)
#         sat_features = dron_position.get_area()
#         hm = get_heat_map(sat_features, drone_features)
#         y_x = get_max_idx(hm)
#         print('y_x', y_x)
#         dy, dx = get_pixel(y_x)
#         dron_position.update_pos(dy, dx)
#         break
#     print(dron_position.y_x)

if __name__ == '__main__':
    main()
