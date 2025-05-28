# from PIL import Image, ImageDraw, ImageFont
import cv2
import os
import numpy as np
import math
# from utils import read_pos_data, undistortion
# from ortho import cam_to_iner_homo_matrix, kps2uvs, get_dst_size_HoM
# import config as cfg
from downloader import *
import os
import ast
import numpy as np
import math
from main import *

source_frames: str = 'data_fall'
source_map: str = 'output19.tif'
source_coor = [83.195558, 54.798793, 83.214901, 54.786596,]
scale = 10
step = 10
line_CLIP = []

# area_tensors = pre_proc(cv2.imread(source_map))
# satellite_features_map = get_features_map(s_encoder, area_tensors[None,:,:,:].to(device), n=1)
# # print(type(satellite_features_map))

with open('autumn_features.npy', 'rb') as f:
    satellite_features_map = np.load(f)
satellite_features_map = torch.from_numpy(satellite_features_map)


# print('=========', satellite_features_map.shape)
# dron_position = DronePos((1000, 6800), satellite_features_map)
# dron_position = DronePos((2000, 2800), satellite_features_map) fall
dron_position = DronePos((2500, 3100), satellite_features_map)
y, x = dron_position.y_x
line_CLIP.append((y, x))

import os
import numpy as np
import cv2
from PIL import Image
import math
from utils import read_pos_data, undistortion
from ortho import cam_to_iner_homo_matrix, kps2uvs, get_dst_size_HoM
import config as cfg

def to_orto(pos):
    frame = pos['image']
    pos['yaw'] = 0
    frame = undistortion(frame, crop=False)
    HoM = cam_to_iner_homo_matrix(pos)
    dst_size, THoM = get_dst_size_HoM(frame, HoM)
    tframe = cv2.warpPerspective(frame, THoM, dst_size)
    return tframe

def draw_square(image, center_x, center_y, side_length, color=(0, 255, 0)):
#     image = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Рассчитываем координаты вершин квадрата
    x1 = center_x - side_length // 2
    y1 = center_y - side_length // 2
    x2 = center_x + side_length // 2
    y2 = center_y + side_length // 2
    
    # Рисуем квадрат на изображении
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 15)
    
    return image

def draw_line_on_map(source_map, source_coor, line_coor, line_CLIP, drone, compass, dron_position):
    img = cv2.imread(source_map)
    img_for_crops = img.copy()
    # Извлечение координат границ карты
    left, top, right, bottom = map(float, source_coor)
    # Масштабирование координат точек линии
    line_x = [(coor[0] - left) / (right - left) * img.shape[1] for coor in line_coor]
    line_y = [(coor[1] - top) / (bottom - top) * img.shape[0] for coor in line_coor]
    # Преобразование координат в целочисленные значения
    line_pts = [(int(x), int(y)) for x, y in zip(line_x, line_y)]
#     print('GPS',line_pts[-1])
    crop = drone
    # crop = np.array(undistorton(drone, crop=True))
    height, width = crop.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), -compass, 1.0)
    crop = cv2.warpAffine(crop, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
#     crop = crop[(crop.shape[0]-crop.shape[0]/2**0.5)//2:(crop.shape[0]-crop.shape[0]/2**0.5)//2+crop.shape[0]/2**0.5,
#             (crop.shape[1]-crop.shape[0]/2**0.5)//2:(crop.shape[1]-crop.shape[0]/2**0.5)//2+crop.shape[0]/2**0.5]
    
    # вручную меняем масштаб в кропе
    size = 56*2
    bias = 0
    center_x = crop.shape[1] // 2 + bias
    center_y = crop.shape[0] // 2 + bias
    left = center_x - size // 2
    right = center_x + size // 2
    top = center_y - size // 2
    bottom = center_y + size // 2
    crop = crop[top:bottom, left:right]
#     crop = cv2.resize(crop, (crop.shape[1]*2, crop.shape[0]*2))
#     crop = crop[112:336, 112:336]
#     crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    
    
#     передаем кроп на инференс КЛИПу
    dron_position, hm = main(crop, dron_position)
    y, x = dron_position.y_x

    

    _, y_x = get_max_idxs(hm)
    pred_y, pred_x = get_pixel(y_x)
    line_CLIP.append((pred_x, pred_y))
    print(pred_y, pred_x, end='\r')   
#     print(x, y)

    
#
    
    
    
 #     временный блок для отрисовки инференса во время остановки смещения зоны поиска
#     print('=======',dron_position.y_x, dron_position.get_area().shape)
    center_y, center_x = dron_position.y_x
    side_length = dron_position.get_area().shape[-1]*64
#     print(dron_position.get_area().shape[-1])
#     img = draw_square(img, center_x, center_y, side_length, color=(0, 255, 0))

    
    
    
    img = draw_square(img, pred_x, pred_y, 30, color=(0, 255, 0))
#     img = draw_square(img, center_x-side_length//2+pred_x*2, center_y-side_length//2+pred_y*2, 35, color=(0, 255, 0))
    # Рисование линии на изображении
    cv2.polylines(img, [np.array(line_pts)], False, (0, 0, 255), thickness=50)
    cv2.polylines(img, [np.array(line_CLIP)], False, (255, 0, 0), thickness=20)
    

#     для отрисовки изображения с камеры дрона
    crop = cv2.resize(crop, (2000, 2000))
    img[img.shape[0]-crop.shape[0]: img.shape[0], img.shape[1]-crop.shape[1]: img.shape[1]] = crop
    
    
         #     Вставка предсказанного изображения
    pred_x_map, pred_y_map = pred_x, pred_y
    if pred_x_map < size:
        pred_x_map = size
    if pred_y_map < size:
        pred_y_map = size
    if pred_x_map > img.shape[1]:
        pred_x_map = img.shape[1] - size
    if pred_y_map > img.shape[0]:
        pred_y_map = img.shape[0] - size
    crop_pred = img_for_crops[pred_y_map - size: pred_y_map + size, pred_x_map - size: pred_x_map + size]
    cv2.imwrite(os.path.join(f'test.png'), crop_pred)
    crop_pred = cv2.resize(crop_pred, (2000,2000), cv2.INTER_NEAREST)
    img[0: crop_pred.shape[0], img.shape[1]-crop_pred.shape[1]: img.shape[1]] = crop_pred
    
    
    
#     drone = cv2.resize(drone, (drone.shape[1]*6, drone.shape[0]*6))
# #     img[0: drone.shape[0], 0: drone.shape[1]] = drone

    # Вставка hm
    blackk = np.zeros_like(hm)
    crop = cv2.merge((hm,hm,hm))
#     crop = cv2.resize(crop, (crop.shape[1]*6, crop.shape[0]*6))
    crop = cv2.resize(crop, (2000, 2000) , cv2.INTER_NEAREST)
#     crop = cv2.applyColorMap(crop, cv2.COLORMAP_JET)
    img[img.shape[0]-crop.shape[0]: img.shape[0], 0: crop.shape[1]] = crop

    # Отображение изображения
    img = cv2.resize(img, (img.shape[1]//scale, img.shape[0]//scale))
    
    
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fontScale = 2
    color = (255, 0, 0) 
    thickness = 2
    img = cv2.putText(img, f'x: {pred_x}, y: {pred_y}', (50, 100) , font,  
                       fontScale, color, thickness, cv2.LINE_AA) 
    
    
    cv2.imwrite(os.path.join(f'Map_with_route.png'), img)
#     cv2.imshow("Map", img)
#     cv2.waitKey(1)
    return img, line_CLIP, dron_position

file_list = os.listdir(source_frames)
# print(file_list)
# os.remove('data/.ipynb_checkpoints')
# sorted_files = sorted(file_list, key=lambda x: int(x[:-4]))
sorted_files = sorted(file_list, key=lambda x: int(x[:-4]) if x[:-4].isdigit() else float('inf'))
line_coor = []
video_writer = cv2.VideoWriter('t.avi', cv2.VideoWriter_fourcc(*'XVID'), 2, (cv2.imread(source_map).shape[1]//scale, cv2.imread(source_map).shape[0]//scale))
# video_writer = cv2.VideoWriter('t.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 1, (cv2.imread(source_map).shape[1]//scale, cv2.imread(source_map).shape[0]//scale))
for file_name in sorted_files[200:700:step]:
    file_path = os.path.join(source_frames, file_name)
    im = Image.open(file_path)
    try:
        metadata = ast.literal_eval(im.info['metadata'])
        attitude = metadata['PixHawk_data']['ATTITUDE']
        pitch = attitude['pitch']
        roll = attitude['roll']
        global_position_int = metadata['PixHawk_data']['GLOBAL_POSITION_INT']
        relative_alt = global_position_int['relative_alt']/1000
        lat = global_position_int['lat']/10**7
        lon = global_position_int['lon']/10**7
        compass = metadata['PixHawk_data']['VFR_HUD']['heading']
        # print(lat, lon, relative_alt)
    except:
        print('INVALID DATA')
    line_coor.append((lon, lat))
    drone = cv2.cvtColor(np.array(im), cv2.COLOR_BGR2RGB)
    
#     'lat': 54.7930643, 'lon': 83.2044773, 'rel_alt': 148554, 'vels': [-490, 235, -123],
# 'pressure': 979.5493774414062, 'temperature': 1407.0, 'altitude': 99.8334405023838,
# 'roll': 0.11929084360599518, 'pitch': 0.5052534341812134, 'yaw': 2.632648468017578,
# 'heading': 2.6179938779914944}
    pos = {
        'image': drone,
        'pitch': pitch,
        'roll': roll,
#         'heading': compass
    }
    drone = to_orto(pos)
    
    img, line_CLIP, dron_position = draw_line_on_map(source_map, source_coor, line_coor, line_CLIP, drone, compass, dron_position)
    video_writer.write(img)

video_writer.release()
cv2.imwrite(os.path.join(f'Map_with_route.png'), img)
cv2.destroyAllWindows()

# from IPython import display

# video = mmcv.VideoReader('t.mp4')
# frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
# display.Video('tt.mp4')
