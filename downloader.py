
from io import BytesIO
from PIL import Image, ImageDraw
from PIL import Image, PngImagePlugin
import io
import math
import multiprocessing
import time
import urllib.request as ur
from math import floor, pi, log, tan, atan, exp
from threading import Thread
import os
import PIL.Image as pil
import cv2
import numpy as np
# from osgeo import gdal, osr
import shutil
from geopy.distance import distance, geodesic
from geopy.point import Point

def clear_folder(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def crop_image_to_center(image, LT_coords, RB_coords, lon1, lat1, lon2, lat2):
    # Извлекаем размеры изображения
    width, height = image.size
    # print('==========', width, height)
    width_degrees, height_degrees = RB_coords[0] - LT_coords[0], LT_coords[1] - RB_coords[1]
    # print('=========', LT_coords, RB_coords, lon1, lat1, lon2, lat2)
    # Преобразуем географические координаты точки в пиксели
    left_frame = round(((lon1 - LT_coords[0]) * width) / width_degrees)
    upper_frame = round(((LT_coords[1] - lat1) * height) / height_degrees)
    right_frame = round(((RB_coords[0] - lon2) * width) / width_degrees)
    lower_frame = round(((lat2 - RB_coords[1]) * height) / height_degrees)
    # print('====================', left_frame, upper_frame, right_frame, lower_frame)
    image = image.crop((left_frame, upper_frame, width - right_frame, height - lower_frame))
    return image

def compute_new_coordinates(latitude, longitude, north_offset, west_offset):
    # Создаем точку с текущими географическими координатами
    current_point = Point(latitude, longitude)

    # Создаем вектор смещения на север и запад
    north_vector = geodesic(meters=north_offset)
    west_vector = geodesic(meters=west_offset)

    # Вычисляем новую точку с учетом смещения
    new_point = north_vector.destination(current_point, 0)
    new_point = west_vector.destination(new_point, 90)

    # Возвращаем новые географические координаты
    return new_point.latitude, new_point.longitude



# ------------------Interchange between WGS-84 and Web Mercator-------------------------
# WGS-84 to Web Mercator
def wgs_to_mercator(x, y):
    y = 85.0511287798 if y > 85.0511287798 else y
    y = -85.0511287798 if y < -85.0511287798 else y

    x2 = x * 20037508.34 / 180
    y2 = log(tan((90 + y) * pi / 360)) / (pi / 180)
    y2 = y2 * 20037508.34 / 180
    return x2, y2

# Web Mercator to WGS-84
def mercator_to_wgs(x, y):
    x2 = x / 20037508.34 * 180
    y2 = y / 20037508.34 * 180
    y2 = 180 / pi * (2 * atan(exp(y2 * pi / 180)) - pi / 2)
    return x2, y2

# -----------------Interchange between GCJ-02 to WGS-84---------------------------
# All public geographic data in mainland China need to be encrypted with GCJ-02, introducing random bias
# This part of the code is used to remove the bias
def transformLat(x, y):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return ret

def transformLon(x, y):
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return ret

def delta(lat, lon):
    ''' 
    Krasovsky 1940
    //
    // a = 6378245.0, 1/f = 298.3
    // b = a * (1 - f)
    // ee = (a^2 - b^2) / a^2;
    '''
    a = 6378245.0  # a: Projection factor of satellite ellipsoidal coordinates projected onto a flat map coordinate system
    ee = 0.00669342162296594323  # ee: Eccentricity of ellipsoid
    dLat = transformLat(lon - 105.0, lat - 35.0)
    dLon = transformLon(lon - 105.0, lat - 35.0)
    radLat = lat / 180.0 * math.pi
    magic = math.sin(radLat)
    magic = 1 - ee * magic * magic
    sqrtMagic = math.sqrt(magic)
    dLat = (dLat * 180.0) / ((a * (1 - ee)) / (magic * sqrtMagic) * math.pi)
    dLon = (dLon * 180.0) / (a / sqrtMagic * math.cos(radLat) * math.pi)
    return {'lat': dLat, 'lon': dLon}

def outOfChina(lat, lon):
    if (lon < 72.004 or lon > 137.8347):
        return True
    if (lat < 0.8293 or lat > 55.8271):
        return True
    return False

def gcj_to_wgs(gcjLon, gcjLat):
    if outOfChina(gcjLat, gcjLon):
        return (gcjLon, gcjLat)
    d = delta(gcjLat, gcjLon)
    return (gcjLon - d["lon"], gcjLat - d["lat"])

def wgs_to_gcj(wgsLon, wgsLat):
    if outOfChina(wgsLat, wgsLon):
        return wgsLon, wgsLat
    d = delta(wgsLat, wgsLon)
    return wgsLon + d["lon"], wgsLat + d["lat"]

# ---------------------------------------------------------
# Get tile coordinates in Google Maps based on latitude and longitude of WGS-84
def wgs_to_tile(j, w, z):
    '''
    Get google-style tile cooridinate from geographical coordinate
    j : Longittude
    w : Latitude
    z : zoom
    '''
    isnum = lambda x: isinstance(x, int) or isinstance(x, float)
    if not (isnum(j) and isnum(w)):
        raise TypeError("j and w must be int or float!")

    if not isinstance(z, int) or z < 0 or z > 22:
        raise TypeError("z must be int and between 0 to 22.")

    if j < 0:
        j = 180 + j
    else:
        j += 180
    j /= 360  # make j to (0,1)

    w = 85.0511287798 if w > 85.0511287798 else w
    w = -85.0511287798 if w < -85.0511287798 else w
    w = log(tan((90 + w) * pi / 360)) / (pi / 180)
    w /= 180  # make w to (-1,1)
    w = 1 - (w + 1) / 2  # make w to (0,1) and left top is 0-point

    num = 2 ** z
    x = floor(j * num)
    y = floor(w * num)
    return x, y

def pixls_to_mercator(zb):
    # Get the web Mercator projection coordinates of the four corners of the area according to the four corner coordinates of the tile
    inx, iny = zb["LT"]  # left top
    inx2, iny2 = zb["RB"]  # right bottom
    length = 20037508.3427892
    sum = 2 ** zb["z"]
    LTx = inx / sum * length * 2 - length
    LTy = -(iny / sum * length * 2) + length

    RBx = (inx2 + 1) / sum * length * 2 - length
    RBy = -((iny2 + 1) / sum * length * 2) + length

    # LT=left top,RB=right buttom
    # Returns the projected coordinates of the four corners
    res = {'LT': (LTx, LTy), 'RB': (RBx, RBy),
           'LB': (LTx, RBy), 'RT': (RBx, LTy)}
    return res

def tile_to_pixls(zb):
    # Tile coordinates are converted to pixel coordinates of the four corners
    out = {}
    width = (zb["RT"][0] - zb["LT"][0] + 1) * 256
    height = (zb["LB"][1] - zb["LT"][1] + 1) * 256
    out["LT"] = (0, 0)
    out["RT"] = (width, 0)
    out["LB"] = (0, -height)
    out["RB"] = (width, -height)
    return out

# ---------------------------------------------------------
class Downloader(Thread):
    # multiple threads downloader
    def __init__(self, index, count, urls, datas):
        # index represents the number of threads
        # count represents the total number of threads
        # urls represents the list of URLs nedd to be downloaded
        # datas represents the list of data need to be returned.
        super().__init__()
        self.urls = urls
        self.datas = datas
        self.index = index
        self.count = count

    def download(self, url):
        HEADERS = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36 Edg/88.0.705.68'}
        header = ur.Request(url, headers=HEADERS)
        err = 0
        while (err < 10):
            try:
                data = ur.urlopen(header).read()
            except:
                err += 1
            else:
                return data
        raise Exception("Bad network link.")

    def run(self):
        for i, url in enumerate(self.urls):
            if i % self.count != self.index:
                continue
            self.datas[i] = self.download(url)

# ---------------------------------------------------------
def getExtent(x1, y1, x2, y2, z, source="Google China"):
    pos1x, pos1y = wgs_to_tile(x1, y1, z)
    pos2x, pos2y = wgs_to_tile(x2, y2, z)
    Xframe = pixls_to_mercator(
        {"LT": (pos1x, pos1y), "RT": (pos2x, pos1y), "LB": (pos1x, pos2y), "RB": (pos2x, pos2y), "z": z})
    for i in ["LT", "LB", "RT", "RB"]:
        Xframe[i] = mercator_to_wgs(*Xframe[i])
    if (source == "Google") or (source == "Google_wt_labels"):
        pass
    elif source == "Google China":
        for i in ["LT", "LB", "RT", "RB"]:
            Xframe[i] = gcj_to_wgs(*Xframe[i])
    else:
        raise Exception("Invalid argument: source.")

    return Xframe

def saveTiff(r, g, b, gt, filePath):
    fname_out = filePath
    driver = gdal.GetDriverByName('GTiff')
    # Create a 3-band dataset
    dset_output = driver.Create(fname_out, r.shape[1], r.shape[0], 3, gdal.GDT_Byte)
    dset_output.SetGeoTransform(gt)
    try:
        proj = osr.SpatialReference()
        proj.ImportFromEPSG(4326)
        dset_output.SetSpatialRef(proj)
    except:
        print("Error: Coordinate system setting failed")
    dset_output.GetRasterBand(1).WriteArray(r)
    dset_output.GetRasterBand(2).WriteArray(g)
    dset_output.GetRasterBand(3).WriteArray(b)
    dset_output.FlushCache()
    dset_output = None
    # print("Image Saved")

# ---------------------------------------------------------
MAP_URLS = {
    "Google": "http://mts0.googleapis.com/vt?lyrs={style}&x={x}&y={y}&z={z}",
    "Google_wt_labels": "http://mts0.googleapis.com/vt?lyrs={style}&x={x}&y={y}&z={z}&apistyle=s.t:0|s.e:l|p.v:off",
    "Google China": "http://mt2.google.cn/vt/lyrs={style}&hl=zh-CN&gl=CN&src=app&x={x}&y={y}&z={z}"}

def get_url(source, x, y, z, style):  #
    if source == 'Google China':
        url = MAP_URLS["Google China"].format(x=x, y=y, z=z, style=style)
    elif source == 'Google':
        url = MAP_URLS["Google"].format(x=x, y=y, z=z, style=style)
    elif source == 'Google_wt_labels':
        url = MAP_URLS["Google_wt_labels"].format(x=x, y=y, z=z, style=style)
    else:
        raise Exception("Unknown Map Source ! ")
    # print(url)
    return url

def get_urls(x1, y1, x2, y2, z, source, style):
    pos1x, pos1y = wgs_to_tile(x1, y1, z)
    pos2x, pos2y = wgs_to_tile(x2, y2, z)

    lenx = pos2x - pos1x + 1
    leny = pos2y - pos1y + 1
    print("Total tiles number：{x} X {y}".format(x=lenx, y=leny))
    # print('get_urls x1 ==>', x1, 'get_urls y1 ==>', y1)
    # print('get_urls x2 ==>', x2, 'get_urls y2 ==>', y2)

    urls = [get_url(source, i, j, z, style) for j in range(pos1y, pos1y + leny) for i in range(pos1x, pos1x + lenx)]
    return urls

# ---------------------------------------------------------
def merge_tiles(datas, x1, y1, x2, y2, z):
    # print('merge_tiles x1 ==>', x1, 'merge_tiles y1 ==>', y1)
    # print('merge_tiles x2 ==>', x2, 'merge_tiles y2 ==>', y2)
    pos1x, pos1y = wgs_to_tile(x1, y1, z)
    pos2x, pos2y = wgs_to_tile(x2, y2, z)
    lenx = pos2x - pos1x + 1
    leny = pos2y - pos1y + 1
    outpic = pil.new('RGBA', (lenx * 256, leny * 256))
    for i, data in enumerate(datas):
        picio = io.BytesIO(data)
        small_pic = pil.open(picio)
        y, x = i // lenx, i % lenx
        outpic.paste(small_pic, (x * 256, y * 256))
    # print('Tiles merge completed')
    return outpic

def download_tiles(urls, multi=10):
    url_len = len(urls)
    datas = [None] * url_len
    if multi < 1 or multi > 20 or not isinstance(multi, int):
        raise Exception("multi of Downloader shuold be int and between 1 to 20.")
    tasks = [Downloader(i, multi, urls, datas) for i in range(multi)]
    for i in tasks:
        i.start()
    for i in tasks:
        i.join()
    return datas

# ---------------------------------------------------------
def main(left, top, right, bottom, zoom, filePath, style='s', server="Google China"):
    """
    Download images based on spatial extent.

    East longitude is positive and west longitude is negative.
    North latitude is positive, south latitude is negative.

    Parameters
    ----------
    left, top : left-top coordinate, for example (100.361,38.866)
        
    right, bottom : right-bottom coordinate
        
    z : zoom

    filePath : File path for storing results, TIFF format
        
    style : 
        m for map; 
        s for satellite; 
        y for satellite with label; 
        t for terrain; 
        p for terrain with label; 
        h for label;
    
    source : Google China (default) or Google
    """
    # ---------------------------------------------------------
    # Получите URL-адреса всех плиток в экстенте
    urls = get_urls(left, top, right, bottom, zoom, server, style)

    # Группируйте URL-адреса в зависимости от количества ядер процессора для выполнения примерно равного объема задач
    urls_group = [urls[i:i + math.ceil(len(urls) / multiprocessing.cpu_count())] for i in
                  range(0, len(urls), math.ceil(len(urls) / multiprocessing.cpu_count()))]

    # Каждый набор URL-адресов соответствует процессу загрузки плиточных карт
    # print('Tiles downloading......')
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    results = pool.map(download_tiles, urls_group)
    pool.close()
    pool.join()
    result = [x for j in results for x in j]
    # print('RESULT ====>', len(result))
    # print('Tiles download complete')

    # Объедините загруженные карты плиток в одну карту
    outpic = merge_tiles(result, left, top, right, bottom, zoom)
    outpic = outpic.convert('RGB')
    r, g, b = cv2.split(np.array(outpic))

    # Получите пространственную информацию о четырех углах объединенной карты и используйте ее для вывода
    extent = getExtent(left, top, right, bottom, zoom, server)
    # print(extent)
    gt = (extent['LT'][0], (extent['RB'][0] - extent['LT'][0]) / r.shape[1], 0, extent['LT'][1], 0,
          (extent['RB'][1] - extent['LT'][1]) / r.shape[0])
    # print(gt)
    if filePath.endswith('tif'):
        saveTiff(r, g, b, gt, filePath)
    elif filePath.endswith(('jpg', 'bmp', 'png')):
        outpic.save(filePath)

# ---------------------------------------------------------
if __name__ == '__main__':
    start_time = time.time()
    # 54.834270, 83.198360 ====== 54.794134, 83.304763
    # main(100.361, 38.866, 100.386, 38.839, 20, 'output.tif', server="Google")
    main(83.195558,
         54.798793,
         83.214901,
         54.786596,
         19,
         'output19.tif',
         style='s',
         server="Google")

    end_time = time.time()
    print('lasted a total of {:.2f} seconds'.format(end_time - start_time))



def undistorton(img: list, crop: bool = True):
    try:
        h, w, _ = img.shape
    except:
        w, h = img.size
    camera_matrix = np.array([[447.27399414, 0, 320],
                              [0, 447.2436399, 180],
                              [0, 0, 1]])
    camera_matrix = np.array([[447.27399414, 0, w//2],
                              [0, 447.2436399, h//2],
                              [0, 0, 1]])
    dist_coefs = np.array([-0.35328536, 0.2090826, 0,  0,  -0.06274637])
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))
    dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    x, y, w, h = roi
    # dst = dst[y:y+h-50, x+70:x+w-20]
    if crop:
        dst = dst[y:y+h,x:x+w]
    dst = Image.fromarray(dst)
    return dst

def write_with_metadata(img: np.array, metadata: str, path: str):
    png_info = PngImagePlugin.PngInfo()
    png_info.add_text('metadata', str(metadata))
    with BytesIO() as output:
        img.save(output, "PNG", pnginfo=png_info)
        binary_data = output.getvalue()
    with open(path, "wb") as file:
        file.write(binary_data)

def resize_and_center_image(img, orig_corners, new_corners):
    new_corners = new_corners.reshape(4, 2).round()
    # print(orig_corners, new_corners)
    # вычисляем высоту и ширину оригинального изображения
    orig_height = orig_corners[3][1] - orig_corners[0][1]
    orig_width = orig_corners[1][0] - orig_corners[0][0]
    # print(orig_width, orig_height)
    # вычисляем высоту и ширину нового изображения
    new_height = int(max(new_corners[3][1], new_corners[2][1]) - min(new_corners[0][1], new_corners[1][1]))
    new_width = int(max(new_corners[1][0], new_corners[2][0]) - min(new_corners[0][0], new_corners[3][0]))
    # print(new_width, orig_height)
    # вычисляем коэффициенты масштабирования для ширины и высоты
    scale_x = new_width / orig_width
    scale_y = new_height / orig_height
    # print(scale_y, scale_x)
    orig_center = [orig_width//2, orig_height//2]
    new_center = [(new_corners[0][0] + new_corners[1][0] + new_corners[2][0] + new_corners[3][0]) / 4,
                  (new_corners[0][1] + new_corners[1][1] + new_corners[2][1] + new_corners[3][1]) / 4]
    shift_vector = np.array(orig_center) - np.array(new_center)

    # Сдвиг второго изображения
    rows, cols, _ = img.shape
    M = np.float32([[1, 0, shift_vector[0]], [0, 1, shift_vector[1]]])
    shifted_image = cv2.warpAffine(img, M, (cols, rows))
    return shifted_image

def get_point_coor(dpp, roll, pitch, compass, relative_alt, camera_viewing_angle, camera_viewing_angle_vert, MAGNETIC_DECLINATION, point):
    relative_alt /= 1000
    if point == 'LT':
        lat_bias = -relative_alt * math.tan(pitch+camera_viewing_angle_vert/2)
        lon_bias = -relative_alt * math.tan(roll+camera_viewing_angle/2)
        distance_polar = (lat_bias ** 2 + lon_bias ** 2) ** 0.5
        angle_polar = math.atan2(lon_bias, lat_bias) + 2*math.pi
        angle_polar_with_compass = angle_polar + (math.radians(compass + MAGNETIC_DECLINATION))
        new_lat_bias = distance_polar * math.cos(angle_polar_with_compass)
        new_lon_bias = distance_polar * math.sin(angle_polar_with_compass)
    elif point == 'RT':
        lat_bias = -relative_alt * math.tan(pitch+camera_viewing_angle_vert/2)
        lon_bias = -relative_alt * math.tan(roll-camera_viewing_angle/2)
        distance_polar = (lat_bias ** 2 + lon_bias ** 2) ** 0.5
        angle_polar = math.atan2(lon_bias, lat_bias) + 2*math.pi
        angle_polar_with_compass = angle_polar + (math.radians(compass + MAGNETIC_DECLINATION))
        new_lat_bias = distance_polar * math.cos(angle_polar_with_compass)
        new_lon_bias = distance_polar * math.sin(angle_polar_with_compass)
    elif point == 'LB':
        lat_bias = -relative_alt * math.tan(pitch-camera_viewing_angle_vert/2)
        lon_bias = -relative_alt * math.tan(roll+camera_viewing_angle/2)
        distance_polar = (lat_bias ** 2 + lon_bias ** 2) ** 0.5
        angle_polar = math.atan2(lon_bias, lat_bias) + 2*math.pi
        angle_polar_with_compass = angle_polar + (math.radians(compass + MAGNETIC_DECLINATION))
        new_lat_bias = distance_polar * math.cos(angle_polar_with_compass)
        new_lon_bias = distance_polar * math.sin(angle_polar_with_compass)
    elif point == 'RB':
        lat_bias = -relative_alt * math.tan(pitch - camera_viewing_angle_vert/2)
        lon_bias = -relative_alt * math.tan(roll - camera_viewing_angle/2)
        distance_polar = (lat_bias ** 2 + lon_bias ** 2) ** 0.5
        angle_polar = math.atan2(lon_bias, lat_bias) + 2 * math.pi
        angle_polar_with_compass = angle_polar + (math.radians(compass + MAGNETIC_DECLINATION))
        new_lat_bias = distance_polar * math.cos(angle_polar_with_compass)
        new_lon_bias = distance_polar * math.sin(angle_polar_with_compass)
    point = compute_new_coordinates(dpp['lat'],
                                    dpp['lon'],
                                    new_lat_bias,
                                    new_lon_bias)
    return point

def resize_with_aspect_ratio(image, new_width=640, new_height=480):
    # print('IMAGE', image.size)
    original_width, original_height = image.size
    aspect_ratio = original_width / original_height
    # print('ASPECT_RATIO', aspect_ratio)
    # Изменяем размер изображения с сохранением соотношения сторон
    if (original_width / new_width) < (original_height / new_height):
        new_width = int(new_height * aspect_ratio)
    else:
        new_height = int(new_width / aspect_ratio)
    resized_image = image.resize((new_width, new_height))
    # print('IMAGE', resized_image.size)
    # _ = Image.new("RGB", (640, 480), color="black")
    # _.paste(img, (0, 0))
    # image1 = _
    return resized_image

def draw_center_of_mass(image):
    try:
        image = np.array(image)
    except:
        pass
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Находим центр масс изображения, игнорируя черные пиксели
    ret, thresh = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
    M = cv2.moments(thresh)
    center_x = int(M['m10'] / M['m00'])
    center_y = int(M['m01'] / M['m00'])

    # Отрисовываем красную точку в центре масс
    # image_with_center = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.circle(image, (center_x, center_y), 15, (255, 0, 0), -1)
    # print(image.shape)
    cv2.circle(image, (image.shape[1]//2, image.shape[0]//2), 20, (0, 255, 0), 5)
    return Image.fromarray(image)
