import os
import cv2
import torch
import time
from flask import Flask, request, render_template
from gevent import pywsgi
from urllib.parse import urlparse
from PIL import Image
from torchvision import transforms
from hubs.model import MattingNetwork
from models.common import UploadResult, SegmentResult, DownloadResult, SteganographyResult
from utils import namedtuple2json, center_crop, hide_qr as hqr, str2md5, img2char as m2c
from utils import steganography as ste, anti_steganography as anti_ste

app = Flask(__name__)

# HOST = "http://39.100.68.34:8000"
HOST = "http://127.0.0.1:8000"
STATIC_PATH = "./static"
STATIC_URL = HOST + "/static"
IMAGES_PATH = os.path.join(STATIC_PATH, 'images')
IMAGES_URL = HOST + "/static/images"
SEGMENT_PATH = os.path.join(STATIC_PATH, 'segments')
SEGMENT_URL = HOST + "/static/segments"
STEGANOGRAPHY_ORIGIN_PATH = os.path.join(STATIC_PATH, 'steganography', 'origin')
STEGANOGRAPHY_ORIGIN_URL = HOST + '/static/steganography/origin'
STEGANOGRAPHY_QRCODE_PATH = os.path.join(STATIC_PATH, 'steganography', 'qrcode')
STEGANOGRAPHY_QRCODE_URL = HOST + '/static/steganography/qrcode'
STEGANOGRAPHY_STE_PATH = os.path.join(STATIC_PATH, 'steganography', 'ste')
STEGANOGRAPHY_STE_URL = HOST + '/static/steganography/ste'

# 加载模型
segmentor = MattingNetwork('mobilenetv3').eval()
segmentor.load_state_dict(torch.load('static/models/rvm_mobilenetv3.pth'))


@app.route('/')
def hello_world():
    print(os.path.join(HOST + SEGMENT_PATH, '111.jpg'))
    result = UploadResult('success', '上传成功', STATIC_PATH)
    return namedtuple2json(result)


@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    # 测试页面
    if request.method == 'GET':
        return render_template('upload_image.html')
    elif request.method == 'POST':
        f = request.files['file']
        fpath = os.path.join(IMAGES_PATH, f.filename)
        f.save(fpath)
        result = UploadResult('success', '上传成功', IMAGES_URL + "/" + f.filename)
        return namedtuple2json(result)


@app.route('/api/hide_qr', methods=['GET', 'POST'])
def hide_qr():
    # 测试页面
    if request.method == 'GET':
        return render_template('upload_image.html')
    elif request.method == 'POST':
        f = request.files['file']
        content = request.values.get('content')
        filename = str2md5(str(time.time())) + ".jpg"
        fpath = os.path.join(IMAGES_PATH, filename)
        f.save(fpath)
        origin = cv2.imread(fpath, 0)
        qr_save_path = fpath.replace(".jpg", ".png")
        hqr(origin, content, qr_save_path)
        result = UploadResult('success', '隐藏成功', IMAGES_URL + "/" + filename.replace(".jpg", ".png"))
        return namedtuple2json(result)


@app.route('/api/img2char', methods=['GET', 'POST'])
def img2char():
    # 测试页面
    if request.method == 'GET':
        return render_template('img2char.html')
    elif request.method == 'POST':
        f = request.files['file']
        filename = str2md5(str(time.time())) + ".jpg"
        fpath = os.path.join(IMAGES_PATH, filename)
        f.save(fpath)
        origin = cv2.imread(fpath, 0)
        char_img = m2c(origin)
        savepath = os.path.join(IMAGES_PATH, "char_" + filename)
        cv2.imwrite(savepath, char_img)
        result = UploadResult('success', '隐藏成功', IMAGES_URL + "/" + "char_" + filename)
        return namedtuple2json(result)


@app.route('/api/download_image', methods=['POST'])
def download_image():
    if request.method != 'POST':
        result = DownloadResult('failed', '不支持的请求方式', None)
    else:
        # 获取请求参数
        pass


@app.route('/api/steganography', methods=['GET', 'POST'])
def steganography():
    if request.method != 'POST':
        result = SteganographyResult('failed', '不支持的请求方式', None)
        return namedtuple2json(result)
    else:
        image_url = request.values.get('image_url')
        content = request.values.get('content')
        path = urlparse(image_url).path
        image = cv2.imread("." + path, cv2.IMREAD_UNCHANGED)
        filename = os.path.basename(image_url)
        filename = filename.replace(filename.split(".")[-1], 'png')
        ste_save_path = os.path.join(STEGANOGRAPHY_STE_PATH, filename)
        if not ste(image, content, ste_save_path):
            result = SteganographyResult('failed', '隐写失败', None)
        else:
            result = SteganographyResult('success', '隐写成功', STEGANOGRAPHY_STE_URL + "/" + filename)
        return namedtuple2json(result)


@app.route('/api/anti_steganography', methods=['POST'])
def anti_steganography():
    if request.method != 'POST':
        result = SteganographyResult('failed', '不支持的请求方式', None)
        return namedtuple2json(result)
    else:
        # 获取上传的图片
        image_url = request.values.get('image_url')
        path = urlparse(image_url).path
        image = cv2.imread("." + path, cv2.IMREAD_UNCHANGED)
        filename = os.path.basename(image_url)
        filename = filename.replace(filename.split(".")[-1], 'png')
        # 读取图片开始解隐写
        qr_save_path = os.path.join(STEGANOGRAPHY_QRCODE_PATH, filename)
        anti_ste(image, qr_save_path)
        result = SteganographyResult('success', '解析成功', STEGANOGRAPHY_QRCODE_URL + "/" + filename)
        return namedtuple2json(result)


@app.route('/api/segment', methods=['POST'])
def segment():
    if request.method != 'POST':
        result = SegmentResult('failed', '不支持的请求方式', None)
        return namedtuple2json(result)
    else:
        # 获取上传的图片
        f = request.files['file']
        fpath = os.path.join(SEGMENT_PATH, f.filename)
        f.save(fpath)
        # 读取图片，中心裁剪，转换成Tensor
        src = Image.open(fpath)
        src = center_crop(src)
        src = (transforms.PILToTensor()(src) / 255.)[None]
        # 抠图
        with torch.no_grad():
            fgr, pha, *rec = segmentor(src)
            segmented = torch.cat([src, pha], dim=1).squeeze(0).permute(1, 2, 0).numpy()
            segmented = cv2.normalize(segmented, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            # 保存路径
            save_name = "segment_" + f.filename.split(".")[0] + ".png"
            save_path = os.path.join(SEGMENT_PATH, save_name)
            Image.fromarray(segmented).save(save_path)
        result = SegmentResult('success', '抠图成功', SEGMENT_URL + "/" + save_name)
        return namedtuple2json(result)


if __name__ == '__main__':
    server = pywsgi.WSGIServer(('0.0.0.0', 8000), app)
    server.serve_forever()
