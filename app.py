import os
import cv2
import torch
from flask import Flask, request, render_template
from gevent import pywsgi
from PIL import Image
from torchvision import transforms
from hubs.model import MattingNetwork
from models.common import UploadResult, SegmentResult, DownloadResult, SteganographyResult
from utils import namedtuple2json, center_crop
from utils import steganography as ste, anti_steganography as anti_ste

app = Flask(__name__)

HOST = "http://39.100.68.34:8000"
STATIC_PATH = "/static"
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
        f.save(os.path.join('static', 'segments', f.filename))
        result = UploadResult('success', '上传成功', STATIC_PATH + "/" + f.filename)
        return namedtuple2json(result)


@app.route('/api/download_image', methods=['POST'])
def download_image():
    if request.method != 'POST':
        result = DownloadResult('failed', '不支持的请求方式', None)
    else:
        # 获取请求参数
        pass


@app.route('/api/steganography', method=['POST'])
def steganography():
    if request.method != 'POST':
        result = SteganographyResult('failed', '不支持的请求方式', None)
        return namedtuple2json(result)
    else:
        # 获取上传的图片
        f = request.files['file']
        fpath = os.path.join(STEGANOGRAPHY_ORIGIN_PATH, f.filename)
        f.save(fpath)
        # 读取图片开始隐写
        image = cv2.imread(fpath)
        ste_save_path = os.path.join(STEGANOGRAPHY_STE_PATH, f.filename)
        if not ste(image, '', ste_save_path):
            result = SteganographyResult('failed', '隐写失败', None)
        else:
            result = SteganographyResult('success', '隐写成功', STEGANOGRAPHY_STE_URL + "/" + f.filename)
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
        result = SegmentResult('success', '抠图成功', os.path.join(HOST + SEGMENT_PATH, save_name))
        return namedtuple2json(result)


if __name__ == '__main__':
    server = pywsgi.WSGIServer(('0.0.0.0', 8000), app)
    server.serve_forever()
