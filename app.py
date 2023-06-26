import json
import os
import cv2
import torch
from flask import Flask, request, render_template
from gevent import pywsgi
from PIL import Image
from torchvision import transforms
from hubs.model import MattingNetwork

from models.common import UploadResult, SegmentResult

app = Flask(__name__)

STATIC_URL = "http://39.100.68.34:8000" + "/static"
STATIC_PATH = "./static"
SEGMENT_PATH = os.path.join(STATIC_PATH, 'segments')

# 加载模型
segmentor = MattingNetwork('mobilenetv3').eval()
segmentor.load_state_dict(torch.load('static/models/rvm_mobilenetv3.pth'))


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    # 测试页面
    if request.method == 'GET':
        return render_template('upload_image.html')
    elif request.method == 'POST':
        f = request.files['file']
        f.save(os.path.join('static', 'segments', f.filename))
        result = UploadResult('success', '上传成功', STATIC_PATH + "/" + f.filename)
        return json.dumps(result)


@app.route('/api/segment', methods=['POST'])
def segment():
    if request.method != 'POST':
        result = SegmentResult('failed', '不支持的请求方式', None)
        return json.dumps(result)
    else:
        # 获取上传的图片
        f = request.files['file']
        fpath = os.path.join(SEGMENT_PATH, f.filename)
        f.save(fpath)
        # 读取图片，转换成Tensor
        src = Image.open(fpath)
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
        result = SegmentResult('success', '抠图成功', os.path.join(STATIC_URL, save_name))
        return json.dumps(result)


if __name__ == '__main__':
    server = pywsgi.WSGIServer(('0.0.0.0', 8000), app)
    server.serve_forever()
