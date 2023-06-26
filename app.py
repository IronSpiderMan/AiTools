import json
import os
import cv2
import torch
from flask import Flask, request, render_template
from gevent import pywsgi
from PIL import Image
from torchvision import transforms
from hubs.model import MattingNetwork

from models.common import UploadResult

app = Flask(__name__)

STATIC_PATH = "http://39.100.68.34:8000" + "/static"

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


@app.route('/api/segment')
def segment():
    # 读取图片，转换成Tensor
    src = Image.open('../static/segments/001/e716048bd4ede0ab02440ec1f28f00c3.jpeg')
    src = (transforms.PILToTensor()(src) / 255.)[None]
    # 抠图
    with torch.no_grad():
        fgr, pha, *rec = segmentor(src)
        segmented = torch.cat([src, pha], dim=1).squeeze(0).permute(1, 2, 0).numpy()
        segmented = cv2.normalize(segmented, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        Image.fromarray(segmented).show()
    return 'Hello'


if __name__ == '__main__':
    server = pywsgi.WSGIServer(('0.0.0.0', 8000), app)
    server.serve_forever()
