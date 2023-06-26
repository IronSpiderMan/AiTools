import torch
from flask import Flask, request
from gevent import pywsgi
from torch.utils.data import DataLoader
from torchvision import transforms

from hubs.inference_utils import ImageSequenceReader
from hubs.model import MattingNetwork

app = Flask(__name__)

# 加载模型
segmentor = MattingNetwork('mobilenetv3').eval().cuda()
segmentor.load_state_dict(torch.load('static/models/rvm_mobilenetv3.pth'))
segmentor.eval()


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/api/upload_image', methods=['GET'])
def upload_image():
    # f = request.files['file']
    # print(f)
    return "上传"


@app.route('/api/segment')
def segment():
    source = ImageSequenceReader('static/segments/001', transforms.ToTensor())
    reader = DataLoader(source, batch_size=12, pin_memory=True)
    src = next(reader)
    fgr, pha, *rec = segmentor(src)
    print(fgr)
    return 'Hello'


if __name__ == '__main__':
    server = pywsgi.WSGIServer(('127.0.0.1', 5000), app)
    server.serve_forever()
