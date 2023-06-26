import os
import time
import torch
from flask import Flask, request, render_template
from gevent import pywsgi
from torch.utils.data import DataLoader
from torchvision import transforms
from hubs.inference_utils import ImageSequenceReader
from hubs.model import MattingNetwork

app = Flask(__name__)

# 加载模型
segmentor = MattingNetwork('mobilenetv3').eval()
segmentor.load_state_dict(torch.load('static/models/rvm_mobilenetv3.pth'))


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/api/upload_image', methods=['POST', 'GET'])
def upload_image():
    if request.method == 'GET':
        return render_template('upload_image.html')
    elif request.method == 'POST':
        f = request.files['file']
        name = str(time.time())
        os.mkdir(os.path.join('static', 'segments', name))
        name = os.path.join('static', 'segments', name, f.filename)
        f.save(name)
        return render_template('upload_image.html', context={"image_url": name})


@app.route('/api/segment')
def segment():
    source = ImageSequenceReader('static/segments/001', transforms.ToTensor())
    reader = DataLoader(source, batch_size=12, pin_memory=True)
    src = next(reader)
    fgr, pha, *rec = segmentor(src)
    print(fgr)
    return 'Hello'


if __name__ == '__main__':
    server = pywsgi.WSGIServer(('0.0.0.0', 8000), app)
    server.serve_forever()
