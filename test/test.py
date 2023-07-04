import json

import cv2
import numpy as np
import torch
from PIL import Image
import qrcode
from collections import namedtuple
from utils import steganography, anti_steganography
from torchvision import transforms
from hubs.model import MattingNetwork


def segment_test():
    # 加载模型
    segmentor = MattingNetwork('mobilenetv3').eval()
    segmentor.load_state_dict(torch.load('../static/models/rvm_mobilenetv3.pth'))

    src = Image.open('../static/segments/001/e716048bd4ede0ab02440ec1f28f00c3.jpeg')
    src = (transforms.PILToTensor()(src) / 255.)[None]
    with torch.no_grad():
        fgr, pha, *rec = segmentor(src)
        segmented = torch.cat([src, pha], dim=1).squeeze(0).permute(1, 2, 0).numpy()
        segmented = cv2.normalize(segmented, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        Image.fromarray(segmented).show()


def steganography_test():
    image = cv2.imread('../static/images/7wywvVpYo8jYd3f3eeb9f62ffc709c51555384af8bf3.jpg')
    steganography(image, '111', '111.png')
    image = cv2.imread('111.png', cv2.IMREAD_UNCHANGED)
    anti_steganography(image, 'qr.png')


if __name__ == '__main__':
    steganography_test()
