import traceback

import cv2
import hashlib
import qrcode
import numpy as np
from PIL import ImageOps, Image

from settings import RESOLUTION

SALT = "Zack Fair"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
}


def pixel2char(pixel):
    char_list = "@#$%&erytuioplkszxcv=+---.     "
    index = int(pixel / 256 * len(char_list))
    return char_list[index]


def img2char(img, scale=5, font_size=15):
    # 调整图片大小
    h, w = img.shape
    re_im = cv2.resize(img, (w // scale, h // scale))
    # 创建一张图片用来填充字符
    char_img = np.ones((h // scale * font_size, w // scale * font_size), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 遍历图片像素
    for y in range(0, re_im.shape[0]):
        for x in range(0, re_im.shape[1]):
            char_pixel = pixel2char(re_im[y][x])
            cv2.putText(char_img, char_pixel, (x * font_size, y * font_size), font, 0.5, (0, 0, 0))
    return char_img


def hide_qr(o, content, output_path):
    h, w, *_ = o.shape
    q = qrcode.make(content)
    q = np.array(q, dtype=np.uint8) * 255
    q = cv2.resize(q, (w, h))
    q = q // 2
    o = o // 2 + 128
    a = 255 - (o - q)
    c = np.uint8(q / (a / 255 + 1e-3))
    imgn = cv2.cvtColor(c, cv2.COLOR_GRAY2BGRA)
    imgn[:, :, 3] = a
    cv2.imwrite(output_path, imgn, [cv2.IMWRITE_JPEG_QUALITY, 100])


def steganography(image, content, save_path):
    """
    图像隐写，把content生成二维码，写入image，保存到save_path
    :param image: 原图的numpy数组
    :param content: 要隐写的内容，即二维码扫出的内容
    :param save_path: 保存路径 /static/steganography/ste 下
    :return: 如果成功，返回save_path，否则返回None
    """
    try:
        h, w, c = image.shape
        if c > 3:
            image = image[:, :, :3]
        qr = qrcode.make(content)
        qr = np.array(qr, dtype=np.uint8)
        qr = cv2.resize(qr, (w, h))
        qr = np.expand_dims(qr, 2).repeat(3, axis=2)
        layer0 = cv2.bitwise_and(image, 1)
        image = cv2.add(
            cv2.subtract(image, layer0),
            qr
        )
        cv2.imwrite(save_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        return save_path
    except Exception as e:
        traceback.print_exc()
        return None


def anti_steganography(image, qr_path):
    """
    反隐写
    :param image: 带有隐写信息的图片numpy数组
    :param qr_path: 解析出来的隐写二维码保存路径，在 /static/steganography/qrcode 下
    :return:
    """
    layer0 = cv2.bitwise_and(image, 1)[:, :, 0]
    layer0[layer0 == 1] = 255
    cv2.imwrite(qr_path, layer0)


def str2md5(dirname):
    """
    对字符串进行md5加密
    :param dirname: 需要加密的字符串
    :return:
    """
    hl = hashlib.md5()
    hl.update(SALT.encode(encoding="utf-8"))
    hl.update(dirname.encode(encoding="utf-8"))
    return hl.hexdigest()


def namedtuple2json(nt):
    """
    把简单的namedtuple转换成json数据
    :param nt: namedtuple对象
    :return:
    """
    nt_json = {}
    for field in nt._fields:
        attr = getattr(nt, field)
        if isinstance(attr, (int, str, float)):
            nt_json[field] = attr
        else:
            nt_json[field] = None
    return nt_json


def center_crop(image):
    """
    接收PIL中的Image对象，缩放并中心截取，保证尺寸为证件照常见尺寸
    :param image:
    :return:
    """
    w, h = image.size
    # 计算目标分辨率
    size = w // 295
    size = 2 if size > 2 else size
    resolution = RESOLUTION[size]
    # 图像缩放
    resized_image = ImageOps.fit(image, resolution, Image.LANCZOS)
    # 计算裁剪位置
    w, h = resized_image.size
    left = (w - resolution[0]) // 2
    top = (h - resolution[1]) // 2
    right = left + resolution[0] - 1
    bottom = top + resolution[1] - 1
    # 中心裁剪
    return resized_image.crop((left, top, right, bottom))


if __name__ == '__main__':
    # print(str2md5('dfsadf'))
    # test nt2json
    # Result = namedtuple('Result', ['f1', 'f2'])
    # r1 = Result(1, 2)
    # print(namedtuple2json(r1))
    image = Image.open(r"D:\Files\图片\图片素材\人物\小松菜奈\d5a30a07e1f855b1558edc93b2c240ba.png")
    center_crop(image).show()
