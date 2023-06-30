import hashlib
from PIL import ImageOps, Image

from settings import RESOLUTION

SALT = "Zack Fair"


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
