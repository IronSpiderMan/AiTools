import hashlib
from collections import namedtuple

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
    :param nt:
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


if __name__ == '__main__':
    # print(str2md5('dfsadf'))
    Result = namedtuple('Result', ['f1', 'f2'])
    r1 = Result(1, 2)
    print(namedtuple2json(r1))
