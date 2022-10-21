import os
import glob
import random
import shutil
from typing import Tuple

from PIL import Image

import PIL
"""
对所有图片进行RGB转化，并且统一调整大小，但图片不发生变形和扭曲（保留图像原始比例）
划分训练集和测试集
"""


def change_image_size(to_size: int, image_path):
    img = PIL.Image.open(image_path).convert('RGB')
    old_size = img.size  # old_size = (w, h)
    ratio = float(to_size / max(old_size))
    # 等比例缩放
    new_size = (int(old_size[0] * ratio), int(old_size[1] * ratio))
    im = img.resize(new_size, Image.ANTIALIAS)  # ANTIALIAS不会对图片造成模糊
    new_im = PIL.Image.new('RGB', (to_size, to_size))
    # 将原来的图片数据放到中间进来
    new_im.paste(im, ((to_size - new_size[0]) // 2, (to_size - new_size[1]) // 2))
    return new_im


def main():
    test_split_ratio = 0.05  # 测试集比例
    desired_size = 64  # 图片统一大小
    raw_path = 'data/raw'

    dirs = glob.glob(os.path.join(raw_path, '*'))
    dirs = [d for d in dirs if os.path.isdir(d)]

    print(f'totally {len(dirs)}')

    for path in dirs:
        # 对每个类别单独处理
        class_name = path.split('\\')[-1]
        # 创建两个文件夹
        os.makedirs(f'data/train/{class_name}', exist_ok=True)
        os.makedirs(f'data/test/{class_name}', exist_ok=True)

        # 匹配文件夹下所有图片
        files = glob.glob(os.path.join(raw_path, class_name, '*.jpg'))
        files += glob.glob(os.path.join(raw_path, class_name, '*.JPG'))  # 是个list可以相加
        files += glob.glob(os.path.join(raw_path, class_name, '*.png'))

        random.seed(0)
        random.shuffle(files)

        test_sample_num = int(len(files) * test_split_ratio)

        # 遍历每一张图片
        for i, file in enumerate(files):  # 获取遍历的index，通常使用enumerate
            new_im = change_image_size(desired_size, file)

            assert new_im.mode == 'RGB'

            if i <= test_sample_num:
                new_im.save(os.path.join(f'data/test/{class_name}',
                                         file.split('\\')[-1].split('.')[0] + '.jpg'))
            else:
                new_im.save(os.path.join(f'data/train/{class_name}',
                                         file.split('\\')[-1].split('.')[0] + '.jpg'))

    test_file = glob.glob(os.path.join('data/test', '*', '*.jpg'))
    train_file = glob.glob(os.path.join('data/train', '*', '*.jpg'))
    print(f'totally files for test {len(test_file)}')
    print(f'totally files for training {len(train_file)}')


if __name__ == '__main__':
    main()