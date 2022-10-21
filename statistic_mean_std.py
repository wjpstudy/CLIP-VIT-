
import os
import glob
import random
import shutil
import numpy as np
from PIL import Image

"""计算图片的均值和方差"""
if __name__ == '__main__':
    train_files = glob.glob(os.path.join('data/train', '*', '*.jpg'))

    result = []

    for file in train_files:
        img = Image.open(file).convert('RGB')
        img = np.array(img).astype(np.uint8)
        img = img/255.  # 变为0~1之间
        result.append(img)

    print(np.shape(result))  # [BS, H, W, C]
    mean = np.mean(result, axis=(0, 1, 2))
    std = np.std(result, axis=(0, 1, 2))
    print(mean)
    print(std)
