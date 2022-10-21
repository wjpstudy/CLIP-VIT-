import glob
import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import clip
import torchvision.transforms
import matplotlib.pyplot as plt

import my_execption
import tqdm
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args_parser():
    parser = argparse.ArgumentParser("image search task", add_help=False)
    # model parameters
    parser.add_argument("--input_size", default=64, type=int)
    parser.add_argument("--dataset_dir", default="data/train")
    parser.add_argument("--test_image_dir", default="data/val_images")
    parser.add_argument("--save_dir", default="out_dir")
    parser.add_argument("--model_name", default="clip")  # resnet50, resnet152, clip
    parser.add_argument("--feature_dict_file", default="feature_dict.npy")  # 存储图像表征的字典
    parser.add_argument("--topk", default=7, type=int)
    parser.add_argument("--mode", default="predict")  # extract 图像表征 predict 预测

    return parser


def main():
    from pprint import pprint
    model_names = timm.list_models(pretrained=True)
    pprint(model_names)
    my_execption.success("成功获取预训练模型列表")

    args = get_args_parser()
    args = args.parse_args()  # 获取超参数

    model = None
    preprocess = None

    if args.model_name != "clip":
        model = timm.create_model(args.model_name, pretrained=True)
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        my_execption.header(f"当前所选模型为: {args.model_name},"
                            "可训练参数数量为: %.2f M" % (n_parameters / 1.e6))
        model.eval()
    else:
        pprint(clip.available_models())
        model_name = "ViT-B/32"
        my_execption.success(f"成功获取clip模型列表，当前所用模型：{model_name}")
        model, preprocess = clip.load(model_name, device=device)
        my_execption.success(f"加载模型成功，{model_name}")

    if args.mode == 'extract':
        my_execption.header(f"正在使用{args.model_name}模型提取图像特征")
        all_vectors = extract_features(args, model, image_path=args.dataset_dir, preprocess=preprocess)
        os.makedirs(f"{args.save_dir}/{args.model_name}", exist_ok=True)
        np.save(f"{args.save_dir}/{args.model_name}/{args.feature_dict_file}", all_vectors)
        my_execption.header(f"{args.model_name}模型提取图像特征完成")

    elif args.mode == 'predict':
        # 加载预测图像
        test_images = glob.glob(os.path.join(args.test_image_dir, '*', '*.jpg'))
        # 加载待检索字典,allow_pickle需要为True
        all_vectors = np.load(f"{args.save_dir}/{args.model_name}/{args.feature_dict_file}", allow_pickle=True)
        all_vectors = all_vectors.item()
        assert isinstance(all_vectors, dict)
        # 获取测试图像的表征
        for image_file in tqdm.tqdm(test_images):
            if args.model_name == 'clip':
                all_vectors[image_file] = extract_features_by_CLIP(model, preprocess, image_file)
            else:
                all_vectors[image_file] = extract_features_single(args, model, image_file)
        # 获取余弦相似度
        similarity, keys = get_similarity_matrix(all_vectors)

        # 找到图片最近的topk个图片
        my_execption.header("正在为测试图片的余弦相似度矩阵进行排序")
        res = {}
        for image_file in tqdm.tqdm(test_images):
            index = keys.index(image_file)
            sim_vec = similarity[index]
            # 排序后得到索引，[::-1]对数组翻转
            sort_index = np.argsort(sim_vec)[::-1][1:args.topk]

            sim_images, sim_scores = [], []

            for ind in sort_index:
                sim_images.append(keys[ind])
                sim_scores.append(sim_vec[ind])

            res[image_file] = (sim_images, sim_scores)

        my_execption.success("排序完成,正在进行图片展示")
        for image_file in test_images:
            plot_sim_image(args, image_file, res[image_file][0], res[image_file][1],
                           num_row=1, num_col=args.topk)
    else:
        my_execption.warning(f"args.mode异常，请输入extract或者predict")


def plot_sim_image(args, image_file, sim_images, sim_scores, num_row, num_col):
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    fig.suptitle(f"模型：{args.model_name}", fontsize=35)

    for j in range(0, num_row * num_col):
        if j == 0:
            # query照片
            img = Image.open(image_file)
            ax = fig.add_subplot(num_row, num_col, 1)
            set_axes(ax, image_file.split(os.sep)[-1], query=True)
        else:
            img = Image.open(sim_images[j-1])
            ax = fig.add_subplot(num_row, num_col, j+1)
            set_axes(ax, sim_images[j-1].split(os.sep)[-1], value=sim_scores[j-1])
        img = img.convert('RGB')
        plt.imshow(img)
        img.close()
    plt.savefig(f"{args.save_dir}/"
                f"{args.model_name}_search_top_{args.topk}_{image_file.split(os.sep)[-1].split('.')[0]}.jpg")
    plt.show()


def set_axes(ax, image, query=False, **kwargs):
    value = kwargs.get("value", None)
    if query:
        ax.set_xlabel(f"query: image: {image}", fontsize=12)
        ax.xaxis.label.set_color('red')
    else:
        ax.set_xlabel("score: %.3f" % value + f"\nimage:{image}", fontsize=12)
        ax.xaxis.label.set_color('blue')

    ax.set_xticks([])
    ax.set_yticks([])


def get_similarity_matrix(vectors_dict):
    """
    计算余弦相似度
    :param vectors_dict: 保存的图像表征字典
    :return: 相似度矩阵
    """
    v = np.array(list(vectors_dict.values()))  # [num, dim]
    numerator = np.matmul(v, v.T)
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    denominator = np.matmul(norm, norm.T)

    sim = numerator / denominator
    keys = list(vectors_dict.keys())

    return sim, keys


def extract_features(args, model, image_path='', preprocess=None):
    """
    抽取image_path下面所有的图片的表征
    其中preprocess只有clip模型会用到
    :param args: 超参数
    :param model: 模型
    :param image_path: 存放图片数据集的目录
    :param preprocess: clip模型的预处理函数preprocess
    :return: 特征数组np
    """
    all_vectors = {}

    for image_file in tqdm.tqdm(glob.glob(os.path.join(image_path, '*', '*.jpg'))):
        if args.model_name == 'clip':
            all_vectors[image_file] = extract_features_by_CLIP(model, preprocess, image_file)
        else:
            all_vectors[image_file] = extract_features_single(args, model, image_file)
    return all_vectors


def extract_features_by_CLIP(model, preprocess, image_file):
    """
    使用CLIP的模型抽取图像的表征
    :param model: clip的模型
    :param preprocess: 对照片进行预处理函数
    :param image_file: 照片路径
    :return: 图片表征 shape:(512,)
    """
    # 生一维度，bs维
    image = preprocess(Image.open(image_file)).unsqueeze(0).to(device)
    with torch.no_grad():
        vec = model.encode_image(image)
        # 再把维度降回去
        vec = vec.squeeze().numpy()

    return vec


def extract_features_single(args, model, file):
    """
    resnet模型的图片表征提取
    :param args: 超参数
    :param model: timm模型
    :param file: 图片文件
    :return: np数组 图片表征(2048,)
    """
    # 手动归一化
    img_rgb = Image.open(file).convert('RGB')
    image = img_rgb.resize((args.input_size, args.input_size), Image.ANTIALIAS)
    # 整型转换为浮点型
    image = torchvision.transforms.ToTensor()(image)
    mean = [0.47043142, 0.43239759, 0.32576062]
    std = [0.37251864, 0.35710091, 0.3417497]
    # resnet要求图片是做了一个标准化的，因此我们要做一个trans
    image = torchvision.transforms.Normalize(mean=mean, std=std)(image).unsqueeze(0)
    with torch.no_grad():
        features = model.forward_features(image)
        vec = model.global_pool(features)
        vec = vec.squeeze().numpy()
    img_rgb.close()
    return vec


if __name__ == '__main__':
    main()