# -*- coding: UTF-8 -*-
import csv

import cv2
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
import easyocr
from PIL import Image
from config import config
import torchvision.transforms as transforms
import torch
import torchvision.models as models
from torch.autograd import Variable
from transformers import BertTokenizer, BertModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform_list = [transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])]
img_to_tensor = transforms.Compose(transform_list)

res_model = models.resnet101(pretrained=True).to(device)
res_model.fc = torch.nn.Linear(2048, config.bert_embedding).to(device)
res_model.eval()

tokenizer = BertTokenizer.from_pretrained("model/bert-base-chinese/")
bert_model = BertModel.from_pretrained("model/bert-base-chinese/")

# 创建reader对象
reader = easyocr.Reader(['ch_sim', 'en'])

shoudong_x = [26, 643, 1268, 2063, 2233, 2383, 2666, 2835, 2986, 3268, 3438, 3590, 3873, 4041, 4328]
shoudong_y = [33, 150]


def get_classes_num(file):
    with open(file, encoding='utf-8', errors='ignore') as fp:
        lines = fp.readlines()
        lines = [l.strip() for l in lines]
    return len(lines)


def split_bottom_bar(src):
    feature_list = []
    # 读取彩色图像
    raw = cv2.imread(src, 1)
    # 灰度图片
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

    binary = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, -5)
    rows, cols = binary.shape
    scale = 8
    # 自适应获取核值
    # 识别横线:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (cols // scale, 1))
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilated_col = cv2.dilate(eroded, kernel, iterations=1)
    # cv2.imshow("excel_horizontal_line", dilated_col)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 识别竖线：
    scale = 2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, rows // scale))
    eroded = cv2.erode(binary, kernel, iterations=1)
    dilated_row = cv2.dilate(eroded, kernel, iterations=3)
    # cv2.imshow("excel_vertical_line：", dilated_row)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 将识别出来的横竖线合起来
    bitwise_and = cv2.bitwise_and(dilated_col, dilated_row)
    # cv2.imshow("excel_bitwise_and", bitwise_and)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 将焦点标识取出来
    ys, xs = np.where(bitwise_and > 0)
    if len(xs) > 0 and len(ys) > 0:
        # 横纵坐标数组
        y_point_arr = []
        x_point_arr = []
        # 通过排序，排除掉相近的像素点，只取相近值的最后一点
        # 这个10就是两个像素点的距离，不是固定的，根据不同的图片会有调整，基本上为单元格表格的高度（y坐标跳变）和长度（x坐标跳变）
        i = 0
        sort_x_point = np.sort(xs)
        for i in range(len(sort_x_point) - 1):
            if sort_x_point[i + 1] - sort_x_point[i] > 50:
                x_point_arr.append(sort_x_point[i])
            i = i + 1
        # 要将最后一个点加入
        x_point_arr.append(sort_x_point[i])

        i = 0
        sort_y_point = np.sort(ys)
        # print(np.sort(ys))
        for i in range(len(sort_y_point) - 1):
            if sort_y_point[i + 1] - sort_y_point[i] > 10:
                y_point_arr.append(sort_y_point[i])
            i = i + 1
        y_point_arr.append(sort_y_point[i])

        # 循环y坐标，x坐标分割表格
        for i in range(len(y_point_arr) - 1):
            for j in range(len(x_point_arr) - 1):
                # 在分割时，第一个参数为y坐标，第二个参数为x坐标
                y1 = int(y_point_arr[i])
                y2 = int(y_point_arr[i + 1])
                x1 = int(x_point_arr[j])
                x2 = int(x_point_arr[j + 1])
                cell = raw[y1:y2, x1:x2]
                # 读取文字
                result = reader.readtext(cell, canvas_size=4096)
                word_list = []
                for r in result:
                    word_list.append(r[1])
                content = ''.join(word_list).replace(' ', '')
                inputs = tokenizer(content, return_tensors="pt")
                text_output = bert_model(**inputs)[1]

                cv2.imwrite('temp/temp.png', cell)
                img = Image.open('temp/temp.png')
                img = img.resize((224, 224))
                pic = img_to_tensor(img).resize_(1, 3, 224, 224).to(device)
                pic_output = res_model(Variable(pic).to(device)).to(device)
                # cv2.imshow("sub_pic_" + str(j), cell)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                feature = torch.cat((text_output, pic_output), 1)
                feature_list.append(feature)
        if 'train' in src and len(feature_list) != 14:
            feature_list.clear()
            # 循环y坐标，x坐标分割表格
            for i in range(len(shoudong_y) - 1):
                for j in range(len(shoudong_x) - 1):
                    # 在分割时，第一个参数为y坐标，第二个参数为x坐标
                    y1 = int(shoudong_y[i])
                    y2 = int(shoudong_y[i + 1])
                    x1 = int(shoudong_x[j])
                    x2 = int(shoudong_x[j + 1])
                    cell = raw[y1:y2, x1:x2]
                    # 读取文字
                    result = reader.readtext(cell, canvas_size=4096)
                    word_list = []
                    for r in result:
                        word_list.append(r[1])
                    content = ''.join(word_list).replace(' ', '')
                    inputs = tokenizer(content, return_tensors="pt")
                    text_output = bert_model(**inputs)[1]

                    cv2.imwrite('temp/temp.png', cell)
                    img = Image.open('temp/temp.png')
                    img = img.resize((224, 224))
                    pic = img_to_tensor(img).resize_(1, 3, 224, 224).to(device)
                    pic_output = res_model(Variable(pic).to(device)).to(device)
                    # cv2.imshow("sub_pic_" + str(j), cell)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    feature = torch.cat((text_output, pic_output), 1)
                    feature_list.append(feature)
        if 'test' in src and len(feature_list) != 13:
            print(src)
        return feature_list
    else:
        return []


def get_bottom(src, out_path):
    # 读取彩色图像
    raw = cv2.imread(src, 1)
    # train
    # cell = raw[3130:3300, 320:4680]
    # test
    cell = raw[3070:3290, 320:4680]
    if cell.size != 0:
        cv2.imwrite(out_path, cell)
    else:
        print('error:' + src)
    # cv2.imshow("bottom_pic", cell)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    pass

    # 分割底部栏
    # split_bottom_bar('dataset/test/img/0.png')

    # 建立数据集
    # img_list = os.listdir('dataset/test/img/')
    # img_list.sort(key=lambda x: int(x.split('.')[0]))
    # f = open('dataset/test/test.csv', 'w', encoding='utf-8', newline='')
    # cw = csv.writer(f)
    # for i in img_list:
    #     path = 'dataset/test/img/' + i
    #     # label = [0, 1, 2, 3, 3, 8, 4, 4, 8, 5, 5, 8, 6, 6]
    #     label = [0, 1, 2, 3, 8, 4, 8, 5, 8, 6, 6, 7, 7]
    #     cw.writerow([path, ','.join(str(i) for i in label)])
    # f.close()
