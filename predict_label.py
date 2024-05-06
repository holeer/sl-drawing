# -*- coding: UTF-8 -*-
import torch
from config import config
import utils
from model_label import CNN
from api_call import baidu_ocr
from tqdm import tqdm
import os
import pprint


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ExtractInfo:

    def __init__(self):
        self.model = None
        self.vocabulary = utils.get_vocabulary(config.label_file)
        self.get_model()  # 模型准备

    def get_model(self):
        self.model = CNN(len(self.vocabulary), device).to(device)
        checkpoint = torch.load(config.checkpoint)
        self.model.load_state_dict(checkpoint["model"],strict=False)
        self.model.eval()

    def predict(self, pic, token):
        content = ''
        texts, probabilities = baidu_ocr.ocr(token, pic)
        for index, p in enumerate(probabilities):
            if p > 0.75:
                content += texts[index].replace(' ', '')
        with torch.no_grad():
            logit = self.model([pic], [content])
            logit = torch.argmax(logit).item()
        label = self.vocabulary[logit]
        return content, label


def predict(pic, token):
    result = []
    extract = ExtractInfo()

    utils.split_bottom_bar(pic)

    img_list = os.listdir(config.temp_dir)
    img_list.sort(key=lambda x: int(x.split('.')[0]))
    for i in tqdm(img_list):
        temp = {}
        content, label = extract.predict(config.temp_dir + i, token)
        temp[label] = content
        result.append(temp)
    attn = extract.model.attn
    if attn.heatmap:
        print('cnt=%d' % attn.cnt)
        
    return result


if __name__ == '__main__':
    token = baidu_ocr.fetch_token()
    result = predict('demo.png', token)
    pprint.pprint(result)
    