# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import BertTokenizer, BertModel, PreTrainedTokenizerFast
from PIL import Image
from torch.autograd import Variable
from attention import SelfAttention
import utils

TEXTCNN = 'model/shibing624--text2vec-base-chinese'
HEATMAP = True

transform_list = [transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])]
img_to_tensor = transforms.Compose(transform_list)


class CNN(nn.Module):
    def __init__(self, num_classes, device):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.fc_temp = nn.Linear(config.bert_embedding * 2, config.bert_embedding)
        self.fc = nn.Linear(config.bert_embedding * 2, self.num_classes)
        # self.bert = BertModel.from_pretrained("model/bert-base-chinese/").to(device)
        # self.tokenizer = BertTokenizer.from_pretrained("model/bert-base-chinese/")
        self.tokenizer = BertTokenizer.from_pretrained(TEXTCNN)
        self.textCNN = BertModel.from_pretrained(TEXTCNN).to(device)
        # self.alexnet = models.alexnet().to(device)
        self.img_fc = torch.nn.Linear(1000, config.bert_embedding).to(device)
        self.vgg = models.vgg16().to(device)
        self.attn = SelfAttention(config.bert_embedding, config.bert_embedding, config.bert_embedding, HEATMAP)

        # self.vgg.eval()
        # self.word2vec.eval()
        # self.resnet.eval()
        # self.bert.eval()

    def forward(self, drawing, content):
        label = 0

        # Attention
        v_mixed = torch.empty(1, config.bert_embedding).to(self.device)
        t_mixed = torch.empty(1, config.bert_embedding).to(self.device)
        for i in range(len(drawing)):
            img = Image.open(drawing[i])
            img = img.resize((512, 128))
            pic = img_to_tensor(img).resize_(1, 3, 512, 128).to(self.device)
            # pic_feature = self.alexnet(Variable(pic).to(self.device))
            pic_feature = self.vgg(Variable(pic).to(self.device))
            pic_feature = self.img_fc(pic_feature)
            # inputs = self.tokenizer(content[i], return_tensors="pt")
            # text_feature = self.bert(**inputs.to(self.device))[1].to(self.device)
            # text_feature = self.word2vec(**inputs.to(self.device))[1].to(self.device)
            if len(content[i]) > 0:
                inputs = self.tokenizer(content[i], return_tensors="pt")
                # text_feature = self.bert(**inputs.to(self.device))[1].to(self.device)
                text_feature = self.textCNN(**inputs.to(self.device))[1].to(self.device)
            else:
                text_feature = torch.zeros(1, config.bert_embedding).to(self.device)
        
            if label == 0:
                v_mixed = pic_feature
                t_mixed = text_feature
                label = 1
            else:
                v_mixed = torch.cat((v_mixed, pic_feature), dim=0)
                t_mixed = torch.cat((t_mixed, text_feature), dim=0)
        v_attn_out = self.attn(v_mixed.unsqueeze(1)).squeeze(1)
        t_attn_out = self.attn(t_mixed.unsqueeze(1)).squeeze(1)
        v_out = torch.cat((v_mixed, v_attn_out), dim=1)
        t_out = torch.cat((t_mixed, t_attn_out), dim=1)
        v_out = self.fc_temp(v_out)
        t_out = self.fc_temp(t_out)
        out = torch.cat((v_out, t_out), dim=1)
        logit = self.fc(out)
        return logit

        # # No Attention
        # mixed = torch.empty(1, config.bert_embedding * 2).to(self.device)
        # for i in range(len(drawing)):
        #     img = Image.open(drawing[i])
        #     img = img.resize((512, 128))
        #     pic = img_to_tensor(img).resize_(1, 3, 512, 128).to(self.device)
        #     pic_feature = self.resnet(Variable(pic).to(self.device))
        #     # pic_feature = self.vgg(Variable(pic).to(self.device))
        #     # pic_feature = self.vgg_fc(pic_feature)
        #     if len(content[i]) > 0:
        #         inputs = self.tokenizer(content[i], return_tensors="pt")
        #         # text_feature = self.word2vec(**inputs.to(self.device))[1].to(self.device)
        #         text_feature = self.bert(**inputs.to(self.device))[1].to(self.device)
        #     else:
        #         text_feature = torch.zeros(1, config.bert_embedding).to(self.device)
        #     mix = torch.cat((pic_feature, text_feature), dim=1)
        #     if label == 0:
        #         mixed = mix
        #         label = 1
        #     else:
        #         mixed = torch.cat((mixed, mix), dim=0)

        # logit = self.fc(mixed)
        # return logit

    def loss(self, logit, label):
        loss_value = F.cross_entropy(logit, label)
        return loss_value


if __name__ == '__main__':
    from thop import profile
    from torchstat import stat

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # vgg = models.vgg16()
    # word2vec = BertModel.from_pretrained("nicoladecao/msmarco-word2vec256000-bert-base-uncased")
    # resnet = models.resnet101()
    # bert = BertModel.from_pretrained("model/bert-base-chinese/")
    model = CNN(7, device)
    # input = torch.randn(1, 3, 224, 224)
    # flops, params = profile(resnet, inputs=(input,))
    # print(params)

    # print(stat(vgg, (3, 224, 224)))
    print(utils.get_parameter_number(model))
