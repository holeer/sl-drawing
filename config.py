# -*- coding: UTF-8 -*-

class Config(object):
    """配置类"""

    def __init__(self):
        # path config
        self.label_file = 'dataset/tag.txt'
        self.train_file = 'dataset/train/train.csv'
        self.test_file = 'dataset/test/test.csv'
        self.log_dir = 'log/'
        self.target_dir = 'output/'
        self.temp_dir = 'temp/'
        # model config
        self.max_length = 512
        self.batch_size = 32
        self.shuffle = True
        self.drop_last = False
        self.rnn_hidden = 128
        self.bert_embedding = 768
        self.epochs = 20
        self.seed = 42
        self.lr = 0.0001
        self.lr_decay = 0.0001
        self.weight_decay = 0.00005
        self.max_grad_norm = 10
        self.patience = 10
        # checkpoint
        # self.checkpoint = None
        self.checkpoint = 'output/best.pth.tar'

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])


config = Config()

