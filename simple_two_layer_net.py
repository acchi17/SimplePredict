# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import h5py
from functions import *


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 重みの初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def showWeightSize(self):
        print(self.params['W1'].shape)
        print(self.params['b1'].shape)
        print(self.params['W2'].shape)
        print(self.params['b2'].shape)

    def getWeightsFromFile(self, filePath):
        weights = []
        def getDataset(name, obj):
          if isinstance(obj, h5py.Dataset):
              nonlocal weights
              weights.append(obj[()])
    
        f = h5py.File(filePath, 'r')
        f.visititems(getDataset)
        self.params['b1'] = weights[0]
        self.params['W1'] = weights[1]
        self.params['b2'] = weights[2]
        self.params['W2'] = weights[3]


    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
    
        a1 = np.dot(x, W1) + b1
        z1 = relu(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        
        return y
    

