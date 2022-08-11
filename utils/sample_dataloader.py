import torch
import torch.nn
import numpy as np



def sample(data,sample_num):
    '''
    :param data:[[]]
    :param sample_num:
    :return: list[],label:[]
    '''
    label_num = len(data)
    x_list=[]
    label_list = []
    for label in range(label_num):
        sample_index = np.random.randint(0,len(data[label]),sample_num)
        for i in sample_index:
            x_list.append(data[label][i])
            label_list.append(label)
    return x_list,label_list






