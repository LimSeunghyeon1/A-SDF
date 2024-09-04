import torch
import numpy as np

'''
인스턴스를 받고, 만약 그 인스턴스가 교체가 필요하다면 라벨을 교체한다.
major한 라벨 이외에는 모두 -100처리 한다. # ignore labels

pkl파일을 불러온다.
pkl 파일 형식: {instance_num: [new_label0, new_label1, new_label2, ...]}
    
'''
class CheckLabelInstance:
    def __init__(self, check_file_path):
        self.check_file = np.load(check_file_path, allow_pickle=True)        
    def check_data_labels(self, instance_num, labels):
        '''
        labels: N
        '''
        assert labels.type() == np.ndarray, "we only support numpy ndarrays."
        assert len(labels.shape) == 1, labels.shape
        #NOTICE: label이 0부터 시작이어야 한다. 현재 partnet-mobility는 1부터 마킹되어있기 때문에 input으로 줄때 -1해야한다.
        assert labels.min() == 0, "labels should be marked 0, 1, 2, ... (starts with 0)"
        if instance_num in self.check_file.keys():
            label_list = self.check_file[instance_num]
            new_labels = np.full_like(labels,-100)
            unique_labels = np.unique(labels)
            for ul in unique_labels:
                new_labels[labels == ul] = label_list[ul]            
            return new_labels
        else:
            return labels
    
    