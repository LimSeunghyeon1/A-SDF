#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

import glob
import logging
import numpy as np
import os
import random
import torch
import torch.utils.data

import asdf.workspace as ws
import re
from plyfile import PlyData
import json
from tools.switch_label import to_switch_label
import time




def get_instance_filenames(data_source, split):
    npzfiles = []
    for dataset in split:
        for class_name in split[dataset]:
            for instance_name in split[dataset][class_name]:
                instance_filename = os.path.join(
                    dataset, class_name, instance_name + ".npz"
                )
                if not os.path.isfile(
                    os.path.join(data_source, ws.sdf_samples_subdir, instance_filename)
                ):
                    # raise RuntimeError(
                    #     'Requested non-existent file "' + instance_filename + "'"
                    # )
                    logging.warning(
                        "Requested non-existent file '{}'".format(instance_filename)
                    )
                npzfiles += [instance_filename]
    return npzfiles

def get_instance_filenames_bi(data_source, pkl_path, category, split):
    '''
    pkl file에 있는 것들 로드
    '''
    # plyfiles = []
    # for dataset in split:
    #     for class_name in split[dataset]:
    #         for instance_name in split[dataset][class_name]:
    #             instance_filename = os.path.join(
    #                 dataset, class_name, instance_name + ".npz"
    #             )
    #             if not os.path.isfile(
    #                 os.path.join(data_source, ws.sdf_samples_subdir, instance_filename)
    #             ):
    #                 # raise RuntimeError(
    #                 #     'Requested non-existent file "' + instance_filename + "'"
    #                 # )
    #                 logging.warning(
    #                     "Requested non-existent file '{}'".format(instance_filename)
    #                 )
    #             npzfiles += [instance_filename]
    # return npzfiles
    total_valid_paths = []
    # dir = self.data_source
    data_dict = np.load(pkl_path, allow_pickle=True)
    for cat in data_dict.keys():
        if cat != category: continue
        for spt in data_dict[cat].keys():
            if split == 'trn':
                raise NotImplementedError
                if spt == 'test': continue
            else:
                assert split == 'test'
                if spt == 'train' or spt == 'val': continue
            instances = data_dict[cat][spt]
            for instance in instances:
                for i in range(100):
                    total_valid_paths.append(os.path.join(data_source, spt, cat, str(instance), f"pose_{i}", "points_with_sdf_label_binary.ply"))
        
                
    return total_valid_paths    

class NoMeshFileError(RuntimeError):
    """Raised when a mesh file is not found in a shape directory"""

    pass


class MultipleMeshFileError(RuntimeError):
    """"Raised when a there a multiple mesh files in a shape directory"""

    pass


def find_mesh_in_directory(shape_dir):
    mesh_filenames = list(glob.iglob(shape_dir + "/**/*.obj")) + list(
        glob.iglob(shape_dir + "/*.obj")
    )
    if len(mesh_filenames) == 0:
        raise NoMeshFileError()
    elif len(mesh_filenames) > 1:
        raise MultipleMeshFileError()
    return mesh_filenames[0]


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]



def read_sdf_samples_into_ram(filename, articulation=False, num_atc_parts=1):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])
    if articulation==True:
        if num_atc_parts==1:
            atc = torch.Tensor([float(re.split('/', filename)[-1][7:11])])
            instance_idx = int(re.split('/', filename)[-1][:4])
            return ([pos_tensor, neg_tensor], atc, instance_idx)
        if num_atc_parts==2:
            atc1 = torch.Tensor([float(re.split('/', filename)[-1][7:11])])
            atc2 = torch.Tensor([float(re.split('/', filename)[-1][11:15])])
            instance_idx = int(re.split('/', filename)[-1][:4])
            return ([pos_tensor, neg_tensor], torch.Tensor([atc1, atc2]), instance_idx)
    else:
        return [pos_tensor, neg_tensor]

def read_sdf_samples_into_ram_bi(filename, normalize_atc, articulation=False, num_atc_parts=1):
    '''
    plyfile을 읽는다.
    '''
    assert articulation
    # npz = np.load(filename)
    # pos_tensor = torch.from_numpy(npz["pos"])
    # neg_tensor = torch.from_numpy(npz["neg"])
    # if articulation==True:
    #     if num_atc_parts==1:
    #         atc = torch.Tensor([float(re.split('/', filename)[-1][7:11])])
    #         instance_idx = int(re.split('/', filename)[-1][:4])
    #         return ([pos_tensor, neg_tensor], atc, instance_idx)
    #     if num_atc_parts==2:
    #         atc1 = torch.Tensor([float(re.split('/', filename)[-1][7:11])])
    #         atc2 = torch.Tensor([float(re.split('/', filename)[-1][11:15])])
    #         instance_idx = int(re.split('/', filename)[-1][:4])
    #         return ([pos_tensor, neg_tensor], torch.Tensor([atc1, atc2]), instance_idx)
    # else:
    #     return [pos_tensor, neg_tensor]
    vertex_data = PlyData.read(filename)['vertex']
    # 필요한 속성 추출
    x = vertex_data['x']
    y = vertex_data['y']
    z = vertex_data['z']
    sdf = vertex_data['sdf']
    label = vertex_data['label']
    
    obj_idx = filename.split('/')[-3]
    assert obj_idx.isdigit(), obj_idx
    obj_idx = int(obj_idx)
    # 라벨을 바꾸어줌
    new_label = np.full_like(label, -100)
    if obj_idx in to_switch_label:
        unique_label = np.unique(label)
        #1,2,3,4,....로 라벨링하기
        for ul in unique_label:
            to_change = to_switch_label[obj_idx][ul-1]+1
            if to_switch_label[obj_idx][ul-1] != -100:
                new_label[label == ul] = to_change
    else:
        # <= num_atc_parts+1인것들 저장
        for i in range(num_atc_parts+2):
            new_label[label == i] = i
    
    #체크 num_atc_parts+1보다 큰라벨은 없어야
    assert new_label.max() == num_atc_parts+1, f"num_atc_parts: {num_atc_parts}, new label: {new_label.max()}"


    xyz = np.vstack((x, y, z, sdf, new_label)).T
    
        
    assert xyz.shape[-1] == 5, xyz.shape
    
    pos = xyz[sdf >= 0]
    neg = xyz[sdf < 0]
    pos_tensor = remove_nans(torch.from_numpy(pos))
    neg_tensor = remove_nans(torch.from_numpy(neg))
    half = min(len(pos_tensor), len(neg_tensor))
    # split the sample into half

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    # samples = torch.cat([sample_pos, sample_neg], 0)
    
    instance_pose_path = '/'.join(filename.split('/')[:-1])
    
    # joint information도 추가
    with open(os.path.join(instance_pose_path, 'joint_cfg.json'), 'r') as f:
        joint_dict = json.load(f)
    
    
    #NOTE: 여기서 atc는 무조건 비정규화된 value
    atc = np.zeros((num_atc_parts))
    atc_limit = np.zeros((num_atc_parts, 2))
    
    for joint_info in joint_dict.values():
        # parent link와 child link 탐색하고, num_atc parts가=1이면 1,2 num_atc_parts=2이면 1,2,3만본다.
        p_idx = joint_info['parent_link']['index']
        c_idx = joint_info['child_link']['index']
        
        # 라벨 스위치
        if obj_idx in to_switch_label:
            to_p_change = to_switch_label[obj_idx][p_idx-1]+1
            if to_switch_label[obj_idx][p_idx-1] != -100:
                print("p_idx change to", p_idx, "to", to_p_change)
                p_idx = to_p_change
            else:
                p_idx = -100
            
            to_c_change = to_switch_label[obj_idx][c_idx-1]+1
            if to_switch_label[obj_idx][c_idx-1] != -100:
                print("c_idx change to", c_idx, "to", to_c_change)
                c_idx = to_c_change
            else:
                c_idx = -100
            
        assert num_atc_parts == 1 or num_atc_parts == 2, num_atc_parts
        # joint는 만약 double이면 라벨 2번과 연결되어 있는 것을 먼저 넣고, 3번이랑 되어 있는 것을 그 다음에 집어넣는다.

        if num_atc_parts == 1:
            if (p_idx == 1 and c_idx == 2) or (p_idx == 2 and c_idx == 1):
                qpos_range = joint_info['qpos_limit'][1] - joint_info['qpos_limit'][0]
                assert np.isfinite(qpos_range), "We only consider for this experiment"
                qpos = joint_info['qpos']
                # 정규화 된 값이 아닌 실제 값을 집어넣는다.
                if normalize_atc:
                    atc[0] = qpos - joint_info['qpos_limit'][0]
                    atc_limit[0][0] = joint_info['qpos_limit'][0]
                    atc_limit[0][1] = joint_info['qpos_limit'][1]
                else:
                    atc[0] = (qpos - joint_info['qpos_limit'][0])* 180 / np.pi 
                    atc_limit[0][0] = joint_info['qpos_limit'][0] * 180 / np.pi
                    atc_limit[0][1] = joint_info['qpos_limit'][1] * 180 / np.pi
           
        else:
            # 베이스(1)에 두가지가 연결된 상태
            if (p_idx == 1 and c_idx == 2) or (p_idx == 2 and c_idx == 1):
                qpos_range = joint_info['qpos_limit'][1] - joint_info['qpos_limit'][0]
                assert np.isfinite(qpos_range), "We only consider for this experiment"
                qpos = joint_info['qpos']
                if normalize_atc:
                    atc[0] = qpos - joint_info['qpos_limit'][0]
                    atc_limit[0][0] = joint_info['qpos_limit'][0]
                    atc_limit[0][1] = joint_info['qpos_limit'][1]
                else:
                    atc[0] = (qpos - joint_info['qpos_limit'][0])* 180 / np.pi 
                    atc_limit[0][0] = joint_info['qpos_limit'][0] * 180 / np.pi
                    atc_limit[0][1] = joint_info['qpos_limit'][1] * 180 / np.pi
            
            elif (p_idx == 1 and c_idx == 3) or (p_idx == 3 and c_idx == 1) or (p_idx == 2 and c_idx == 3) or (p_idx == 3 and c_idx == 2):
                qpos_range = joint_info['qpos_limit'][1] - joint_info['qpos_limit'][0]
                assert np.isfinite(qpos_range), "We only consider for this experiment"
                qpos = joint_info['qpos']
                if normalize_atc:
                    atc[1] = qpos - joint_info['qpos_limit'][0]
                    atc_limit[1][0] = joint_info['qpos_limit'][0]
                    atc_limit[1][1] = joint_info['qpos_limit'][1]
                else:
                    atc[1] = (qpos - joint_info['qpos_limit'][0])* 180 / np.pi 
                    atc_limit[1][0] = joint_info['qpos_limit'][0] * 180 / np.pi
                    atc_limit[1][1] = joint_info['qpos_limit'][1] * 180 / np.pi
    
        
    print("atc", atc)
    print("atc limit", atc_limit)
    assert np.all(atc != 0), atc 
    assert articulation
    
    
    # test 할때는 object index가 필요 없기 때문에 그것 대신 qpos_limit을 집어넣는다.
    
    return [sample_pos, sample_neg], torch.Tensor(atc), torch.Tensor(atc_limit)

def read_sdf_samples_into_ram_rbo(filename, articulation=False, num_atc_parts=1):
    npz = np.load(filename)
    pos_tensor = torch.from_numpy(npz["pos"])
    neg_tensor = torch.from_numpy(npz["neg"])
    if articulation==True:
        if num_atc_parts==1:
            atc = torch.Tensor([float(re.split('/', filename)[-1][-8:-4])])
            instance_idx = int(re.split('/', filename)[-1][:4])
            return ([pos_tensor, neg_tensor], atc, instance_idx)
    else:
        return [pos_tensor, neg_tensor]

def unpack_sdf_samples_bi(filename, normalize_atc, subsample=None, articulation=False, num_atc_parts=1):
    vertex_data = PlyData.read(filename)['vertex']
    # 필요한 속성 추출
    x = vertex_data['x']
    y = vertex_data['y']
    z = vertex_data['z']
    sdf = vertex_data['sdf']
    label = vertex_data['label']
    
    obj_idx = filename.split('/')[-3]
    assert obj_idx.isdigit(), obj_idx
    obj_idx = int(obj_idx)
    # 라벨을 바꾸어줌
    new_label = np.full_like(label, -100)
    
    if obj_idx in to_switch_label:
        unique_label = np.unique(label)
        #1,2,3,4,....로 라벨링하기
        for ul in unique_label:
            to_change = to_switch_label[obj_idx][ul-1]+1
            if to_switch_label[obj_idx][ul-1] != -100:
                new_label[label == ul] = to_change
    else:
        # <= num_atc_parts+1인것들 저장
        for i in range(num_atc_parts+2):
            new_label[label == i] = i
    
    #체크 num_atc_parts+1보다 큰라벨은 없어야
    assert new_label.max() == num_atc_parts+1, f"num_atc_parts: {num_atc_parts}, new label: {new_label.max()}"


    xyz = np.vstack((x, y, z, sdf, new_label)).T
    assert xyz.shape[-1] == 5, xyz.shape
    
    pos = xyz[sdf >= 0]
    neg = xyz[sdf < 0]
    assert subsample is not None
    if subsample is None:
        return xyz
    pos_tensor = remove_nans(torch.from_numpy(pos))
    neg_tensor = remove_nans(torch.from_numpy(neg))
    
    # split the sample into half
    half = int(subsample / 2)

    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()

    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)
    
    instance_pose_path = '/'.join(filename.split('/')[:-1])
    
    # joint information도 추가
    with open(os.path.join(instance_pose_path, 'joint_cfg.json'), 'r') as f:
        joint_dict = json.load(f)
    
    
    atc = np.zeros((num_atc_parts))
    for joint_info in joint_dict.values():
        # parent link와 child link 탐색하고, num_atc parts가=1이면 1,2 num_atc_parts=2이면 1,2,3만본다.
        p_idx = joint_info['parent_link']['index']
        c_idx = joint_info['child_link']['index']
        
        # 라벨 스위치
        if obj_idx in to_switch_label:
            to_p_change = to_switch_label[obj_idx][p_idx-1]+1
            if to_switch_label[obj_idx][p_idx-1] != -100:
                p_idx = to_p_change
            else:
                p_idx = -100
            
            to_c_change = to_switch_label[obj_idx][c_idx-1]+1
            if to_switch_label[obj_idx][c_idx-1] != -100:
                c_idx = to_c_change
            else:
                c_idx = -100
            
        assert num_atc_parts == 1 or num_atc_parts == 2, num_atc_parts
        # joint는 만약 double이면 라벨 2번과 연결되어 있는 것을 먼저 넣고, 3번이랑 되어 있는 것을 그 다음에 집어넣는다.
        if num_atc_parts == 1:
            if (p_idx == 1 and c_idx == 2) or (p_idx == 2 and c_idx == 1):
                qpos_range = joint_info['qpos_limit'][1] - joint_info['qpos_limit'][0]
                assert np.isfinite(qpos_range), "We only consider for this experiment"
                qpos = joint_info['qpos']
                if normalize_atc:
                    qpos_normalized = (joint_info['qpos'] - joint_info['qpos_limit'][0]) / qpos_range
                    atc[0] = qpos_normalized
                
                else:
                    #형식에 맞게 degree로 바꾸어줌
                    #prismatic도 degree로 치환해서 하더라 from CARTO
                    #minimum 0로 하기
                    # atc[0] = qpos - joint_info['qpos_limit'][0]
                    atc[0] = (qpos - joint_info['qpos_limit'][0]) * 180 / np.pi
        else:
            # 베이스(1)에 두가지가 연결된 상태
            if (p_idx == 1 and c_idx == 2) or (p_idx == 2 and c_idx == 1):
                qpos_range = joint_info['qpos_limit'][1] - joint_info['qpos_limit'][0]
                assert np.isfinite(qpos_range), "We only consider for this experiment"
                qpos = joint_info['qpos']
                if normalize_atc:
                    qpos_normalized = (joint_info['qpos'] - joint_info['qpos_limit'][0]) / qpos_range
                    atc[0] = qpos_normalized
                
                else:
                    #형식에 맞게 degree로 바꾸어줌
                    atc[0] = (qpos - joint_info['qpos_limit'][0]) * 180 / np.pi
                    # atc[0] = qpos - joint_info['qpos_limit'][0]
            elif (p_idx == 1 and c_idx == 3) or (p_idx == 3 and c_idx == 1) or (p_idx == 2 and c_idx == 3) or (p_idx == 3 and c_idx == 2):
                qpos_range = joint_info['qpos_limit'][1] - joint_info['qpos_limit'][0]
                assert np.isfinite(qpos_range), "We only consider for this experiment"
                qpos = joint_info['qpos']
                if normalize_atc:
                    qpos_normalized = (joint_info['qpos'] - joint_info['qpos_limit'][0]) / qpos_range
                    atc[1] = qpos_normalized
            
                else:
                    #형식에 맞게 degree로 바꾸어줌
                    atc[1] = (qpos - joint_info['qpos_limit'][0]) * 180 / np.pi
                    # else:
                    #     atc[1] = qpos - joint_info['qpos_limit'][0]
    assert np.all(atc != 0), atc 
            
    assert articulation
    
    # if articulation==True:
    #     if num_atc_parts==1:
    #         atc = torch.Tensor([float(re.split('/', filename)[-1][7:11])])
    #         instance_idx = int(re.split('/', filename)[-1][:4])
    #         return (samples, atc, instance_idx)
    #     if num_atc_parts==2:
    #         atc1 = torch.Tensor([float(re.split('/', filename)[-1][7:11])])
    #         atc2 = torch.Tensor([float(re.split('/', filename)[-1][11:15])])
    #         instance_idx = int(re.split('/', filename)[-1][:4])
    #         return (samples, torch.Tensor([atc1, atc2]), instance_idx)
    # else:
    #     return samples
    return samples, torch.Tensor(atc)
       
    

def unpack_sdf_samples(filename, subsample=None, articulation=False, num_atc_parts=1):
    npz = np.load(filename) 
    
    # subsmaple = 16000
    if subsample is None:
        return npz

    
    pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
    neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))
    
    # split the sample into half
    half = int(subsample / 2)

    # half개 만큼 pos_tensor와 neg_tensor를 무작위로 선택
    random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
    random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
    
    sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    sample_neg = torch.index_select(neg_tensor, 0, random_neg)

    samples = torch.cat([sample_pos, sample_neg], 0)

    if articulation==True:
        if num_atc_parts==1:
            atc = torch.Tensor([float(re.split('/', filename)[-1][7:11])])
            instance_idx = int(re.split('/', filename)[-1][:4])
            return (samples, atc, instance_idx)
        if num_atc_parts==2:
            atc1 = torch.Tensor([float(re.split('/', filename)[-1][7:11])])
            atc2 = torch.Tensor([float(re.split('/', filename)[-1][11:15])])
            instance_idx = int(re.split('/', filename)[-1][:4])
            return (samples, torch.Tensor([atc1, atc2]), instance_idx)
    else:
        return samples


def unpack_sdf_samples_from_ram(data, subsample=None, articulation=False, num_atc_parts=1):
    if subsample is None:
        return data
    if articulation==True:
        pos_tensor = data[0][0]
        neg_tensor = data[0][1]
        atc = data[1]
        instance_idx = data[2]
    else:
        pos_tensor = data[0]
        neg_tensor = data[1]        

    # split the sample into half
    half = int(subsample / 2)

    pos_size = pos_tensor.shape[0]
    neg_size = neg_tensor.shape[0]

    #pos_start_ind = random.randint(0, pos_size - half)
    #sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if pos_size <= half:
        random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
        sample_pos = torch.index_select(pos_tensor, 0, random_pos)
    else:
        pos_start_ind = random.randint(0, pos_size - half)
        sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

    if neg_size <= half:
        random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
        sample_neg = torch.index_select(neg_tensor, 0, random_neg)
    else:
        neg_start_ind = random.randint(0, neg_size - half)
        sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

    samples = torch.cat([sample_pos, sample_neg], 0)

    if articulation==True:
        # print("go", samples, "atc", atc)
        # print("instance idx", instance_idx)
        return (samples, atc, instance_idx)
    else:
        return samples

class SDFSamples(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source,
        split,
        subsample,
        load_ram=False,
        print_filename=False,
        num_files=1000000,
        articulation=False,
        num_atc_parts=1,
    ):
        self.subsample = subsample

        self.data_source = data_source
        self.npyfiles = get_instance_filenames(data_source, split)
        self.articualtion = articulation
        self.num_atc_parts = num_atc_parts

        logging.debug(
            "using "
            + str(len(self.npyfiles))
            + " shapes from data source "
            + data_source
        )

        self.load_ram = load_ram

        if load_ram:
            self.loaded_data = []
            for f in self.npyfiles:
                filename = os.path.join(self.data_source, ws.sdf_samples_subdir, f)
                npz = np.load(filename)
                pos_tensor = remove_nans(torch.from_numpy(npz["pos"]))
                neg_tensor = remove_nans(torch.from_numpy(npz["neg"]))

                if self.articualtion==True:
                    if self.num_atc_parts==1:
                        atc = torch.Tensor([float(re.split('/', filename)[-1][7:11])])
                        
                        instance_idx = int(re.split('/', filename)[-1][:4])
                        self.loaded_data.append(
                            (
                            [
                                pos_tensor[torch.randperm(pos_tensor.shape[0])],
                                neg_tensor[torch.randperm(neg_tensor.shape[0])],
                            ],
                            atc,
                            instance_idx,
                            )
                        )
                    if self.num_atc_parts==2:
                        atc1 = torch.Tensor([float(re.split('/', filename)[-1][7:11])])
                        atc2 = torch.Tensor([float(re.split('/', filename)[-1][11:15])])
                        instance_idx = int(re.split('/', filename)[-1][:4])
                        self.loaded_data.append(
                            (
                            [
                                pos_tensor[torch.randperm(pos_tensor.shape[0])],
                                neg_tensor[torch.randperm(neg_tensor.shape[0])],
                            ],
                            [atc1, atc2],
                            instance_idx,
                            )
                        )

                else:
                    self.loaded_data.append(
                        [
                            pos_tensor[torch.randperm(pos_tensor.shape[0])],
                            neg_tensor[torch.randperm(neg_tensor.shape[0])],
                        ],
                    )

    def __len__(self):
        return len(self.npyfiles)

    def __getitem__(self, idx):
        filename = os.path.join(
            self.data_source, ws.sdf_samples_subdir, self.npyfiles[idx]
        )
        if self.load_ram:
            return (
                unpack_sdf_samples_from_ram(self.loaded_data[idx], self.subsample, self.articualtion, self.num_atc_parts),
                idx,
            )
        else:
            return unpack_sdf_samples(filename, self.subsample, self.articualtion, self.num_atc_parts), idx
            # return unpack_sdf_samples(filename, self.num_atc_parts, self.subsample, self.articualtion, self.num_atc_parts), idx



'''
Partnet mobility datset으로 a-sdf학습시킬때
'''
class SDFSamplesBI(torch.utils.data.Dataset):
    def __init__(
        self,
        data_source, 
        pkl_path, #pickle
        category,
        split,
        subsample,
        normalize_atc,
        load_ram=False,
        print_filename=False,
        num_files=1000000,
        articulation=False,
        num_atc_parts=1,
    ):
        self.pkl_path = pkl_path
        self.subsample = subsample
        self.split = split
        self.data_source = data_source
        # self.npyfiles = get_instance_filenames(data_source, split)
        self.category = category
        self.obj_id2lat_vec = dict() #obj_idx 별 lat_vec에 사용되는 인덱스 저장
        
        self.files = self._load_data()
        self.articualtion = articulation
        self.num_atc_parts = num_atc_parts
        self.normalize_atc = normalize_atc

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        obj_idx = filename.split('/')[-3]
        assert obj_idx.isdigit(), obj_idx
        obj_idx = int(obj_idx)        
        samples, atc = unpack_sdf_samples_bi(filename, normalize_atc=self.normalize_atc, subsample=self.subsample, articulation=self.articualtion, num_atc_parts=self.num_atc_parts)
        return (samples, atc, self.obj_id2lat_vec[obj_idx])
    def _load_data(self):
        total_valid_paths = []
        # dir = self.data_source
        cnt = 0
        data_dict = np.load(self.pkl_path, allow_pickle=True)
        for cat in data_dict.keys():
            if cat != self.category: continue
            for spt in data_dict[cat].keys():
                if self.split == 'trn':
                    if spt == 'test': continue
                else:
                    assert self.split == 'test'
                    if spt == 'train' or spt == 'val': continue
                instances = data_dict[cat][spt]
                for instance in instances:
                    self.obj_id2lat_vec[instance] = cnt
                    cnt += 1
                    for i in range(100):
                        total_valid_paths.append(os.path.join(self.data_source, spt, cat, str(instance), f"pose_{i}","points_with_sdf_label_binary.ply"))
        return total_valid_paths    