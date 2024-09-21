import os
import numpy as np
import sys
sys.path.append('.')
from tools_asdf.switch_label import to_switch_label 
from collections import defaultdict
import json
import math
import pandas as pd

pth = '/home/shlim/data/projects/Articulation_project/A-SDF/experiments/'
x_sum = 0
cnt = 0
#default dict로써 
'''
revised version: 형평성을 위해, json 파일을 참조해 만약 atc가 range 바깥에 있다고 한다면, 그 값을 바깥쪽 끝으로 바꾼다.
확인: json파일로 발견한 ground truth값과 atc_vec을 빼서 atc_err가 나오는지 확인 

(mode_category_{exp_type}.csv)
최종적으로 csv파일로 pred_atc_vec (pred_atc_0, pred_atc_1, pred_atc_2), truncated_pred_atc_vec (...), gt_atc_vec (...), 
-> csv 파일은 총 4개 (one_prismatic, double_prismatic, one_revolute, double_revolute)
mean err, truncated_mean_err   
'''
'''
{mode:{category:{{instnce_num:, pred_atc_0:, pred_atc_1:, pred_atc_2:, truncated_pred_atc_0, ...., \
    gt_atc_0, gt ...., }}}}
'''
right = 0
wrong = 0
err_dict = {'ttt': {'one_prismatic': defaultdict(lambda: []), 'double_prismatic': defaultdict(lambda: []),\
    'one_revolute': defaultdict(lambda: []), 'double_revolute': defaultdict(lambda: [])}, 'no_ttt': {'one_prismatic': defaultdict(lambda: []), 'double_prismatic': defaultdict(lambda: []),\
    'one_revolute': defaultdict(lambda: []), 'double_revolute': defaultdict(lambda: [])}}
mode_result_dict = {'ttt': {'one_prismatic': defaultdict(lambda:defaultdict(lambda:{})), 'double_prismatic': defaultdict(lambda:defaultdict(lambda:{})),\
    'one_revolute': defaultdict(lambda:defaultdict(lambda:{})), 'double_revolute': defaultdict(lambda:defaultdict(lambda:{}))}, 'no_ttt': {'one_prismatic': defaultdict(lambda:defaultdict(lambda:{})), 'double_prismatic': defaultdict(lambda:defaultdict(lambda:{})),\
    'one_revolute': defaultdict(lambda:defaultdict(lambda:{})), 'double_revolute': defaultdict(lambda:defaultdict(lambda:{}))}}
for dirpath, dirname, filenames in os.walk(pth):
    for filename in filenames:
        if '.npy' in filename:
            if 'atc_err.npy' in filename:
                continue
            try:
                instance_num, _, pose_num = filename[:-4].split('_')
            except ValueError:
                continue
            assert pose_num.isdigit()
            index_name = filename[:-4]
                
            mode, category, exp_type, _a, _b, _c = dirpath.split('/')[-6:]
            if _a == '1000' and _b == 'Codes' and _c == 'partnet_mobility':
                if 'testset_ttt' in exp_type:
                    exp_type = 'ttt'
                else:
                    exp_type = 'no_ttt'

                atc_vec = np.load(os.path.join(dirpath, filename))
                err_vec = np.load(os.path.join(dirpath, filename[:-4] + '_atc_err.npy'))
                assert len(atc_vec) == 1,f"dirpath{dirpath}, filename{filename}, atc {atc_vec}"
                atc_vec = atc_vec[0]
                assert instance_num.isdigit(), instance_num
                instance_num = int(instance_num)
                if len(atc_vec) == 1:
                    assert 'one_' in dirpath, f"dirpath{dirpath}, filename{filename}, atc {atc_vec}"
                    
                elif len(atc_vec) == 2:
                    assert 'double_' in dirpath, dirpath
                    
                else:
                    raise NotImplementedError
                with open(os.path.join('../../pose_data/', 'test', category, str(instance_num), f'pose_{pose_num}', 'joint_cfg.json'), 'rb') as f:
                    json_dict = json.load(f)
                
                for jd in json_dict.values():
                    p_idx = jd['parent_link']['index'] - 1
                    c_idx = jd['child_link']['index'] - 1
                    switched=False
                    if instance_num in to_switch_label.keys():
                        # print("original index", p_idx, c_idx)
                        p_idx = to_switch_label[instance_num][p_idx]
                        c_idx = to_switch_label[instance_num][c_idx]
                        # print("--to swtiched index", p_idx, c_idx)
                        if not(p_idx >=0 and p_idx <= len(atc_vec) and c_idx >=0 and c_idx <= len(atc_vec)):
                            continue
                    atc_idx = 100 #dummy
                    if len(atc_vec) == 1:
                        if (p_idx == 0 and c_idx) == 1 or (p_idx == 1 and c_idx == 0):
                            atc_idx = 0
                    else:
                        if (p_idx == 0 and c_idx) == 1 or (p_idx == 1 and c_idx == 0):
                            atc_idx = 0
                        if (p_idx == 0 and c_idx == 2) or (p_idx == 2 and c_idx == 0) or (p_idx == 1 and c_idx == 2) or (p_idx == 2 and c_idx == 1):
                            atc_idx = 1
                        
                    if atc_idx != 100:
                        min_qpos, max_qpos = jd['qpos_limit']
                        if jd['type'] == 'revolute_unwrapped':
                            min_qpos = min_qpos * 180 / math.pi
                            max_qpos = max_qpos * 180 / math.pi
                            qpos = jd['qpos'] * 180 / math.pi
                        else:
                            qpos = jd['qpos']
                        
                        ## min_qpos를 0으로 놓고 compute했음
                        max_qpos -= min_qpos
                        qpos -= min_qpos
                        min_qpos = 0
                        
                            
                        if atc_vec[atc_idx] < min_qpos:
                            mode_result_dict[exp_type][mode][category][index_name][f'truncated_pred_atc_{atc_idx}'] = min_qpos
                        elif atc_vec[atc_idx] > max_qpos:
                            mode_result_dict[exp_type][mode][category][index_name][f'truncated_pred_atc_{atc_idx}'] = max_qpos
                        else:
                            mode_result_dict[exp_type][mode][category][index_name][f'truncated_pred_atc_{atc_idx}'] = atc_vec[atc_idx]

                        
                        mode_result_dict[exp_type][mode][category][index_name][f'gt_atc_{atc_idx}'] = qpos
                        
                        
                            
                err_check = 0
                for i, atc in enumerate(atc_vec):
                    mode_result_dict[exp_type][mode][category][index_name][f'pred_atc_{i}'] = atc_vec[i]
                    assert f'truncated_pred_atc_{i}' in mode_result_dict[exp_type][mode][category][index_name].keys(), f"I {i}, exp_type {exp_type}, mode, {mode}, category, {category}, index_name, {index_name}: {mode_result_dict[mode][category][index_name]}"
                    err_check += abs(mode_result_dict[exp_type][mode][category][index_name][f'gt_atc_{i}'] - atc_vec[i])
                # print(mode_result_dict[mode][category][instance_num], "mode", mode, "category", category, "instance num", instance_num)
                if not abs(err_vec - err_check / len(atc_vec)) < 1e-3:
                    # print(f"error reported{err_vec}, but we found{err_check / len(atc_vec)}, exp name: {exp_type}, mode:{mode},category:{category}, index_name{index_name}, posenum{pose_num}\
                    # {mode_result_dict[exp_type][mode][category][index_name]}")
                    wrong += 1
                    err_dict[exp_type][mode][category].append(index_name)
                else:
                    right+=1

# for exp_type in ['ttt', 'no_ttt']: 
#     for mode in err_dict[exp_type].keys():
#         for category in err_dict[exp_type][mode]:
#             # print("before", err_dict[mode][category])
#             # exit(0)
#             err_dict[exp_type][mode][category] = sorted(err_dict[exp_type][mode][category], key=lambda x: (int(x.split('_')[0]), int(x[:-4].split('_')[-1])) )
        
            # assert len(err_dict[exp_type][mode][category]) % 100 == 0, f"exp_type:{exp_type}, mode:{mode}, category:{category}, {len(err_dict[exp_type][mode][category])}, {err_dict[exp_type][mode][category]}" 
# print(err_dict['ttt']['one_revolute']['StorageFurniture'], len(err_dict['ttt']['one_revolute']['StorageFurniture']))

# 재귀적으로 defaultdict를 dict로 변환하는 함수
def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, dict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d


err_avg_dict = {'err_avg': {'ttt': {'one_prismatic': defaultdict(lambda:defaultdict(lambda:0)), 'double_prismatic': defaultdict(lambda:defaultdict(lambda:0)),\
    'one_revolute': defaultdict(lambda:defaultdict(lambda:0)), 'double_revolute': defaultdict(lambda:defaultdict(lambda:0))}, 'no_ttt': {'one_prismatic': defaultdict(lambda:defaultdict(lambda:0)), 'double_prismatic': defaultdict(lambda:defaultdict(lambda:0)),\
    'one_revolute': defaultdict(lambda:defaultdict(lambda:0)), 'double_revolute': defaultdict(lambda:defaultdict(lambda:0))}}, 
    
    'truncated_err_avg':{'ttt': {'one_prismatic': defaultdict(lambda:defaultdict(lambda:0)), 'double_prismatic': defaultdict(lambda:defaultdict(lambda:0)),\
    'one_revolute': defaultdict(lambda:defaultdict(lambda:0)), 'double_revolute': defaultdict(lambda:defaultdict(lambda:0))}, 'no_ttt': {'one_prismatic': defaultdict(lambda:defaultdict(lambda:0)), 'double_prismatic': defaultdict(lambda:defaultdict(lambda:0)),\
    'one_revolute': defaultdict(lambda:defaultdict(lambda:0)), 'double_revolute': defaultdict(lambda:defaultdict(lambda:0))}}}

base_dir = '/'.join(os.path.abspath(__file__).split('/')[:-1])
for exp in mode_result_dict.keys():
    for mode in mode_result_dict[exp].keys():
        for category in mode_result_dict[exp][mode].keys():
            instance_list = []
            for index_name in mode_result_dict[exp][mode][category].keys():
                mode_result_dict[exp][mode][category][index_name]["index_name"] = index_name
                instance_list.append(mode_result_dict[exp][mode][category][index_name])
            
            csv_path = os.path.join(base_dir,'csv_folder', mode, category, 'data.csv')
            os.makedirs(os.path.join(base_dir,'csv_folder', mode, category), exist_ok=True)
            data = pd.DataFrame(instance_list)
            data.to_csv(csv_path, index=False)
            err_sum = 0
            cnt = 0
            truncated_err_sum = 0
            for v in instance_list:
                if 'one' in mode: 
                    err_sum += abs(v['pred_atc_0'] - v['gt_atc_0'])
                    truncated_err_sum += abs(v['truncated_pred_atc_0'] - v['gt_atc_0'])
                    cnt += 1
                elif 'double' in mode:
                    err_sum += (abs(v['pred_atc_0'] - v['gt_atc_0']) + abs(v['pred_atc_1'] - v['gt_atc_1'])) / 2
                    truncated_err_sum += (abs(v['truncated_pred_atc_0']-v['gt_atc_0']) + abs(v['truncated_pred_atc_1']-v['gt_atc_1'])) / 2
                    cnt += 1
            
            err_avg = err_sum / cnt
            truncated_err_avg = truncated_err_sum / cnt
            if exp == 'ttt':
                exp_str = 'Results_recon_testset_ttt'
            else:
                exp_str = 'Results_recon_testset'
            err_avg_path = os.path.join('experiments', mode, category, exp_str, '1000', "Codes","partnet_mobility","final_atc_err.npy")
            err_avg_check = np.load(err_avg_path)
            assert abs(err_avg_check-err_avg) < 1e-3, err_avg_path
            err_avg_dict['err_avg'][exp][mode][category] = err_avg
            err_avg_dict['truncated_err_avg'][exp][mode][category] = truncated_err_avg


err_avg_dict = defaultdict_to_dict(err_avg_dict)
with open(os.path.join(base_dir,'csv_folder', 'err.json'), 'w') as f:
    json.dump(err_avg_dict, f, indent=4)
        
        

print("==========================================")

'''
csv 파일을 만들자
'''
#csv_folder = 
# for mode in mode_result_dict.keys():
#     if 'one_' in mode:
#         atc = 1
#     elif 'double_' in mode:
#         atc = 2
#     for category in mode_result_dict[mode].keys():
#         csv_path = f'test/table1_results/{mode}_{category}.csv'
#         df = pd.DataFrame(mode_result_dict[mode][category])
#         df.to_csv(csv_path, index=False)
#         print("save", csv_path)
        
#         '''
#         mode, category별 평균 atc_err 
#         ''' 
#         pred_atc_err = []
#         truncated_pred_atc_err = []
        
#         for v in mode_result_dict[mode][category]:
#             for a in range(atc):
#                 truncated_pred_atc_err.append(abs(mode_result_dict[mode][category][f"truncated_pred_atc_{a}"] - mode_result_dict[mode][category][f"gt_atc_{a}"]))
#                 pred_atc_err.append()       
    


'''
코어 정보를 산출해서 txt파일에 저장
mode
    category
        instance num: 평균 atc_err
        ...
        평균 atc_err (이때 반드시 예전에 만들었던 final_atc_err와 비교)
    평균 atc_err
            
         
ground truth 저장
'''



