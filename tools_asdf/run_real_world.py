import os
import subprocess
import multiprocessing

# 각 gpu에서 프로세스 실행하기
modes = ["one_revolute"]

categories = {
    "one_revolute": ["Box"]
}

# 실행할 함수
def run_process(mode, category, seed):
    specs_path = os.path.join(f'experiments_real_{seed}', mode, category)
    
    # CUDA_VISIBLE_DEVICES 환경 변수 설정하여 GPU 지정
    command = f'python -m torch.distributed.run --nproc_per_node=1 train_bi_ddp_real.py -e {specs_path} --seed {seed}'
    print("STARTING", command)
    subprocess.run(command, shell=True, capture_output=True, text=True)

# 멀티 프로세싱 실행 함수
def run_all_processes():
    seeds = [256, 355]
    for seed in seeds:
        for mode in modes:
            for category in categories[mode]:            
                run_process(mode, category, seed)



if __name__ == "__main__":
    run_all_processes()
