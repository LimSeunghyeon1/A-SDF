import os
import subprocess
import multiprocessing

# 각 gpu에서 프로세스 실행하기
modes = ["one_revolute"]

categories = {
    # "double_revolute": ['Eyeglasses', "Refrigerator", "Stapler", "StorageFurniture", "TrashCan"],
    # "double_revolute": ['Eyeglasses', "Refrigerator",  "StorageFurniture", "TrashCan"],
    # "one_prismatic": ["StorageFurniture", "Table", "Toaster"],
    "one_revolute": ["Box"]
}

# 실행할 함수
def run_process(mode, category, seed):
    specs_path = os.path.join(f'experiments_real_{seed}', mode, category)
    
    # CUDA_VISIBLE_DEVICES 환경 변수 설정하여 GPU 지정
    command = f'python test_bi_real.py -e {specs_path} -c 1000 -m recon_testset && python test_bi_real.py -e {specs_path} -c 1000 -m recon_testset_ttt'
    print(f"STARTING: {command}")
    
    # Error handling for subprocess
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        print(result.stdout)  # Output of the command
    except subprocess.CalledProcessError as e:
        print(f"Error running command {command}: {e.stderr}")

# 멀티 프로세싱 실행 함수
def run_all_processes():
    cnt = 0
    processes = []
    seeds = [256, 355]
    for seed in seeds:
        for mode in modes:
            for category in categories[mode]:
                # 멀티 프로세싱 사용하여 각 프로세스를 병렬로 실행
                # p = multiprocessing.Process(target=run_process, args=(mode, category, cnt))
                # p.start()
                # processes.append(p)
                # cnt = (cnt + 1) % 8  # Rotate through GPUs
                run_process(mode, category, seed)
    

if __name__ == "__main__":
    run_all_processes()
