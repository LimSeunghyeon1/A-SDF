import os
import subprocess
import multiprocessing

# 각 gpu에서 프로세스 실행하기
modes = ["double_revolute", "one_revolute", "one_prismatic"]

categories = {
    "double_revolute": ['Eyeglasses', "Refrigerator", "Stapler", "StorageFurniture", "TrashCan"],
    # "double_revolute": ['Eyeglasses', "Refrigerator",  "StorageFurniture", "TrashCan"],
    "one_prismatic": ["StorageFurniture", "Table", "Toaster"],
    "one_revolute": ["Box", "Dishwasher", "Door", "Laptop", "Microwave", "Oven", "Pliers", "Refrigerator", "Scissors", "StorageFurniture", "TrashCan"]
}

# 실행할 함수
def run_process(mode, category, gpu_id):
    specs_path = os.path.join('experiments', mode, category)
    
    # CUDA_VISIBLE_DEVICES 환경 변수 설정하여 GPU 지정
    command = f'CUDA_VISIBLE_DEVICES={gpu_id} python test_bi.py -e {specs_path} -c 1000 -m recon_testset && CUDA_VISIBLE_DEVICES={gpu_id} python test_bi.py -e {specs_path} -c 1000 -m recon_testset_ttt'
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
    for mode in modes:
        for category in categories[mode]:
            # 멀티 프로세싱 사용하여 각 프로세스를 병렬로 실행
            p = multiprocessing.Process(target=run_process, args=(mode, category, cnt))
            p.start()
            processes.append(p)
            cnt = (cnt + 1) % 8  # Rotate through GPUs
    
    # 모든 프로세스가 끝날 때까지 대기
    for p in processes:
        p.join()

if __name__ == "__main__":
    run_all_processes()
