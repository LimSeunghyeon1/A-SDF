import os
import subprocess
import multiprocessing

# 각 gpu에서 프로세스 실행하기
modes = ["double_revolute", "one_revolute", "one_prismatic"]

categories = {
    "double_revolute": ['Eyeglasses', "Refrigerator", "Stapler", "StorageFurniture", "TrashCan"],
    "one_prismatic": ["StorageFurniture", "Table", "Toaster"],
    "one_revolute": ["Box", "Dishwasher", "Door", "Laptop", "Microwave", "Oven", "Pliers", "Refrigerator", "Scissors", "StorageFurniture", "TrashCan"]
}

# 실행할 함수
def run_process(mode, category):
    specs_path = os.path.join('experiments', mode, category)
    
    # CUDA_VISIBLE_DEVICES 환경 변수 설정하여 GPU 지정
    command = f'python -m torch.distributed.run --nproc_per_node=8 train_bi_ddp.py -e {specs_path}'
    print("STARTING", command)
    subprocess.run(command, shell=True, capture_output=True, text=True)

# 멀티 프로세싱 실행 함수
def run_all_processes():

    for mode in modes:
        for category in categories[mode]:            
            run_process(mode, category)



if __name__ == "__main__":
    run_all_processes()
