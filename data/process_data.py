import os
import shutil


DATA_DIR = 'data'
RAW_DATA_DIR = 'data/Solomon_converted'

def check(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().strip().split('\n')

    n = int(lines[1])

    has_zero_in_d = False
    for i in range(2, n + 2):
        parts = lines[i].split()
        
        if float(parts[2]) == 0.0:
            has_zero_in_d = True
            break

    if not has_zero_in_d:
        return True

    for i, line in enumerate(lines[n + 2:]):
        for j, x in enumerate(line.split()):
            if i == j:
                continue

            if not float(x):
                return False
    
    return True

def process():
    if not os.path.exists(DATA_DIR):
        print(f"LỖI: Không tìm thấy thư mục '{DATA_DIR}")
        return

    if not os.path.exists(RAW_DATA_DIR):
        print(f"LỖI: Không tìm thấy thư mục '{RAW_DATA_DIR}")
        return 

    raw_data_files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.txt')])
    file_count = len([f for f in os.listdir(DATA_DIR) if f.endswith('.txt')])

    for file in raw_data_files:
        file = os.path.join(RAW_DATA_DIR, file)

        if check(file):
            file_count += 1
            file_name = f'testcase{file_count}.txt'
            file_path = os.path.join(DATA_DIR, file_name)

            shutil.copy2(file, file_path)

            print(f'Đã copy file {file}')

        else:
            print(f'Giá trị trong file {file} không tốt')

        
if __name__ == "__main__":
    process()
