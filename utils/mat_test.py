import h5py

file_path = 'D:/桌面/深度学习/data/IXI/data_train_T1.h5'
try:
    with h5py.File(file_path, 'r') as f:
        print("File opened successfully.")
        print(f.keys())  # 列出文件中的数据集或变量
except Exception as e:
    print(f"Failed to open file {file_path}: {e}")