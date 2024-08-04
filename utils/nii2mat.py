import os
import nibabel as nib
import numpy as np
import scipy.io


def load_nii_files(directory, file_names):
    data_list = []

    for file in file_names:
        file_path = os.path.join(directory, file)
        img = nib.load(file_path)
        img_data = img.get_fdata()
        img_data_reshaped = np.transpose(img_data, (2, 0, 1))  # 转换形状为 (#image, width, height)

        # 归一化到0-1范围内
        img_data_reshaped = (img_data_reshaped - np.min(img_data_reshaped)) / (
                    np.max(img_data_reshaped) - np.min(img_data_reshaped))

        data_list.append(img_data_reshaped)

    return np.concatenate(data_list, axis=0)


def save_mat_file(data, output_path):
    scipy.io.savemat(output_path, {'data': data})


def process_data(input_directory, file_names, output_path):
    data = load_nii_files(input_directory, file_names)
    save_mat_file(data, output_path)
    print(f"Saved {output_path} with shape {data.shape}")


# 文件路径设置
base_dir = "D:/桌面/深度学习/data/IXI"
ixi_t1_dir = os.path.join(base_dir, "IXI-T1")
ixi_t2_dir = os.path.join(base_dir, "IXI-T2")

# 获取文件名
t1_files = sorted([f for f in os.listdir(ixi_t1_dir) if f.endswith('.nii.gz')])
t2_files = sorted([f for f in os.listdir(ixi_t2_dir) if f.endswith('.nii.gz')])

# 提取共有文件名
common_files = set(f.replace('-T1', '') for f in t1_files).intersection(f.replace('-T2', '') for f in t2_files)


# 按范围选择文件
def get_file_names(files, common_files, start, end, suffix):
    return [f"{name}{suffix}.nii.gz" for name in sorted(common_files)[start:end]]


# 文件范围设置
ranges = [(0, 25), (25, 30), (30, 40)]

# 处理对比度1和对比度2的文件
for i, (start, end) in enumerate(ranges):
    t1_selected_files = get_file_names(t1_files, common_files, start, end, '-T1')
    t2_selected_files = get_file_names(t2_files, common_files, start, end, '-T2')

    process_data(ixi_t1_dir, t1_selected_files, os.path.join(base_dir,
                                                             f"data_train_contrast1.mat" if i == 0 else f"data_val_contrast1.mat" if i == 1 else f"data_test_contrast1.mat"))
    process_data(ixi_t2_dir, t2_selected_files, os.path.join(base_dir,
                                                             f"data_train_contrast2.mat" if i == 0 else f"data_val_contrast2.mat" if i == 1 else f"data_test_contrast2.mat"))