import os
import nibabel as nib
import numpy as np
import h5py


def load_nii_files(directory, file_names, t1_directory=None):
    data_list = []
    for file in file_names:
        file_path = os.path.join(directory, file)
        img = nib.load(file_path)
        img_data = img.get_fdata()

        if t1_directory:
            t1_file_path = os.path.join(t1_directory, file.replace('-T2', '-T1'))
            t1_img = nib.load(t1_file_path)
            t1_shape = t1_img.shape
            if img_data.shape[2] < t1_shape[2]:
                padding = ((0, 0), (0, 0), (0, t1_shape[2] - img_data.shape[2]))
                img_data = np.pad(img_data, padding, mode='constant', constant_values=0)
                print(f"Padded file: {file}, new shape: {img_data.shape}")
            else:
                img_data = img_data[:, :, :t1_shape[2]]
                print(f"Cropped file: {file}, new shape: {img_data.shape}")

        img_data_reshaped = np.transpose(img_data, (2, 0, 1))  # 转换形状为 (#image, width, height)
        img_data_reshaped = (img_data_reshaped - np.min(img_data_reshaped)) / (
                    np.max(img_data_reshaped) - np.min(img_data_reshaped))
        data_list.append(img_data_reshaped)
    return np.concatenate(data_list, axis=0)


def save_h5_file(data, output_path):
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('data', data=data)
        print(f"Saved {output_path} with shape {data.shape}")


def process_data(input_directory, file_names, output_path, t1_directory=None):
    data = load_nii_files(input_directory, file_names, t1_directory)
    save_h5_file(data, output_path)


# 文件路径设置
base_dir = "D:/桌面/深度学习/data/IXI"
ixi_t1_dir = os.path.join(base_dir, "IXI-T1")
ixi_t2_dir = os.path.join(base_dir, "IXI-T2")

# 获取文件名
t1_files = sorted([f for f in os.listdir(ixi_t1_dir) if f.endswith('.nii.gz')])
t2_files = sorted([f for f in os.listdir(ixi_t2_dir) if f.endswith('.nii.gz')])

# 提取共有文件名
common_files = sorted(set(f.replace('-T1.nii.gz', '') for f in t1_files).intersection(
    f.replace('-T2.nii.gz', '') for f in t2_files))

# 确保两者共有的前25个文件为train，另外5个为val，最后10个为test
train_files = common_files[:25]
val_files = common_files[25:30]
test_files = common_files[30:40]


# 定义保存路径和文件处理函数
def process_and_save_data(file_list, subset_name):
    t1_selected_files = [f"{name}-T1.nii.gz" for name in file_list]
    t2_selected_files = [f"{name}-T2.nii.gz" for name in file_list]
    process_data(ixi_t1_dir, t1_selected_files, os.path.join(base_dir, f"data_{subset_name}_T1.h5"))
    process_data(ixi_t2_dir, t2_selected_files, os.path.join(base_dir, f"data_{subset_name}_T2.h5"),
                 t1_directory=ixi_t1_dir)


# 处理数据集
process_and_save_data(train_files, 'train')
process_and_save_data(val_files, 'val')
process_and_save_data(test_files, 'test')