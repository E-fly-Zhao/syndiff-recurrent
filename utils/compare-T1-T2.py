# import os
# import nibabel as nib
# import numpy as np
#
#
# def load_nii_data(file_path):
#     img = nib.load(file_path)
#     return img.get_fdata()
#
#
# def compare_files(dir1, dir2):
#     # 获取文件列表
#     files1 = {f.replace('-T1.nii.gz', '') for f in os.listdir(dir1) if f.endswith('.nii.gz')}
#     files2 = {f.replace('-T2.nii.gz', '') for f in os.listdir(dir2) if f.endswith('.nii.gz')}
#
#     # 找到共同的文件名
#     common_files = files1.intersection(files2)
#
#     # 比较共同文件的张量
#     for file_name in common_files:
#         file1_path = os.path.join(dir1, file_name + '-T1.nii.gz')
#         file2_path = os.path.join(dir2, file_name + '-T2.nii.gz')
#
#         data1 = load_nii_data(file1_path)
#         data2 = load_nii_data(file2_path)
#
#         # 检查张量的形状和内容是否一致
#         if data1.shape != data2.shape:
#             print(f"Shape mismatch for {file_name}: {data1.shape} vs {data2.shape}")
#         elif not np.allclose(data1, data2):
#             print(f"Data mismatch for {file_name}")
#         else:
#             print(f"Files match: {file_name}")
#
#
# # 设置目录路径
# dir1 = "D:/桌面/深度学习/data/IXI/IXI-T1"
# dir2 = "D:/桌面/深度学习/data/IXI/IXI-T2"
#
# # 执行比较
# compare_files(dir1, dir2)

import os
import nibabel as nib
import matplotlib.pyplot as plt

# 文件路径
t1_file = "D:/桌面/深度学习/data/IXI/IXI-T1/IXI002-Guys-0828-T1.nii.gz"
t2_file = "D:/桌面/深度学习/data/IXI/IXI-T2/IXI002-Guys-0828-T2.nii.gz"

# 加载NIfTI文件
t1_img = nib.load(t1_file)
t2_img = nib.load(t2_file)

# 获取图像数据
t1_data = t1_img.get_fdata()
t2_data = t2_img.get_fdata()

# 可视化第slice_index片
slice_index_1 = 9
slice_index_2 = 0

# 检查切片索引是否在图像数据范围内
if slice_index_1 < t1_data.shape[2] and slice_index_2 < t2_data.shape[2]:
    plt.figure(figsize=(12, 6))

    # 可视化T1图像的第50片
    plt.subplot(1, 2, 1)
    plt.imshow(t1_data[:, :, slice_index_1], cmap="gray")
    plt.title("T1 Image - Slice 10")
    plt.axis("off")

    # 可视化T2图像的第50片
    plt.subplot(1, 2, 2)
    plt.imshow(t2_data[:, :, slice_index_2], cmap="gray")
    plt.title("T2 Image - Slice 1")
    plt.axis("off")

    plt.show()
else:
    print(f"Slice index {slice_index_1} and {slice_index_2} is out of range for one of the images.")