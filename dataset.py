# import torch.utils.data
# import numpy as np, h5py
# import random
#
#
# def CreateDatasetSynthesis(phase, input_path, contrast1 = 'T1', contrast2 = 'T2'):
#     '''
#
#     创建一个包含两种对比度图像数据的 PyTorch 数据集
#     Args:
#         phase: 数据集阶段（例如，训练、验证或测试）
#         input_path:数据存储的路径
#         contrast1:
#         contrast2:图像对比度类型
#
#     Returns:
#
#     '''
#
#     target_file = input_path + "/data_{}_{}.mat".format(phase, contrast1)
#     data_fs_s1=LoadDataSet(target_file)
#
#     target_file = input_path + "/data_{}_{}.mat".format(phase, contrast2)
#     data_fs_s2=LoadDataSet(target_file)
#
#     dataset=torch.utils.data.TensorDataset(torch.from_numpy(data_fs_s1),torch.from_numpy(data_fs_s2))
#     return dataset
#
#
#
# #Dataset loading from load_dir and converintg to 256x256
# def LoadDataSet(load_dir, variable = 'data_fs', padding = True, Norm = True):
#     '''
#
#     加载数据集并对其进行预处理
#     Args:
#         load_dir:数据文件的路径
#         variable:要加载的数据变量名（默认 'data_fs'）
#         padding:是否对图像进行填充以达到 256x256 的大小（默认 True）
#         Norm:是否对数据进行归一化（默认 True）
#
#     Returns:
#
#     '''
#     f = h5py.File(load_dir,'r')
#     if np.array(f[variable]).ndim==3:
#         data=np.expand_dims(np.transpose(np.array(f[variable]),(0,2,1)),axis=1)
#     else:
#         data=np.transpose(np.array(f[variable]),(1,0,3,2))
#     data=data.astype(np.float32)
#     if padding:
#         pad_x=int((256-data.shape[2])/2)
#         pad_y=int((256-data.shape[3])/2)
#         print('padding in x-y with:'+str(pad_x)+'-'+str(pad_y))
#         data=np.pad(data,((0,0),(0,0),(pad_x,pad_x),(pad_y,pad_y)))
#     if Norm:
#         data=(data-0.5)/0.5
#     return data
import torch.utils.data
import numpy as np
import pydicom
import os

def CreateDatasetSynthesis(phase, input_path, contrast1='T1', contrast2='T2'):
    if input_path.endswith('.mat'):
        target_file = input_path + "/data_{}_{}.mat".format(phase, contrast1)
        data_fs_s1 = LoadDataSet(target_file)

        target_file = input_path + "/data_{}_{}.mat".format(phase, contrast2)
        data_fs_s2 = LoadDataSet(target_file)
    else:
        data_fs_s1 = LoadDataSet(input_path + "/" + phase + "/" + contrast1, dicom=True)
        data_fs_s2 = LoadDataSet(input_path + "/" + phase + "/" + contrast2, dicom=True)

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(data_fs_s1), torch.from_numpy(data_fs_s2))
    return dataset

def LoadDataSet(load_dir, variable='data_fs', padding=True, Norm=True, dicom=False):
    if dicom:
        return LoadDataSetDICOM(load_dir, padding, Norm)
    else:
        return LoadDataSetMAT(load_dir, variable, padding, Norm)

def LoadDataSetMAT(load_dir, variable='data_fs', padding=True, Norm=True):
    import h5py  # Import here to ensure h5py is only required for MAT files
    f = h5py.File(load_dir, 'r')
    if np.array(f[variable]).ndim == 3:
        data = np.expand_dims(np.transpose(np.array(f[variable]), (0, 2, 1)), axis=1)
    else:
        data = np.transpose(np.array(f[variable]), (1, 0, 3, 2))
    data = data.astype(np.float32)
    if padding:
        pad_x = int((256 - data.shape[2]) / 2)
        pad_y = int((256 - data.shape[3]) / 2)
        print('padding in x-y with:' + str(pad_x) + '-' + str(pad_y))
        data = np.pad(data, ((0, 0), (0, 0), (pad_x, pad_x), (pad_y, pad_y)))
    if Norm:
        data = (data - 0.5) / 0.5
    return data

def LoadDataSetDICOM(directory, padding=True, Norm=True):
    dicom_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.dcm')]
    dicom_files.sort()  # Ensure the files are sorted correctly

    data = []
    for file in dicom_files:
        ds = pydicom.dcmread(file)
        img = ds.pixel_array
        data.append(img)

    data = np.array(data, dtype=np.float32)
    data = np.expand_dims(data, axis=1)  # Add channel dimension

    if padding:
        pad_x = int((256 - data.shape[2]) / 2)
        pad_y = int((256 - data.shape[3]) / 2)
        print('padding in x-y with:' + str(pad_x) + '-' + str(pad_y))
        data = np.pad(data, ((0, 0), (0, 0), (pad_x, pad_x), (pad_y, pad_y)))

    if Norm:
        data = (data - np.min(data)) / (np.max(data) - np.min(data))  # Normalize to [0, 1]
        data = (data - 0.5) / 0.5  # Normalize to [-1, 1]

    return data