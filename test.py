from model.net import (
    Restormer_Encoder,
    Restormer_Decoder1,
    BaseFeatureExtractor,
    DetailFeatureExtractor,
)
import os
import numpy as np
from utils.evaluator import Evaluator
import torch
import torch.nn as nn
from utils.imageUtils import img_save, image_read_cv2
import warnings
import logging
import cv2

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
ckpt_path = r"checkPoints/epoch40.pth"
print(f"{ckpt_path}\n")
for dataset_name in ["TNO", "RoadScene", "MSRS"]:  #
#for dataset_name in ["TNO", "RoadScene", "MSRS", "M3FD"]:  #
    print(f"The test result of {dataset_name}:")
    test_folder = os.path.join("test_img", dataset_name)
    test_out_folder = os.path.join("test_result", dataset_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
    Decoder = nn.DataParallel(Restormer_Decoder1()).to(device)
    BaseFuseLayer = nn.DataParallel(BaseFeatureExtractor(dim=64, num_heads=8)).to(
        device
    )
    DetailFuseLayer = nn.DataParallel(DetailFeatureExtractor(num_layers=1)).to(device)

    Encoder.load_state_dict(torch.load(ckpt_path)["DIDF_Encoder"])
    Decoder.load_state_dict(torch.load(ckpt_path)["DIDF_Decoder"])
    BaseFuseLayer.load_state_dict(torch.load(ckpt_path)["BaseFuseLayer"])
    DetailFuseLayer.load_state_dict(torch.load(ckpt_path)["DetailFuseLayer"])
    Encoder.eval()
    Decoder.eval()
    BaseFuseLayer.eval()
    DetailFuseLayer.eval()

    with torch.no_grad():
        for img_name in os.listdir(os.path.join(test_folder, "ir")):
            # 读取红外图像（灰度）
            data_IR = image_read_cv2(os.path.join(test_folder, "ir", img_name), mode='GRAY')[
                          np.newaxis, np.newaxis, ...] / 255.0

            # 读取可见光图像的 Y 通道
            vi_image = image_read_cv2(os.path.join(test_folder, "vi", img_name), mode='YCrCb')
            data_VIS = cv2.split(vi_image)[0][np.newaxis, np.newaxis, ...] / 255.0

            # 保持 Cr 和 Cb 通道
            data_VIS_BGR = cv2.imread(os.path.join(test_folder, "vi", img_name))
            _, data_VIS_Cr, data_VIS_Cb = cv2.split(cv2.cvtColor(data_VIS_BGR, cv2.COLOR_BGR2YCrCb))

            # 将数据转换为张量并移动到设备上
            data_IR, data_VIS = torch.FloatTensor(data_IR).to(device), torch.FloatTensor(data_VIS).to(device)

            # 特征提取和融合
            feature_V_B, feature_V_D, feature_V = Encoder(data_VIS)
            feature_I_B, feature_I_D, feature_I = Encoder(data_IR)
            feature_F_B = BaseFuseLayer(feature_V_B + feature_I_B)
            feature_F_D = DetailFuseLayer(feature_V_D + feature_I_D)
            data_Fuse, _ = Decoder(data_VIS, data_IR, feature_F_B, feature_F_D)

            # 将数据归一化到 [0, 1] 范围
            data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (torch.max(data_Fuse) - torch.min(data_Fuse))

            # 存RGB图
            fi = np.squeeze((data_Fuse * 255.0).cpu().numpy()).astype(np.uint8)
            ycrcb_fi = np.dstack((fi, data_VIS_Cr, data_VIS_Cb))
            rgb_fi = cv2.cvtColor(ycrcb_fi, cv2.COLOR_YCrCb2RGB)
            img_save(rgb_fi, img_name.split(sep='.')[0], test_out_folder)

    eval_folder = test_out_folder
    ori_img_folder = test_folder

