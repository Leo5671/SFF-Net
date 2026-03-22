# -*- coding: utf-8 -*-

import os
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.loss import Fusionloss, cc, infoNCE_loss, EnhancedFrequencyLoss
import kornia
from model.kernel_loss import kernelLoss
from utils.evaluator import average_similarity
from model.net import (
    Restormer_Encoder,
    Restormer_Decoder,
    Restormer_Decoder1,
    BaseFeatureExtractor,
    DetailFeatureExtractor,
)
from utils.dataset import H5Dataset
import subprocess

"""
------------------------------------------------------------------------------
Environment Settings
------------------------------------------------------------------------------
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
GPU_number = os.environ["CUDA_VISIBLE_DEVICES"]

"""
------------------------------------------------------------------------------
Loss Function
------------------------------------------------------------------------------
"""
gaussianLoss = kernelLoss("gaussian")
laplaceLoss = kernelLoss("laplace")
freq_loss = EnhancedFrequencyLoss().cuda()
MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
criteria_fusion = Fusionloss()
Loss_ssim = kornia.losses.SSIM(11, reduction="mean")

"""
------------------------------------------------------------------------------
Training HyperParameters
------------------------------------------------------------------------------
"""
batch_size = 4
num_epochs = 40
windows_size = 11
lr = 1e-4
weight_decay = 0
clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5

"""
------------------------------------------------------------------------------
Loss Function Coefficient
------------------------------------------------------------------------------
"""
coeff_ssim = 5.0#5.0
coeff_mse = 1.0
coeff_tv = 5.0#5.0
coeff_decomp = 2.0
coeff_nice = 0.1
coeff_cc_basic = 2.0
coeff_gauss = 1.0
coeff_laplace = 1.0#1.0
coeff_freq = 1.0

"""
------------------------------------------------------------------------------
Save Format Settings
------------------------------------------------------------------------------
"""
result_name = f"epoch"

"""
------------------------------------------------------------------------------
Build Model
------------------------------------------------------------------------------
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
Encoder = nn.DataParallel(Restormer_Encoder()).to(device)
Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
Decoder1 = nn.DataParallel(Restormer_Decoder1()).to(device)
BaseFuseLayer = nn.DataParallel(BaseFeatureExtractor(dim=64, num_heads=8)).to(device)
DetailFuseLayer = nn.DataParallel(DetailFeatureExtractor(num_layers=1)).to(device)

optimizer1 = torch.optim.Adam(Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer5 = torch.optim.Adam(Decoder1.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)
scheduler5 = torch.optim.lr_scheduler.StepLR(optimizer5, step_size=optim_step, gamma=optim_gamma)

"""
------------------------------------------------------------------------------
DataSet and DataLoader
------------------------------------------------------------------------------
"""
trainloader = DataLoader(
    H5Dataset(r"data/dataSet4Training_imgsize_128_stride_200.h5"),
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

loader = {'train': trainloader}
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

"""
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
"""
torch.backends.cudnn.benchmark = True
total_start_time = time.time()

Encoder.train()
Decoder.train()
Decoder1.train()
BaseFuseLayer.train()
DetailFuseLayer.train()

for epoch in range(num_epochs):
    epoch_start_time = time.time()

    # 初始化累积变量
    total_loss1, total_mse_loss, total_cc_loss_phase1 = 0.0, 0.0, 0.0
    total_tv_loss, total_mmd_loss, total_laplace_loss = 0.0, 0.0, 0.0
    total_gauss_loss, total_ince_loss, total_ccb_loss = 0.0, 0.0, 0.0
    total_loss2, total_fusionloss, total_cc_loss_phase2 = 0.0, 0.0, 0.0
    total_sim_cos, total_sim_pearson, total_dist_euclidean = 0.0, 0.0, 0.0

    for i, (img_VI, img_IR) in enumerate(loader['train']):
        img_VI, img_IR = img_VI.cuda(), img_IR.cuda()

        # Phase I
        Encoder.zero_grad()
        Decoder.zero_grad()
        Decoder1.zero_grad()
        BaseFuseLayer.zero_grad()
        DetailFuseLayer.zero_grad()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()
        optimizer5.zero_grad()

        feature_V_B, feature_V_D, _ = Encoder(img_VI)
        feature_I_B, feature_I_D, _ = Encoder(img_IR)

        data_VI_hat, _ = Decoder(img_VI, feature_V_B, feature_V_D)
        data_IR_hat, _ = Decoder(img_IR, feature_I_B, feature_I_D)

        cc_loss_B = cc(feature_V_B, feature_I_B)
        cc_loss_D = cc(feature_V_D, feature_I_D)

        ssim_loss = coeff_ssim * (Loss_ssim(img_IR, data_IR_hat) + Loss_ssim(img_VI, data_VI_hat))
        mse_loss = coeff_mse * (MSELoss(img_VI, data_VI_hat) + MSELoss(img_IR, data_IR_hat))
        tv_loss = coeff_tv * (
            L1Loss(kornia.filters.SpatialGradient()(img_VI), kornia.filters.SpatialGradient()(data_VI_hat)) +
            L1Loss(kornia.filters.SpatialGradient()(img_IR), kornia.filters.SpatialGradient()(data_IR_hat))
        )
        cc_loss = coeff_decomp * (cc_loss_D ** 2) / (1.01 + cc_loss_B)
        laplace_loss = coeff_laplace * laplaceLoss(feature_V_B, feature_I_B)
        gauss_loss = coeff_gauss * gaussianLoss(feature_V_B, feature_I_B)
        ince_loss = coeff_nice * infoNCE_loss(feature_V_B, feature_I_B)
        basic_cc_loss = coeff_cc_basic * cc_loss_B
        mmd_loss = laplace_loss + gauss_loss + basic_cc_loss + ince_loss
        loss1 = ssim_loss + mse_loss + cc_loss + tv_loss + mmd_loss

        similarity_cos = average_similarity(feature_V_B, feature_I_B, "cosine")
        similarity_pearson = average_similarity(feature_V_B, feature_I_B, "pearson")
        distance_euclidean = average_similarity(feature_V_B, feature_I_B, "euclidean")

        loss1.backward()
        nn.utils.clip_grad_norm_(Encoder.parameters(), clip_grad_norm_value)
        nn.utils.clip_grad_norm_(Decoder1.parameters(), clip_grad_norm_value)
        optimizer1.step()
        optimizer5.step()

        # Phase II
        feature_V_B, feature_V_D, feature_V = Encoder(img_VI)
        feature_I_B, feature_I_D, feature_I = Encoder(img_IR)

        feature_F_B = BaseFuseLayer(feature_I_B + feature_V_B)
        feature_F_D = DetailFuseLayer(feature_I_D + feature_V_D)

        data_Fuse, feature_F= Decoder1(img_VI, img_IR, feature_F_B, feature_F_D)

        cc_loss_B = cc(feature_V_B, feature_I_B)
        cc_loss_D = cc(feature_V_D, feature_I_D)
        cc_loss = coeff_decomp * (cc_loss_D ** 2) / (1.01 + cc_loss_B)
        fusionloss, _, _ = criteria_fusion(img_VI, img_IR, data_Fuse)
        loss_dict = freq_loss(data_Fuse, img_IR, img_VI)
        loss2 = fusionloss + coeff_freq * loss_dict["total"] + cc_loss
        

        loss2.backward()
        nn.utils.clip_grad_norm_(Encoder.parameters(), clip_grad_norm_value)
        nn.utils.clip_grad_norm_(Decoder1.parameters(), clip_grad_norm_value)
        nn.utils.clip_grad_norm_(BaseFuseLayer.parameters(), clip_grad_norm_value)
        nn.utils.clip_grad_norm_(DetailFuseLayer.parameters(), clip_grad_norm_value)
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        optimizer4.step()
        optimizer5.step()

        # 累积指标
        batch_samples = img_VI.size(0)
        total_loss1 += loss1.item() * batch_samples
        total_mse_loss += mse_loss.item() * batch_samples
        total_cc_loss_phase1 += cc_loss.item() * batch_samples
        total_tv_loss += tv_loss.item() * batch_samples
        total_mmd_loss += mmd_loss.item() * batch_samples
        total_laplace_loss += laplace_loss.item() * batch_samples
        total_gauss_loss += gauss_loss.item() * batch_samples
        total_ince_loss += ince_loss.item() * batch_samples
        total_ccb_loss += basic_cc_loss.item() * batch_samples

        total_loss2 += loss2.item() * batch_samples
        total_fusionloss += fusionloss.item() * batch_samples
        total_cc_loss_phase2 += cc_loss.item() * batch_samples

        total_sim_cos += similarity_cos * batch_samples
        total_sim_pearson += similarity_pearson * batch_samples
        total_dist_euclidean += distance_euclidean * batch_samples

    # 计算epoch平均值
    num_samples = len(loader['train'].dataset)
    avg_loss1 = total_loss1 / num_samples
    avg_mse = total_mse_loss / num_samples
    avg_cc_phase1 = total_cc_loss_phase1 / num_samples
    avg_tv = total_tv_loss / num_samples
    avg_mmd = total_mmd_loss / num_samples
    avg_laplace = total_laplace_loss / num_samples
    avg_gauss = total_gauss_loss / num_samples
    avg_ince = total_ince_loss / num_samples
    avg_ccb = total_ccb_loss / num_samples

    avg_loss2 = total_loss2 / num_samples
    avg_fusion = total_fusionloss / num_samples
    avg_cc_phase2 = total_cc_loss_phase2 / num_samples

    avg_sim_cos = total_sim_cos / num_samples
    avg_sim_pearson = total_sim_pearson / num_samples
    avg_dist_euclidean = total_dist_euclidean / num_samples

    # 计算剩余时间
    elapsed_time = time.time() - total_start_time
    avg_epoch_time = elapsed_time / (epoch + 1)
    remaining_epochs = num_epochs - (epoch + 1)
    time_left = str(datetime.timedelta(seconds=int(avg_epoch_time * remaining_epochs)))

    # 打印epoch信息
    print(f"\n[Epoch {epoch+1}/{num_epochs}]\n"
          f"L1: {avg_loss1:.2f} | MSE: {avg_mse:.2f} | CC1: {avg_cc_phase1:.2f}|"
          f"TV: {avg_tv:.2f} | MMD: {avg_mmd:.2f} | Laplace: {avg_laplace:.2f}|"
          f"Gauss: {avg_gauss:.2f} | InCE: {avg_ince:.2f} | CCB: {avg_ccb:.2f}\n"
          f"L2: {avg_loss2:.2f} | Fusion: {avg_fusion:.2f} | CC2: {avg_cc_phase2:.2f}|"
          f"SimCos: {avg_sim_cos:.2f} | SimPearson: {avg_sim_pearson:.2f} | DistEuclid: {avg_dist_euclidean:.2f}\n"
          f"Epoch Time: {datetime.timedelta(seconds=int(time.time() - epoch_start_time))} | "
          f"Remaining: {time_left}\n"
          "------------------------------------------------------------------")

  
    if (epoch + 1) % 40 == 0:
        save_path = os.path.join(f"checkPoints/{result_name}{epoch+1}.pth")
        checkpoint = {
            'DIDF_Encoder': Encoder.state_dict(),
            'DIDF_Decoder': Decoder1.state_dict(),
            'BaseFuseLayer': BaseFuseLayer.state_dict(),
            'DetailFuseLayer': DetailFuseLayer.state_dict(),
        }
        torch.save(checkpoint, save_path)

    # 学习率调整
    for param_group in optimizer1.param_groups:
        param_group['lr'] = max(param_group['lr'], 1e-6)
    for param_group in optimizer2.param_groups:
        param_group['lr'] = max(param_group['lr'], 1e-6)
    for param_group in optimizer3.param_groups:
        param_group['lr'] = max(param_group['lr'], 1e-6)
    for param_group in optimizer4.param_groups:
        param_group['lr'] = max(param_group['lr'], 1e-6)
    for param_group in optimizer5.param_groups:
        param_group['lr'] = max(param_group['lr'], 1e-6)

# 训练完成后运行测试
try:
    subprocess.run(["python", "test.py"], check=True)
    print("测试脚本已成功执行")
except subprocess.CalledProcessError as e:
    print(f"测试脚本执行失败: {e}")
except Exception as e:
    print(f"未知错误: {e}")
