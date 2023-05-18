import argparse
import os

import numpy as np
from PIL import Image
from PIL.Image import Resampling
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as Transforms
from torchvision.models.inception import inception_v3
import torch_fidelity
from tqdm import tqdm

import eval_models as models


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', default='LPIPS')
    parser.add_argument('--predict_dir', default='./result/bg_ver1/output/')
    parser.add_argument('--ground_truth_dir', default='./data/zalando-hd-resize/test/image')
    parser.add_argument('--resolution', type=int, default=1024)

    opt = parser.parse_args()
    return opt


def Evaluation(opt, pred_list, gt_list):
    T1 = Transforms.ToTensor()
    T2 = Transforms.Compose([Transforms.Resize((128, 128)),
                             Transforms.ToTensor(),
                             Transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                  std=(0.5, 0.5, 0.5))])
    T3 = Transforms.Compose([Transforms.Resize((299, 299)),
                             Transforms.ToTensor(),
                             Transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                  std=(0.5, 0.5, 0.5))])

    splits = 1  # Hyper-parameter for IS score

    # lpips_model = lpips.LPIPS(net='alex').cuda()
    torch.cuda.set_device('cuda:2')
    model = models.PerceptualLoss(model='net-lin', net='alex', use_gpu=True, gpu_ids=[2])
    model.eval()
    avg_ssim, avg_mse, avg_distance = 0.0, 0.0, 0.0
    preds = np.zeros((len(gt_list), 1000))
    lpips_list = []
    iterator = tqdm(pred_list, desc="Calculate SSIM, MSE, LPIPS...", total=len(pred_list))
    with torch.no_grad():
        for i, img_pred in enumerate(iterator):
            # Calculate SSIM
            gt_img = Image.open(os.path.join(opt.ground_truth_dir, gt_list[i]))
            gt_np = np.asarray(gt_img.convert('L'))

            pred_img = Image.open(os.path.join(opt.predict_dir, img_pred))
            pred_img = pred_img.resize(gt_img.size, Resampling.BICUBIC)
            assert gt_img.size == pred_img.size, f"{gt_img.size} vs {pred_img.size}"
            pred_np = np.asarray(pred_img.convert('L'))
            avg_ssim += ssim(gt_np, pred_np, data_range=255, gaussian_weights=True, use_sample_covariance=False)

            # Calculate LPIPS
            gt_img_LPIPS = T2(gt_img).unsqueeze(0).cuda()
            pred_img_LPIPS = T2(pred_img).unsqueeze(0).cuda()
            lpips_list.append((img_pred, model.forward(gt_img_LPIPS, pred_img_LPIPS).item()))
            avg_distance += lpips_list[-1][1]
            # gt_img_LPIPS = T1(gt_img).unsqueeze(0).cuda()
            # pred_img_LPIPS = T1(pred_img).unsqueeze(0).cuda()
            # lpips_list.append(lpips_model(pred_img_LPIPS, gt_img_LPIPS).squeeze().item())
            # avg_distance += lpips_list[-1]

            gt_img_MSE = T1(gt_img).unsqueeze(0).cuda()
            pred_img_MSE = T1(pred_img).unsqueeze(0).cuda()
            avg_mse += F.mse_loss(gt_img_MSE, pred_img_MSE)

            # print(f"step: {i + 1} evaluation... lpips:{lpips_list[-1]}")

        avg_ssim /= len(gt_list)
        avg_mse = avg_mse / len(gt_list)
        avg_distance = avg_distance / len(gt_list)

    f = open(os.path.join(opt.predict_dir, 'eval.txt'), 'a')
    f.write(f"SSIM : {avg_ssim:06} / MSE : {avg_mse:06} / LPIPS : {avg_distance:06}\n")

    f.close()
    return avg_ssim, avg_mse, avg_distance


def main():
    opt = get_opt()

    # Output과 Ground Truth Data
    pred_list = os.listdir(opt.predict_dir)
    gt_list = os.listdir(opt.ground_truth_dir)
    pred_list.sort()
    gt_list.sort()

    avg_ssim, avg_mse, avg_distance = Evaluation(opt, pred_list, gt_list)
    print("SSIM : %.6f / MSE : %.6f / LPIPS : %.6f" % (avg_ssim, avg_mse, avg_distance))


if __name__ == '__main__':
    main()
