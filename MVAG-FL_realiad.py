import torch
import torch.nn as nn
from dataset import get_data_transforms, get_strong_transforms
from torchvision.datasets import ImageFolder
import numpy as np
import random
import os
from torch.utils.data import DataLoader, ConcatDataset

from models.CVFL import ViTV50, ViTillv2

from models import vit_encoder
from torch.nn.init import trunc_normal_
from models.vision_transformer import Block as VitBlock, bMlp, Attention, LinearAttention, \
    LinearAttention2,LinearAttentionV9, LinearAttentionV13
from dataset import MVTecDataset, RealIADDataset
import torch.backends.cudnn as cudnn
import argparse
from utils2 import evaluation_batch, global_cosine_hm_percent, regional_cosine_focal, \
    regional_cosine_hm, WarmCosineScheduler
from torch.nn import functional as F
from functools import partial
from ptflops import get_model_complexity_info
from optimizers import StableAdamW
import warnings
import copy
import logging
from sklearn.metrics import roc_auc_score, average_precision_score
import itertools

warnings.filterwarnings("ignore")


def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    logger.addHandler(streamHandler)

    if not save_path is None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def train(item_list, self=None):
    setup_seed(1)

    total_iters = 50000
    batch_size = 10
    image_size = 448
    crop_size = 392

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_data_list = []
    test_data_list = []
    for idx, category in enumerate(item_list):
        train_set = RealIADDataset(root=root_path, category=category, transform=data_transform,
                                   gt_transform=gt_transform, phase='train')
        train_set.classes, train_set.class_to_idx = category, {category: idx}

        test_set = RealIADDataset(root=root_path, category=category, transform=data_transform,
                                  gt_transform=gt_transform, phase="test")

        train_datasets.append(train_set)
        test_datasets.append(test_set)

    # Combine train datasets and create data loader
    train_dataloader = DataLoader(ConcatDataset(train_datasets), batch_size=batch_size,
                                  shuffle=True, num_workers=64, drop_last=True)

    # Initialize encoder and decoder
    encoder = vit_encoder.load('dinov2reg_vit_base_14')
    embed_dim, num_heads = 768, 12
    target_layers = list(range(2, 10))

    # Bottleneck layer
    bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.4)])

    # Decoder layers
    decoder = nn.ModuleList([
        VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-8), attn_drop=0., attn=LinearAttentionV18)
        for _ in range(8)
    ])

    # Build ViT model
    model = ViTV50(encoder=encoder, bottleneck=bottleneck, decoder=decoder, target_layers=target_layers,
                   mask_neighbor_size=7, fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
                   fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]])

    print_fn("\n===== Model Statistics =====")
    print_fn(f"Trainable Parameters: {count_parameters(model) / 1e6:.2f}M")
    input_shape = (1, 3, 392, 392)
    dummy_input = torch.randn(*input_shape).to(device)
    try:
        from thop import profile
        flops, _ = profile(model, inputs=(dummy_input,),
                           kwargs={'jitter_scale': 0, 'jitter_prob': 0},
                           verbose=False)
        print_fn(f"FLOPs: {flops / 1e9:.2f}G")
    except:
        print_fn("FLOPs calculation requires thop library: pip install thop")


    import time
    def benchmark(model, input_size=(392, 392), device='cuda:2', repeats=100):
        model.eval()
        dummy_input = torch.randn(1, 3, *input_size).to(device)
        model = model.to(device)
        for _ in range(10):
            _ = model(dummy_input)
        timings = []
        for _ in range(repeats):
            start = time.time()
            _ = model(dummy_input)
            torch.cuda.synchronize()
            end = time.time()
            timings.append(end - start)
        avg_time = sum(timings) / len(timings) * 1000
        return avg_time
    avg_infer_time = benchmark(model)
    print_fn(f"Inference Time: {avg_infer_time:.2f}ms per image")
    model = model.to(device)
    trainable = nn.ModuleList([bottleneck, decoder])

    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    optimizer = StableAdamW([{'params': trainable.parameters()}],
                            lr=2e-3, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(optimizer, base_value=2e-3, final_value=2e-4, total_iters=total_iters,
                                       warmup_iters=100)
    print_fn('train image number: {}'.format(len(train_data)))

    it = 0
    for epoch in range(int(np.ceil(total_iters / len(train_dataloader)))):
        model.train()
        loss_list = []
        for img, label in train_dataloader:
            img, label = img.to(device), label.to(device)
            scale = min(5.0, 2.0 + (5.0 - 2.0) * (it / 50000.0))
            prob = min(0.8, 0.2 + (0.8 - 0.2) * (it / 50000.0))

            en, de = model(img, jitter_scale=scale, jitter_prob=prob)

            p_final = 0.8
            p = min(p_final * it / 1000, p_final)

            loss1 = global_cosine_hm_percent(en, de, p=p, factor=0.1)
            loss2 = regional_cosine_focal(en, de, p=p)
            loss = 0.5 * loss1 + 0.5 * loss2

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()

            if (it + 1) % 50000 == 0 or it == 50000:
                print(1)
                torch.save(model.state_dict(), os.path.join(args.save_dir, args.save_name, 'model.pth'))

                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

                for item, test_data in zip(item_list, test_data_list):
                    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False,
                                                                 num_workers=4)
                    results = evaluation_batch(model, test_dataloader, device, max_ratio=0.01, resize_mask=256)
                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
                    auroc_sp_list.append(auroc_sp)
                    ap_sp_list.append(ap_sp)
                    f1_sp_list.append(f1_sp)
                    auroc_px_list.append(auroc_px)
                    ap_px_list.append(ap_px)
                    f1_px_list.append(f1_px)
                    aupro_px_list.append(aupro_px)
                    print_fn(
                        '{}: I-Auroc: {:.4f}, I-AP: {:.4f}, I-F1: {:.4f}, P-AUROC: {:.4f}, P-AP: {:.4f}, P-F1: {:.4f}, P-AUPRO: {:.4f}'.format(
                            item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))
                print_fn(
                    'Mean: I-Auroc: {:.4f}, I-AP: {:.4f}, I-F1: {:.4f}, P-AUROC: {:.4f}, P-AP: {:.4f}, P-F1: {:.4f}, P-AUPRO: {:.4f}'.format(
                        np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                        np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))
                model.train()
            it += 1
            if it == total_iters:
                break
    return


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "2"
    import argparse
    from datetime import datetime

    current_file_path = __file__
    current_file_name = os.path.basename(current_file_path)

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')  # Format time as "YYYYMMDD_HHMMSS"
    default_save_name = f"{current_time}-{current_file_name}V11"

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_path', type=str, default='/data/Real-IAD')
    parser.add_argument('--save_dir', type=str, default='./saved_results')
    parser.add_argument('--save_name', type=str, default=default_save_name)
    args = parser.parse_args()

    item_list = ['audiojack', 'bottle_cap', 'button_battery', 'end_cap', 'eraser', 'fire_hood',
                 'mint', 'mounts', 'pcb', 'phone_battery', 'plastic_nut', 'plastic_plug',
                 'porcelain_doll', 'regulator', 'rolled_strip_base', 'sim_card_set', 'switch', 'tape',
                 'terminalblock', 'toothbrush', 'toy', 'toy_brick', 'transistor1', 'usb',
                 'usb_adaptor', 'u_block', 'vcpill', 'wooden_beads', 'woodstick', 'zipper']

    # Initialize logger for tracking training process
    logger = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    train(item_list)