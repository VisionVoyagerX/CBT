import argparse
import yaml
from pathlib import Path
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import L1Loss
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
from torchmetrics import MetricCollection, PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchinfo import summary

from model.CrossFormer import *
from data_loader.DataLoader import DIV2K, GaoFen2, Sev2Mod, WV3
from utils import *
import os
import numpy as np


def main(args):
    def scaleMinMax(x):
        return ((x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x)))
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    config_file = args.config
    # try:
    with open(get_config_path() / config_file, 'r') as file:
        config_data = yaml.safe_load(file)

        # input pipeline
        tr_dataset = eval(config_data['data_pipeline']
                          ['train']['dataset'])  # str to object
        tr_path = Path(config_data['data_pipeline']['train']['path'])
        tr_mslr_size = config_data['data_pipeline']['train']['mslr_img_size']
        tr_pan_size = config_data['data_pipeline']['train']['pan_img_size']

        tr_augmentation_list = []
        tr_shuffle = config_data['data_pipeline']['train']['preprocessing']['shuffle']
        tr_cropping_on_the_fly = config_data['data_pipeline']['train']['preprocessing']['cropping_on_the_fly']
        if config_data['data_pipeline']['train']['preprocessing']['RandomHorizontalFlip']['enable']:
            tr_augmentation_list.append(RandomHorizontalFlip(
                p=config_data['data_pipeline']['train']['preprocessing']['RandomHorizontalFlip']['prob']))
        if config_data['data_pipeline']['train']['preprocessing']['RandomVerticalFlip']['enable']:
            tr_augmentation_list.append(RandomVerticalFlip(
                p=config_data['data_pipeline']['train']['preprocessing']['RandomVerticalFlip']['prob']))
        if config_data['data_pipeline']['train']['preprocessing']['RandomRotation']['enable']:
            tr_augmentation_list.append(RandomRotation(
                degrees=config_data['data_pipeline']['train']['preprocessing']['RandomRotation']['degrees']))

        val_dataset = eval(
            config_data['data_pipeline']['validation']['dataset'])
        val_path = Path(config_data['data_pipeline']['validation']['path'])
        val_mslr_size = config_data['data_pipeline']['validation']['mslr_img_size']
        val_pan_size = config_data['data_pipeline']['validation']['pan_img_size']
        val_shuffle = config_data['data_pipeline']['validation']['preprocessing']['shuffle']
        val_cropping_on_the_fly = config_data['data_pipeline']['validation']['preprocessing']['cropping_on_the_fly']
        val_steps = config_data['data_pipeline']['validation']['val_steps']

        test_dataset = eval(config_data['data_pipeline']['test']['dataset'])
        test_path = Path(config_data['data_pipeline']['test']['path'])
        test_mslr_size = config_data['data_pipeline']['test']['mslr_img_size']
        test_pan_size = config_data['data_pipeline']['test']['pan_img_size']
        test_cropping_on_the_fly = config_data['data_pipeline']['test']['preprocessing']['cropping_on_the_fly']

        # general settings
        model_name = config_data['general_settings']['name']
        model_type = eval(config_data['general_settings']['model_type'])
        continue_from_checkpoint = True
        checkpoint_name = config_data['general_settings']['checkpoint_name']
        if checkpoint_name:
            checkpoint_path = get_checkpoint_path() / model_name / checkpoint_name

        # task
        upscale = config_data['task']['upscale']
        mslr_to_pan_scale = config_data['task']['mslr_to_pan_scale']

        # network configs
        patch_size = config_data['network']['patch_size']
        in_chans = config_data['network']['in_chans']
        embed_dim = config_data['network']['embed_dim']
        depths = config_data['network']['depths']
        num_heads = config_data['network']['num_heads']
        window_size = config_data['network']['window_size']
        compress_ratio = config_data['network']['compress_ratio']
        squeeze_factor = config_data['network']['squeeze_factor']
        conv_scale = config_data['network']['conv_scale']
        overlap_ratio = config_data['network']['overlap_ratio']
        mlp_ratio = config_data['network']['mlp_ratio']
        qkv_bias = config_data['network']['qkv_bias']
        qk_scale = config_data['network']['qk_scale']
        drop_rate = config_data['network']['drop_rate']
        attn_drop_rate = config_data['network']['attn_drop_rate']
        drop_path_rate = config_data['network']['drop_path_rate']
        norm_layer = eval(config_data['network']['norm_layer'])
        ape = config_data['network']['ape']
        patch_norm = config_data['network']['patch_norm']
        img_range = config_data['network']['img_range']
        upsampler = config_data['network']['upsampler']
        resi_connection = config_data['network']['resi_connection']

        # training_settings
        batch_size = config_data['training_settings']['batch_size']
        optimizer_type = eval(
            config_data['training_settings']['optimizer']['type'])
        learning_rate = config_data['training_settings']['optimizer']['learning_rate']
        betas = config_data['training_settings']['optimizer']['betas']
        lr_decay_type = eval(
            config_data['training_settings']['scheduler']['type'])
        lr_gamma = config_data['training_settings']['scheduler']['gamma']
        loss_type = eval(config_data['training_settings']['loss']['type'])

    '''except FileNotFoundError:
        print(f"Config file '{get_config_path() / config_file}' not found.")
        return
    except yaml.YAMLError as exc:
        print(f"Error while parsing YAML in config file '{config_file}': {exc}")
        return
'''
    # Prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Initialize DataLoader
    train_dataset = tr_dataset(
        tr_path, transforms=[(RandomHorizontalFlip(1), 0.3), (RandomVerticalFlip(1), 0.3)])
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=tr_shuffle, drop_last=True)

    validation_dataset = val_dataset(
        val_path)
    validation_loader = DataLoader(
        dataset=validation_dataset, batch_size=batch_size, shuffle=val_shuffle)

    te_dataset = test_dataset(
        test_path)
    test_loader = DataLoader(
        dataset=te_dataset, batch_size=1, shuffle=False)

    # Initialize Model, optimizer, criterion and metrics
    model = model_type(pan_img_size=(tr_pan_size[0], tr_pan_size[1]), pan_low_size_ratio=mslr_to_pan_scale, patch_size=patch_size, in_chans=in_chans,
                       embed_dim=embed_dim, depths=depths, num_heads=num_heads, window_size=window_size, compress_ratio=compress_ratio,
                       squeeze_factor=squeeze_factor, conv_scale=conv_scale, overlap_ratio=overlap_ratio, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                       qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer,
                       ape=ape, patch_norm=patch_norm, upscale=upscale, img_range=img_range, upsampler=upsampler, resi_connection=resi_connection,
                       mslr_mean=train_dataset.mslr_mean.to(device), mslr_std=train_dataset.mslr_std.to(device), pan_mean=train_dataset.pan_mean.to(device),
                       pan_std=train_dataset.pan_std.to(device)).to(device)

    optimizer = optimizer_type(
        model.parameters(), lr=learning_rate, betas=(betas[0], betas[1]))
    criterion = loss_type().to(device)

    val_metric_collection = MetricCollection({
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': StructuralSimilarityIndexMeasure().to(device)
    })

    test_metric_collection = MetricCollection({
        'psnr': PeakSignalNoiseRatio().to(device),
        'ssim': StructuralSimilarityIndexMeasure().to(device)
    })

    val_report_loss = 0
    test_report_loss = 0
    tr_metrics = []
    val_metrics = []
    test_metrics = []

    '''summary(model, [(1, 1, 256, 256), (1, in_chans, 64, 64)],
            dtypes=[torch.float32, torch.float32])'''

    # load checkpoint
    if continue_from_checkpoint:
        tr_metrics, val_metrics = load_checkpoint(torch.load(
            checkpoint_path), model, optimizer, tr_metrics, val_metrics)

    """# evaluation mode
    model.eval()
    with torch.no_grad():
        print("\n==> Start evaluating ...")
        eval_progress_bar = tqdm(iter(validation_loader), total=len(
            validation_loader), desc="Evaluation", leave=False, bar_format='{desc:<8}{percentage:3.0f}%|{bar:15}{r_bar}')
        for pan, mslr, mshr in eval_progress_bar:
            # forward
            pan, mslr, mshr = pan.to(device), mslr.to(device), mshr.to(device)
            mssr = model(pan, mslr)
            val_loss = criterion(mssr, mshr)
            val_metric = val_metric_collection.forward(mssr, mshr)
            val_report_loss += val_loss

            # report metrics
            eval_progress_bar.set_postfix(
                loss=f'{val_loss.item()}', psnr=f'{val_metric["psnr"].item():.2f}', ssim=f'{val_metric["ssim"].item():.2f}')

        # compute metrics total
        val_report_loss = val_report_loss / len(validation_loader)
        val_metric = val_metric_collection.compute()
        val_metrics.append({'loss': val_report_loss.item(),
                            'psnr': val_metric['psnr'].item(),
                            'ssim': val_metric['ssim'].item()})

        print(
            f'\nEvaluation: avg_loss = {val_report_loss.item():.4f} , avg_psnr= {val_metric["psnr"]:.4f}, avg_ssim={val_metric["ssim"]:.4f}')

        # reset metrics
        val_report_loss = 0
        val_metric_collection.reset()
        print("==> End evaluating <==\n")"""

    # test model
    model.eval()
    with torch.no_grad():
        print("\n==> Start testing ...")
        test_progress_bar = tqdm(iter(test_loader), total=len(
            test_loader), desc="Testing", leave=False, bar_format='{desc:<8}{percentage:3.0f}%|{bar:15}{r_bar}')
        for pan, mslr, mshr in test_progress_bar:
            # forward
            pan, mslr, mshr = pan.to(device), mslr.to(
                device), mshr.to(device)
            mssr = model(pan, mslr)
            test_loss = criterion(mssr, mshr)
            test_metric = test_metric_collection.forward(mssr, mshr)
            test_report_loss += test_loss

            # report metrics
            test_progress_bar.set_postfix(
                loss=f'{test_loss.item()}', psnr=f'{test_metric["psnr"].item():.2f}', ssim=f'{test_metric["ssim"].item():.2f}')

        # compute metrics total
        test_report_loss = test_report_loss / len(test_loader)
        test_metric = test_metric_collection.compute()
        test_metrics.append({'loss': test_report_loss.item(),
                             'psnr': test_metric['psnr'].item(),
                             'ssim': test_metric['ssim'].item()})

        print(
            f'\nTesting: avg_loss = {test_report_loss.item():.4f} , avg_psnr= {test_metric["psnr"]:.4f}, avg_ssim={test_metric["ssim"]:.4f}')

        # reset metrics
        test_report_loss = 0
        test_metric_collection.reset()
        print("==> End testing <==\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation of CrossFormer')
    parser.add_argument('-c', '--config', type=str,
                        help='config file name', required=True)

    args = parser.parse_args()

    main(args)
