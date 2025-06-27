import torch
import numpy as np
import random
import os
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import json
from argparse import ArgumentParser
from model.resnet import wide_resnet50_2
from model.de_resnet import de_wide_resnet50_2
from model.apr import APR
from tools.eval_helper import evaluation
from loss import cosine_similarity_loss
from dataset.dataset import MVTecDataset_test, MVTecDataset_train

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--rec_lr', default=0.001, type=float)
    parser.add_argument('--distill_lr', default=0.005, type=float)
    parser.add_argument('--classes', nargs="+", default=['carpet'])
    parser.add_argument('--save_folder', default='./experiments/MVTec-AD', type=str)
    parser.add_argument('--dataset_path', default='/home/sylin/LTX/datasets/MVTec-AD', type=str)
    parser.add_argument('--anomaly_source_path', default='/home/sylin/LTX/datasets/dtd/images', type=str)
    pars = parser.parse_args()
    return pars

def train(_class_, pars):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(pars.save_folder + '/' + _class_):
        os.makedirs(pars.save_folder + '/' + _class_)
    save_model_path = pars.save_folder + '/' + _class_ + '/' + 'wres50_' + _class_ + '.pth'

    train_path = pars.dataset_path + '/' + _class_ + '/train'
    test_path = pars.dataset_path + '/' + _class_

    train_data = MVTecDataset_train(
        target=_class_,
        dataset_path=train_path,
        anomaly_source_path=pars.anomaly_source_path,
        resize = (pars.image_size, pars.image_size),
        perlin_scale=6,
        min_perlin_scale=0,
        perlin_noise_threshold=0.5,
        transparency_range=[0.15, 1.]
    )
    test_data = MVTecDataset_test(dataset_path=test_path, resize=(pars.image_size, pars.image_size))

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=pars.batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = torch.nn.DataParallel(encoder).to(device)
    bn = torch.nn.DataParallel(bn).to(device)
    encoder.eval()

    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = torch.nn.DataParallel(decoder).to(device)

    APR_module = APR(base=64, temp=0.5, sigma=0.1)
    APR_module = torch.nn.DataParallel(APR_module).to(device)

    optimizer_rec = torch.optim.Adam(list(APR_module.parameters()), lr=pars.apr_lr, betas=(0.5, 0.999))
    optimizer_distill = torch.optim.Adam(list(decoder.parameters()) + list(bn.parameters()), lr=pars.distill_lr, betas=(0.5, 0.999))

    best_score = 0
    best_epoch = 0
    best_auroc_px = 0
    best_auroc_sp = 0
    best_aupro_px = 0

    auroc_px_list = []
    auroc_sp_list = []
    aupro_px_list = []

    loss_distill = []
    total_loss = []

    history_infor = {}

    epoch_dict = {
        'carpet': 10, 'leather': 10, 'wood': 100, 'grid': 260, 'tile': 260,
        'hazelnut': 160, 'metal_nut': 160, 'pill': 200, 'bottle': 200, 'cable': 240,
        'toothbrush': 280, 'capsule': 400, 'screw': 350, 'transistor': 350, 'zipper': 350
    }
    num_epoch = epoch_dict.get(_class_, 0)

    print(f'class: {_class_}, Training with {num_epoch} Epoch, nums_gpu: {torch.cuda.device_count()}')
    for epoch in tqdm(range(1, num_epoch + 1)):
        bn.train()
        APR_module.train()
        decoder.train()
        loss_distill_running = 0
        total_loss_running = 0

        # gradient acc
        accumulation_steps = 2

        for i, (img, img_aug, _) in enumerate(train_dataloader):
            img = img.to(device)
            img_aug = img_aug.to(device)

            inputs = encoder(img_aug)
            inputs_normal = encoder(img)
            outputs = decoder(bn(inputs))
            outputs_rec = APR_module(outputs, inputs)

            l_kd = cosine_similarity_loss(inputs_normal, outputs_rec)
            loss = l_kd
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer_distill.step()
                optimizer_rec.step()
                optimizer_distill.zero_grad()
                optimizer_rec.zero_grad()

            total_loss_running += loss.detach().cpu().item()
            loss_distill_running += l_kd.detach().cpu().item()

        auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader, device, APR_module)
        auroc_px_list.append(auroc_px)
        auroc_sp_list.append(auroc_sp)
        aupro_px_list.append(aupro_px)

        loss_distill.append(loss_distill_running)
        total_loss.append(total_loss_running)

        figure = plt.gcf()  # get current figure
        figure.set_size_inches(8, 12)
        fig, ax = plt.subplots(3, 2, figsize=(8, 12))
        ax[0][0].plot(auroc_px_list)
        ax[0][0].set_title('auroc_px')
        ax[0][1].plot(auroc_sp_list)
        ax[0][1].set_title('auroc_sp')
        ax[1][0].plot(aupro_px_list)
        ax[1][0].set_title('aupro_px')
        ax[2][0].plot(loss_distill)
        ax[2][0].set_title('loss_distill')
        ax[2][1].plot(total_loss)
        ax[2][1].set_title('total_loss')
        plt.savefig(pars.save_folder + '/' + _class_ + '/monitor_traning.png', dpi=100)

        print('Epoch {}, Sample Auroc: {:.4f}, Pixel Auroc:{:.4f}, Pixel Aupro: {:.4f}'.format
            (epoch, auroc_sp, auroc_px, aupro_px)
        )

        if (auroc_px + auroc_sp + aupro_px) / 3 > best_score:
            best_score = (auroc_px + auroc_sp + aupro_px) / 3
            best_auroc_px = auroc_px
            best_auroc_sp = auroc_sp
            best_aupro_px = aupro_px
            best_epoch = epoch

            torch.save({
                'decoder': decoder.state_dict(),
                'bn': bn.state_dict(),
                'rec': APR_module.state_dict()}, save_model_path
            )

            history_infor['auroc_sp'] = best_auroc_sp
            history_infor['auroc_px'] = best_auroc_px
            history_infor['aupro_px'] = best_aupro_px
            history_infor['epoch'] = best_epoch
            with open(os.path.join(pars.save_folder + '/' + _class_, f'history.json'), 'w') as f:
                json.dump(history_infor, f)
    return best_auroc_sp, best_auroc_px, best_aupro_px


if __name__ == '__main__':
    setup_seed(111)
    pars = get_args()
    mvtec_classes = ['carpet', 'leather', 'grid', 'tile', 'wood', 'cable', 'capsule', 'pill', 'screw', 'transistor', 'zipper', 'bottle', 'hazelnut', 'metal_nut', 'toothbrush']
    metrics = {'class': [], 'AUROC_sample': [], 'AUROC_pixel': [], 'AUPRO_pixel': []}

    print('Training with classes: ', mvtec_classes)
    for c in mvtec_classes:
        auroc_sp, auroc_px, aupro_px = train(c, pars)
        print('Best score of class: {}, Auroc sample: {:.4f}, Auroc pixel:{:.4f}, Pixel Aupro: {:.4f}'.format(c, auroc_sp,auroc_px,aupro_px))
        metrics['class'].append(c)
        metrics['AUROC_sample'].append(auroc_sp)
        metrics['AUROC_pixel'].append(auroc_px)
        metrics['AUPRO_pixel'].append(aupro_px)
        pd.DataFrame(metrics).to_csv(f'{pars.save_folder}/metrics_results.csv', index=False)