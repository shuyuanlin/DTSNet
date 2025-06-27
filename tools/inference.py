import torch
import numpy as np
import random
import os
import pandas as pd
from argparse import ArgumentParser
from model.resnet import wide_resnet50_2
from model.de_resnet import de_wide_resnet50_2
from tools.eval_helper import evaluation
from model.apr import APR
from dataset.dataset import MVTecDataset_test, VisADataset_test, BTADDataset_test

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
    parser.add_argument('--checkpoint_folder', default = './experiments/MVTec-AD', type=str)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--classes', nargs="+", default=["pill"])
    parser.add_argument('--dataset_name', nargs="+", default='MVTec-AD', type=str, help='VisA, BTAD')
    parser.add_argument('--dataset_path', default='/home/sylin/LTX/datasets', type=str)
    pars = parser.parse_args()
    return pars

def inference(_class_, pars):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_path = pars.dataset_path + '/' + pars.dataset_name + "/" + _class_
    test_data = MVTecDataset_test(dataset_path=test_path, resize=(pars.image_size, pars.image_size))
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = torch.nn.DataParallel(encoder).to(device)
    bn = torch.nn.DataParallel(bn).to(device)

    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = torch.nn.DataParallel(decoder).to(device)

    APR_module = APR(base=64, temp=0.5, sigma=0.1)
    APR_module = torch.nn.DataParallel(APR_module).to(device)

    checkpoint_class  = pars.checkpoint_folder + '/' + _class_ + '/' + 'wres50_'+_class_+'.pth'
    ckp = torch.load(checkpoint_class, map_location='cpu')
    bn.load_state_dict(ckp['bn'])
    decoder.load_state_dict(ckp['decoder'])
    APR_module.load_state_dict(ckp['rec'])
  
    auroc_px, auroc_sp, aupro_px = evaluation(encoder, bn, decoder, test_dataloader, device, APR_module)
    print('{}: Sample Auroc: {:.4f}, Pixel Auroc:{:.4f}, Pixel Aupro: {:.4f}'.format(_class_, auroc_sp, auroc_px, aupro_px))
    return auroc_sp, auroc_px, aupro_px

if __name__ == '__main__':
    pars = get_args()

    item_list = ['carpet','grid','leather','tile','wood','bottle','cable','capsule','hazelnut','metal_nut','pill','screw','toothbrush','transistor','zipper']
    setup_seed(111)
    metrics = {'class': [], 'AUROC_sample':[], 'AUROC_pixel': [], 'AUPRO_pixel': []}
    
    for c in item_list:
        auroc_sp, auroc_px, aupro_px = inference(c, pars)
        metrics['class'].append(c)
        metrics['AUROC_sample'].append(auroc_sp)
        metrics['AUROC_pixel'].append(auroc_px)
        metrics['AUPRO_pixel'].append(aupro_px)
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(f'{pars.checkpoint_folder}/metrics_checkpoints.csv', index=False)

    metrics['class'].append("average")
    metrics['AUROC_sample'].append(np.mean(metrics['AUROC_sample']))
    metrics['AUROC_pixel'].append(np.mean(metrics['AUROC_pixel']))
    metrics['AUPRO_pixel'].append(np.mean(metrics['AUPRO_pixel']))
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f'{pars.save_folder}/metrics_checkpoints.csv', index=False)

    print("------------Average------------")
    print('Image Auroc: {:.4f}, Pixel Auroc:{:.4f}, Pixel Aupro: {:.4f}'.format(
        np.mean(metrics['AUROC_sample']), np.mean(metrics['AUROC_pixel']), np.mean(metrics['AUPRO_pixel']) )
    )