import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from model.resnet import wide_resnet50_2
from model.de_resnet import de_wide_resnet50_2
from model.apr import APR
from dataset.dataset import MVTecDataset_test, VisADataset_test, BTADDataset_test
from tools.eval_helper import cal_anomaly_map, gaussian_filter, min_max_norm, cvt2heatmap, show_cam_on_image

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
    parser.add_argument('--save_folder', default = './experiments/MVTec-AD/visualization', type=str)
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--classes', nargs="+", default=["pill"])
    parser.add_argument('--dataset_name', nargs="+", default='MVTec-AD', type=str, help='VisA, BTAD')
    parser.add_argument('--dataset_path', default='/home/sylin/LTX/datasets', type=str)
    pars = parser.parse_args()
    return pars


def vis(_class_, pars):
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

    checkpoint_class = pars.checkpoint_folder + '/' + _class_ + '/' + 'wres50_' + _class_ + '.pth'
    ckp = torch.load(checkpoint_class, map_location='cpu')
    bn.load_state_dict(ckp['bn'])
    decoder.load_state_dict(ckp['decoder'])
    APR_module.load_state_dict(ckp['rec'])

    encoder.eval()
    APR_module.eval()
    bn.eval()
    decoder.eval()

    count = 0
    save_vis_path = pars.save_folder + '/' + _class_
    if not os.path.exists(save_vis_path):
        os.makedirs(save_vis_path)

    with torch.no_grad():
        for (img, gt, label, _, _) in test_dataloader:
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            outputs = APR_module(outputs, inputs)  # rec

            anomaly_map, _ = cal_anomaly_map(inputs, outputs, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            ano_map = min_max_norm(anomaly_map)
            ano_map = cvt2heatmap(ano_map * 255)

            img = img.permute(0, 2, 3, 1).cpu().numpy()[0]
            img = np.uint8(min_max_norm(img) * 255)

            ano_map = show_cam_on_image(img, ano_map)
            gt = np.array(gt.squeeze())

            fig, ax = plt.subplots(1, 3, figsize=(7, 7))
            plt.axis('off')
            ax[0].imshow(img)
            ax[0].set_title('Image')
            ax[0].axis('off')
            ax[1].imshow(gt, cmap='gray')
            ax[1].set_title('Ground True')
            ax[1].axis('off')
            ax[2].imshow(ano_map)
            ax[2].set_title('Anomaly Map')
            ax[2].axis('off')
            # plt.show()
            # exit()
            plt.savefig(save_vis_path + "/" + str(count) + ".png")

            count += 1

if __name__ == '__main__':
    pars = get_args()

    item_list = ['carpet', 'grid', 'leather', 'tile', 'wood', 'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill', 'screw', 'toothbrush', 'transistor', 'zipper']
    setup_seed(111)
    for c in item_list:
        vis(c, pars)
        print('{}类别可视化完毕'.format(c))
