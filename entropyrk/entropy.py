##----------------------------------------------------------
# written by Fei Pan
#
# to get the entropy ranking from Inter-domain adaptation process 
#-----------------------------------------------------------
import sys
from tqdm import tqdm
import argparse
import os
import os.path as osp
import pprint
import torch
import numpy as np
from PIL import Image
from torch import nn
from torch.utils import data
from advent.model.deeplabv2 import get_deeplab_v2
from advent.model.discriminator import get_fc_discriminator
from advent.dataset.cityscapes import CityscapesDataSet
from advent.utils.func import prob_2_entropy
import torch.nn.functional as F
from advent.utils.func import loss_calc, bce_loss
from advent.domain_adaptation.config import cfg, cfg_from_file
from matplotlib import pyplot as plt
from matplotlib import image  as mpimg



#------------------------------------- color -------------------------------------------
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)
# The rare classes trainID from cityscapes dataset
# These classes are: 
#    wall, fence, pole, traffic light, trafflic sign, terrain, rider, truck, bus, train, motor.
rare_class = [3, 4, 5, 6, 7, 9, 12, 14, 15, 16, 17]


def colorize(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    
    return new_mask    

def colorize_save(output_pt_tensor, name):
    output_np_tensor = output_pt_tensor.cpu().data[0].numpy()
    mask_np_tensor   = output_np_tensor.transpose(1,2,0) 
    mask_np_tensor   = np.asarray(np.argmax(mask_np_tensor, axis=2), dtype=np.uint8)
    mask_Img         = Image.fromarray(mask_np_tensor)
    mask_color       = colorize(mask_np_tensor)  

    name = name.split('/')[-1]
    mask_Img.save('./color_masks/%s' % (name))
    mask_color.save('./color_masks/%s_color.png' % (name.split('.')[0]))

def find_rare_class(output_pt_tensor):
    output_np_tensor = output_pt_tensor.cpu().data[0].numpy()
    mask_np_tensor   = output_np_tensor.transpose(1,2,0)
    mask_np_tensor   = np.asarray(np.argmax(mask_np_tensor, axis=2), dtype=np.uint8)
    mask_np_tensor   = np.reshape(mask_np_tensor, 512*1024)
    unique_class     = np.unique(mask_np_tensor).tolist()
    commom_class     = set(unique_class).intersection(rare_class)
    return commom_class


def cluster_subdomain(entropy_list, lambda1):
    entropy_list = sorted(entropy_list, key=lambda img: img[1])
    copy_list = entropy_list.copy()
    entropy_rank = [item[0] for item in entropy_list]

    easy_split = entropy_rank[ : int(len(entropy_rank) * lambda1)]
    hard_split = entropy_rank[int(len(entropy_rank)* lambda1): ]

    with open('easy_split.txt','w+') as f:
        for item in easy_split:
            f.write('%s\n' % item)

    with open('hard_split.txt','w+') as f:
        for item in hard_split:
            f.write('%s\n' % item)

    return copy_list

def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda(device)

def get_arguments():
    """
    Parse input arguments 
    """
    parser = argparse.ArgumentParser(description="Code for evaluation")

    parser.add_argument('--best_iter', type=int, default=70000,
                        help='iteration with best mIoU')
    parser.add_argument('--normalize', type=bool, default=False,
                        help='add normalizor to the entropy ranking')
    parser.add_argument('--lambda1', type=float, default=0.67, 
                        help='hyperparameter lambda to split the target domain')
    parser.add_argument('--cfg', type=str, default='../ADVENT/advent/scripts/configs/advent.yml',
                        help='optional config file' )
    return parser.parse_args()

def main(args):

    # load configuration file 
    device = cfg.GPU_ID    
    assert args.cfg is not None, 'Missing cfg file'
    cfg_from_file(args.cfg)

    if not os.path.exists('./color_masks'):
        os.mkdir('./color_masks')

    cfg.EXP_NAME = f'{cfg.SOURCE}2{cfg.TARGET}_{cfg.TRAIN.MODEL}_{cfg.TRAIN.DA_METHOD}'
    cfg.TEST.SNAPSHOT_DIR[0] = osp.join(cfg.EXP_ROOT_SNAPSHOT, cfg.EXP_NAME)

    # load model with parameters trained from Inter-domain adaptation
    model_gen = get_deeplab_v2(num_classes=cfg.NUM_CLASSES, multi_level=cfg.TEST.MULTI_LEVEL)
    
    restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0], f'model_{args.best_iter}.pth')
    
    print("Loading the generator:", restore_from)
    
    load_checkpoint_for_evaluation(model_gen, restore_from, device)
    
    # load data
    target_dataset = CityscapesDataSet(root=cfg.DATA_DIRECTORY_TARGET,
                                       list_path=cfg.DATA_LIST_TARGET,
                                       set=cfg.TRAIN.SET_TARGET,
                                       info_path=cfg.TRAIN.INFO_TARGET,
                                       max_iters=None,
                                       crop_size=cfg.TRAIN.INPUT_SIZE_TARGET,
                                       mean=cfg.TRAIN.IMG_MEAN)
    
    target_loader = data.DataLoader(target_dataset,
                                    batch_size=cfg.TRAIN.BATCH_SIZE_TARGET,
                                    num_workers=cfg.NUM_WORKERS,
                                    shuffle=True,
                                    pin_memory=True,
                                    worker_init_fn=None)

    target_loader_iter = enumerate(target_loader)

    # upsampling layer
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)

    entropy_list = []
    for index in tqdm(range(len(target_loader))):
        _, batch = target_loader_iter.__next__()
        image, _, _, name = batch
        with torch.no_grad():
            _, pred_trg_main = model_gen(image.cuda(device))
            pred_trg_main    = interp_target(pred_trg_main)
            if args.normalize == True:
                normalizor = (11-len(find_rare_class(pred_trg_main))) / 11.0 + 0.5
            else:
                normalizor = 1
            pred_trg_entropy = prob_2_entropy(F.softmax(pred_trg_main))
            entropy_list.append((name[0], pred_trg_entropy.mean().item() * normalizor))
            colorize_save(pred_trg_main, name[0])

    # split the enntropy_list into 
    cluster_subdomain(entropy_list, args.lambda1)

if __name__ == '__main__':
    args = get_arguments()
    print('Called with args:')
    main(args)
