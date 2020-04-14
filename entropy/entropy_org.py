##----------------------------------------------------------
# written by Fei Pan
# to get the discriminator value for each target train images
#-----------------------------------------------------------
import sys
sys.path.append('/home/feipan/ws/ADVENT')
from tqdm import tqdm
import argparse
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
import pdb


best_iter = 70000
save_path = './masks_colors'
#------------------------------------- color -------------------------------------------
palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)
#------------------------------------- color -------------------------------------------

def get_arguments():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description="Code for evaluation")
    parser.add_argument('--best_iter', type=int,
                        default=None, help='iteration with best mIoU')
    parser.add_argument('')
    parser.add_argument('--cfg', type=str, 
                        default='./ADVENT/advent/scripts/configs/advent.yml',
                        help='optional config file' )
    return parser.parse_args()

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


def load_checkpoint_for_evaluation(model, checkpoint, device):
    saved_state_dict = torch.load(checkpoint)
    model.load_state_dict(saved_state_dict)
    model.eval()
    model.cuda(device)

def sort_list(list_file):
    origin_list_file = list_file
    sorted_list_file = sorted(origin_list_file, key=lambda img: img[1])
    sorted_list_file_copy = sorted_list_file.copy()

    for i, val in enumerate(sorted_list_file):
        sorted_list_file[i]="{0:70s}, {1:2.5f}".format(val[0], val[1])
        
    out = open("sorted_trg_images.txt","w+")
    for line in sorted_list_file:
        out.write(str(line))
        out.write('\n')
    out.close()

    return sorted_list_file_copy

def rank_masks(list_file):
    for i, val in enumerate(list_file):
        value, name = val[1], val[0].split('/')[-1]
        mask_color  = Image.open('./color_masks/%s_color.png' % (name.split('.')[0])) 
        image       = Image.open('../ADVENT/data/Cityscapes/leftImg8bit/train/'+val[0]).resize(mask_color.size)
        mask_color.save('./imgs_masks/%04d_%s_%.5f_mask.png' % (i, name.split('.')[0], value))
        image.save('./imgs_masks/%04d_%s_%.5f.png' % (i, name.split('.')[0], value))

def main(config_file):
   
    # LOAD ARGS
    device    = cfg.GPU_ID
    assert config_file is not None, 'Missing cfg file'
    cfg_from_file(config_file)

    print('Using config:')
    pprint.pprint(cfg)

    #--------------- deeplab_v2 ------------------------------------------------
    model_gen = get_deeplab_v2(num_classes=cfg.NUM_CLASSES,
                           multi_level=cfg.TEST.MULTI_LEVEL)
    restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0], f'model_{cfg.TEST.BEST_ITER}.pth')
    print("Evaluating model:", restore_from)
    load_checkpoint_for_evaluation(model_gen, restore_from, device)
    #--------------- discriminator ---------------------------------------------
    model_d_main = get_fc_discriminator(num_classes=cfg.NUM_CLASSES)
    model_d_main.train()
    model_d_main.to(device)
    restore_from = osp.join(cfg.TEST.SNAPSHOT_DIR[0], f'model_{cfg.TEST.BEST_ITER}_D_main.pth')
    print("Load Discriminator:", restore_from)
    load_checkpoint_for_evaluation(model_d_main, restore_from, device)
    #------------- interploate --------------------------------------------------
    input_size_target = cfg.TRAIN.INPUT_SIZE_TARGET
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear',
                                align_corners=True)
    pdb.set_trace()
    #---------------- dataloader ------------------------------------------------
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
    #------------------ eval -------------------------------------------------------
    # hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
    target_loader_iter = enumerate(target_loader)
    target_label = 1
    dis_list = []
    for index in tqdm(range(len(target_loader))):
        _, batch = target_loader_iter.__next__()
        image, _, _, name = batch
        with torch.no_grad():
            # pdb.set_trace()
            _, pred_trg_main = model_gen(image.cuda(device))
            pred_trg_main    = interp_target(pred_trg_main)
            # --------- get discriminator value ------------
            pred_trg_entropy = prob_2_entropy(F.softmax(pred_trg_main))
            # d_out_main       = d_main(pred_trg_entropy)
            # d_out_main_sig   = F.sigmoid(d_out_main)
            # loss_d_main      = bce_loss(d_out_main, target_label) / 2
            # pdb.set_trace()
            # d_out_main_value = F.sigmoid(d_out_main).mean()
            # dis_list.append((name[0], pred_trg_entropy.mean().item(), loss_d_main.mean().item()))
            dis_list.append((name[0], pred_trg_entropy.mean().item()))
            # --------- colorize and save the mask -----------
            colorize_save(pred_trg_main, name[0])
        # if index == 10:
        #     break
    dis_list = sort_list(dis_list)
    rank_masks(dis_list)

if __name__ == '__main__':
    args = get_arguments()
    print('Called with args:')
    print(args)
    main(args.cfg)
