import sys
import os
import torch
import time
import copy
import random
import argparse
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from tqdm import tqdm
from tensorboardX import SummaryWriter

from utils.losses import SpreadLoss, DiceLoss
from utils.metrics import get_accuracy, IOU2
from utils.helpers import update_ema
from utils.commons import init_seeds
from utils.utils_ours import weighted_mse_loss, dft_HighPass 
import torch.nn.functional as F
from utils import ramp_ups


def get_ip_data(data):
    return data['weak_data'].cuda(), data['strong_data'].cuda(), data['weak_mask'].cuda(), data['strong_mask'].cuda(), data['action'].cuda()


def data_concat(ip1, ip2, dims=0):
    return torch.cat([ip1, ip2], dim=dims)

def val_model_interface(minibatch):
    data = minibatch['weak_data'].cuda()
    action = minibatch['action'].cuda()
    label_mask = minibatch['weak_mask'].cuda()
    empty_vector = torch.zeros(action.shape[0]).cuda()

    st_loc_pred, predicted_action, _ = model(data, action, empty_vector, 0, 0)
    t_loc_pred, predicted_action_ema, _ = ema_model(data, action, empty_vector, 0, 0)

    class_loss, _ = criterion_cls(predicted_action, action)
    loss1 = criterion_loc_1(st_loc_pred, label_mask)
    loss2 = criterion_loc_2(st_loc_pred, label_mask)

    sup_loc_loss = loss1 + loss2
    total_loss = sup_loc_loss + class_loss
    return st_loc_pred, t_loc_pred, predicted_action, predicted_action_ema, label_mask, action, total_loss, sup_loc_loss, class_loss


def train_model_interface(args, label_minibatch, unlabel_minibatch, epoch, global_step, wt_ramp):
    # torch.float32 and weak_label_data.type(torch.cuda.FloatTensor) also equals torch.float32
    weak_label_data, strong_label_data, weak_label_mask, strong_label_mask, label_action = get_ip_data(label_minibatch)
    weak_unlabel_data, strong_unlabel_data, weak_unlabel_mask, strong_unlabel_mask, unlabel_action = get_ip_data(unlabel_minibatch)
    # # print(weak_label_data.shape, weak_unlabel_data.shape, strong_label_data.shape, strong_unlabel_data.shape)
    # print(weak_label_mask.shape, weak_unlabel_mask.shape, strong_label_mask.shape, strong_unlabel_mask.shape)

    # randomize
    concat_labels = torch.cat([torch.ones(len(label_action)), torch.zeros(len(unlabel_action))], dim=0).cuda()
    random_indices = torch.randperm(len(concat_labels))

    # # reshuffle original data
    concat_weak_data = data_concat(weak_label_data, weak_unlabel_data)[random_indices, :, :, :, :]
    concat_strong_data = data_concat(strong_label_data, strong_unlabel_data)[random_indices, :, :, :, :]
    concat_action = data_concat(label_action, unlabel_action)[random_indices]
    concat_weak_loc = data_concat(weak_label_mask, weak_unlabel_mask)[random_indices, :, :, :, :]
    concat_strong_loc = data_concat(strong_label_mask, strong_unlabel_mask)[random_indices, :, :, :, :]
    concat_labels = concat_labels[random_indices]
        
    # Labeled indexes
    labeled_vid_index = torch.where(concat_labels == 1)[0]
    
   # passing inputs to models
    # thresh_epoch = 11
    # STUDENT MODEL
    #random_noise = torch.rand(concat_strong_data.shape).cuda()
    #random_noise = (random_noise-0.5)/10
    #concat_strong_data = concat_strong_data+random_noise
    st_loc_pred, predicted_action_cls, st_action_feat = model(concat_strong_data, concat_action, concat_labels, epoch, args.thresh_epoch)


    # LOC LOSS SUPERVISED - STUDENT
    # labeled predictions
    labeled_st_pred_loc = st_loc_pred[labeled_vid_index]
    # labeled gt
    labeled_gt_loc = concat_strong_loc[labeled_vid_index]
    # calculate losses
    sup_loc_loss_1 = criterion_loc_1(labeled_st_pred_loc, labeled_gt_loc)
    sup_loc_loss_2 = criterion_loc_2(labeled_st_pred_loc, labeled_gt_loc)
    # print(sup_loc_loss_1, sup_loc_loss_2)

    # Classification loss SUPERVISED - STUDENT
    class_loss, _ = criterion_cls(predicted_action_cls[labeled_vid_index], concat_action[labeled_vid_index])

    # UPDATE EMA
    update_ema(model, ema_model, global_step, args.ema_val)

    # TEACHER MODEL
    with torch.no_grad():
        t_loc_pred, predicted_action_cls_ema, teacher_action_feat = ema_model(concat_weak_data, concat_action, concat_labels, epoch,
                                                         args.thresh_epoch)
    
    loc_cons_loss_1 = loc_const_criterion(st_loc_pred, t_loc_pred)
    
    batch_filter = dft_HighPass(F.sigmoid(t_loc_pred), radius = 4).cuda()
    batch_filter_noise = dft_HighPass(F.sigmoid(st_loc_pred), radius = 4).cuda()

    t_loc_pred = F.avg_pool3d(t_loc_pred, kernel_size=(3,3,3), stride=1, padding=1 )
    st_loc_pred = F.avg_pool3d(st_loc_pred, kernel_size=(3,3,3), stride=1, padding=1 )

    loss_wt_var_1 = weighted_mse_loss(st_loc_pred, t_loc_pred, batch_filter)
    loss_wt_var_2 = weighted_mse_loss(st_loc_pred, t_loc_pred, batch_filter_noise)
        
    total_cons_loss = (wt_ramp * (loss_wt_var_1 + loss_wt_var_2)) + ((1 - wt_ramp) * loc_cons_loss_1)
    
    sup_loc_loss = sup_loc_loss_1 + sup_loc_loss_2
    total_loss = args.wt_loc * sup_loc_loss + args.wt_cls * class_loss + args.wt_cons * total_cons_loss

    return st_loc_pred, predicted_action_cls, predicted_action_cls_ema, concat_weak_loc, concat_action, total_loss, sup_loc_loss, class_loss, total_cons_loss


def train(args, model, ema_model, labeled_train_loader, unlabeled_train_loader, optimizer, epoch, save_path, writer,
         global_step, ramp_wt):
    start_time = time.time()
    steps = len(unlabeled_train_loader)
    model.train(mode=True)
    model.training = True
    ema_model.train(mode=True)
    ema_model.training = True
    total_loss = []
    accuracy = []
    acc_ema = []
    sup_loc_loss = []
    class_loss = []
    loc_consistency_loss = []

    start_time = time.time()


    labeled_iterloader = iter(labeled_train_loader)

    for batch_id, unlabel_minibatch in enumerate(unlabeled_train_loader):

        global_step += 1

        # u dnt place it between loss.backward and optimizer.step
        # but can place it anywhere else
        optimizer.zero_grad()

        try:
            label_minibatch = next(labeled_iterloader)

        except StopIteration:
            labeled_iterloader = iter(labeled_train_loader)
            label_minibatch = next(labeled_iterloader)

        _, predicted_action, predicted_action_ema, _, action, loss, s_loss, c_loss, cc_loss = train_model_interface(
            args, label_minibatch, unlabel_minibatch, epoch, global_step, ramp_wt(epoch))

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())
        sup_loc_loss.append(s_loss.item())
        class_loss.append(c_loss.item())
        loc_consistency_loss.append(cc_loss.item())
        accuracy.append(get_accuracy(predicted_action, action))
        acc_ema.append(get_accuracy(predicted_action_ema, action))

        if (batch_id + 1) % args.pf == 0:
            r_total = np.array(total_loss).mean()
            r_loc = np.array(sup_loc_loss).mean()
            r_class = np.array(class_loss).mean()
            r_cc_class = np.array(loc_consistency_loss).mean()
            r_acc = np.array(accuracy).mean()
            r_acc_ema = np.array(acc_ema).mean()
            
            print(f'[TRAIN] epoch-{epoch:0{len(str(args.epochs))}}/{args.epochs},'
                  f'batch-{batch_id + 1:0{len(str(steps))}}/{steps}'
                  f'\t [LOSS ] loss-{r_total:.3f}, cls-{r_class:.3f}, loc-{r_loc:.3f}, const-{r_cc_class:.3f}'
                  f'\t [ACC] ST-{r_acc:.3f}, T-{r_acc_ema:.3f}')

            # summary writing
            total_step = (epoch - 1) * len(unlabeled_train_loader) + batch_id + 1
            info_loss = {
                'loss': r_total,
                'loss_loc': r_loc,
                'loss_cls': r_class,
                'loss_consistency': r_cc_class
            }
            info_acc = {
                'acc': r_acc,
                'acc_ema': r_acc_ema
            }
            writer.add_scalars('train/loss', info_loss, total_step)
            writer.add_scalars('train/acc', info_acc, total_step)
            sys.stdout.flush()

    end_time = time.time()
    print("Training time: ", end_time - start_time)

    train_total_loss = np.array(total_loss).mean()

    return global_step, train_total_loss


def validate(model, ema_model, val_data_loader, epoch):
    steps = len(val_data_loader)
    model.eval()
    model.training = False

    ema_model.eval()
    ema_model.training = False
    total_loss = []
    accuracy = []
    acc_ema = []
    sup_loc_loss = []
    class_loss = []
    total_IOU_s = 0
    validiou_s = 0

    total_IOU_t = 0
    validiou_t = 0
    print('\nVALIDATION STARTED...')
    start_time = time.time()

    with torch.no_grad():

        for _, minibatch in enumerate(val_data_loader):

            st_loc_pred, t_loc_pred, predicted_action, predicted_action_ema, gt_loc_map, action, loss, s_loss, c_loss = val_model_interface(minibatch)
            total_loss.append(loss.item())
            sup_loc_loss.append(s_loss.item())
            class_loss.append(c_loss.item())
            accuracy.append(get_accuracy(predicted_action, action))
            acc_ema.append(get_accuracy(predicted_action_ema, action))

            # STUDENT
            maskout_s = st_loc_pred.cpu().data.numpy()
            # TEACHER
            maskout_t = t_loc_pred.cpu().data.numpy()
            # utils.show(maskout_s[0])

            # use threshold to make mask binary
            maskout_s[maskout_s > 0] = 1
            maskout_s[maskout_s < 1] = 0

            maskout_t[maskout_t > 0] = 1
            maskout_t[maskout_t < 1] = 0
            # utils.show(maskout_s[0])

            truth_np = gt_loc_map.cpu().data.numpy()
            for a in range(minibatch['weak_data'].shape[0]):
                iou_s = IOU2(truth_np[a], maskout_s[a])
                iou_t = IOU2(truth_np[a], maskout_t[a])
                if iou_s == iou_s:
                    total_IOU_s += iou_s
                    validiou_s += 1

                if iou_t == iou_t:
                    total_IOU_t += iou_t
                    validiou_t += 1

    val_epoch_time = time.time() - start_time
    print("Validation time: ", val_epoch_time)

    r_total = np.array(total_loss).mean()
    r_loc = np.array(sup_loc_loss).mean()
    r_class = np.array(class_loss).mean()
    r_acc = np.array(accuracy).mean()
    r_acc_ema = np.array(acc_ema).mean()
    average_IOU_s = total_IOU_s / validiou_s
    average_IOU_t = total_IOU_t / validiou_t

    print(f'[VAL] EPOCH-{epoch:0{len(str(args.epochs))}}/{args.epochs}'
          f'\t [LOSS] loss-{r_total:.3f}, cls-{r_class:.3f}, loc-{r_loc:.3f}'
          f'\t [ACC] ST-{r_acc:.3f}, T-{r_acc_ema:.3f}' 
          f'\t [IOU ] ST-{average_IOU_s:.3f}, T-{average_IOU_t:.3f}')
    sys.stdout.flush()
    return r_total


def parse_args():
    parser = argparse.ArgumentParser(description='loc_const')
    parser.add_argument('--bs', type=int, default=8, help='mini-batch size')
    parser.add_argument('--pf', type=int, default=100, help='print frequency every batch')
    parser.add_argument('--epochs', type=int, default=1, help='number of total epochs to run')
    parser.add_argument('--model_name', type=str, default='i3d', help='model name')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--sup_loc_loss', type=str, default='dice', help='dice or iou loss')
    parser.add_argument('--exp_id', type=str, default='debug', help='experiment name')

    parser.add_argument('--pkl_file_label', type=str, default='train_annots_10_labeled_random.pkl', help='label subset')
    parser.add_argument('--pkl_file_unlabel', type=str, default='train_annots_90_unlabeled_random.pkl',
                        help='unlabele subset')
    parser.add_argument('--const_loss', type=str, default='l2', help='consistency loss type')
    parser.add_argument('--wt_loc', type=float, default=1, help='loc loss weight')
    parser.add_argument('--wt_cls', type=float, default=1, help='Classification loss weight')
    parser.add_argument('--wt_cons', type=float, default=0.1, help='class consistency loss weight')

    parser.add_argument('--aug', action='store_true', help='use augmentation for unlabeled dataset or not')
    parser.add_argument('-at', '--aug_type', type=int, help="0-spatial, 1- temporal, 2 - both")
    parser.add_argument('-ema', '--ema_val', type=float, help="0.5-0.99")

    parser.add_argument('-d', '--dataset', default="ucf", type=str, metavar='TYPE',
                    choices=['ucf', 'jhmdb'],
                    help='dataset to use')
    # parser.add_argument('--masking', action="store_true", help="use masking")
    # parser.add_argument('--mask_ratio', type=int, default=11, help='mask ratio')

    parser.add_argument('--all_actions', action='store_true', help='use rest 77 classes')
    parser.add_argument('--thresh_epoch', type=int, default=11, help='thresh epoch to introduce pseudo labels')
    parser.add_argument('--sig_map', action='store_true', help='use sigmoid probab maps')

    # Burn-in params
    parser.add_argument('-burn', '--burn_in', action='store_true', help='use burn in weights')
    parser.add_argument('-bw', '--burn_wts', type=str, default='debug', help='experiment name')

    parser.add_argument('--pretrain', action='store_true', help='use pretrained wts or not')

    # define seed params
    parser.add_argument('--seed', type=int, default=47, help='seed for initializing training.')
    parser.add_argument('--seed_data', type=int, default=37, help='seed variation pickle files')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print(vars(args))
    
    init_seeds(args.seed)

    USE_CUDA = True if torch.cuda.is_available() else False
    if torch.cuda.is_available() and not USE_CUDA:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # HYPERPARAMS
    TRAIN_BATCH_SIZE = args.bs
    VAL_BATCH_SIZE = args.bs
    N_EPOCHS = args.epochs
    LR = args.lr

    # LOAD DATASET
    from datasets.ucf_dataloader_st_augs_v1_speedup import UCF101DataLoader, collate_fn_train, collate_fn_test
            
    labeled_trainset = UCF101DataLoader('train', [224, 224], cl=8, file_id=args.pkl_file_label, 
                                        aug_mode=args.aug_type, subset_seed=args.seed_data)
    unlabeled_trainset = UCF101DataLoader('train', [224, 224], cl=8, file_id=args.pkl_file_unlabel,
                                        aug_mode=args.aug_type, subset_seed=args.seed_data)
    validationset = UCF101DataLoader('test',[224, 224], cl=8, file_id='testlist.txt',
                                        aug_mode=0, subset_seed=args.seed_data)

    print(len(labeled_trainset), len(unlabeled_trainset), len(validationset))
    
    labeled_train_data_loader = DataLoader(
        dataset=labeled_trainset,
        batch_size=(TRAIN_BATCH_SIZE) // 2,
        num_workers=8,
        shuffle=True, 
        collate_fn=collate_fn_train

    )

    unlabeled_train_data_loader = DataLoader(
        dataset=unlabeled_trainset,
        batch_size=(TRAIN_BATCH_SIZE) // 2,
        num_workers=8,
        shuffle=True,
        collate_fn=collate_fn_train
    )

    val_data_loader = DataLoader(
        dataset=validationset,
        batch_size=VAL_BATCH_SIZE,
        num_workers=8,
        shuffle=False,
        collate_fn=collate_fn_test
    )

    print(len(labeled_train_data_loader), len(unlabeled_train_data_loader), len(val_data_loader))

    from models.capsules_ucf101_semi_sup_pa import CapsNet

    # Load pretrained weights
    model = CapsNet()
    if args.burn_in:
        print("weights loaded")
        model.load_previous_weights(args.burn_wts)
    # model.load_previous_weights('main_weights/15_active.pth')

    

    if USE_CUDA:
        model = model.cuda()
    ema_model = copy.deepcopy(model)

    # losses
    global criterion_cls
    global criterion_loc_1
    global criterion_loc_2
    global loc_const_criterion
    # global global_step
    global_step = 0

    criterion_cls = SpreadLoss(num_class=24, m_min=0.2, m_max=0.9)
    criterion_loc_1 = nn.BCEWithLogitsLoss(size_average=True)
    criterion_loc_2 = DiceLoss()

    if args.const_loss == 'jsd':
        loc_const_criterion = torch.nn.KLDivLoss(size_average=False, reduce=False).cuda()

    elif args.const_loss == 'l2':
        loc_const_criterion = nn.MSELoss()

    elif args.const_loss == 'l1':
        loc_const_criterion = nn.L1Loss()

    assert (args.const_loss == 'l2') or (args.const_loss == 'l1')
    print("Loc consistency criterion: ", loc_const_criterion)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0, eps=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', min_lr=1e-7, patience=5, factor=0.1,
                                                     verbose=True)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 45], gamma=0.1, verbose=True)
    ramp_wt = ramp_ups.sigmoid_rampup(N_EPOCHS)

    exp_id = args.exp_id
    save_path = os.path.join('./train_log_wts', exp_id)
    model_save_dir = os.path.join(save_path, time.strftime('%m-%d-%H-%M'))
    writer = SummaryWriter(model_save_dir)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    print(f"Save at: {model_save_dir}")

    prev_best_train_loss = 10000
    prev_best_train_loss_model_path = None
    gs = 0

    for e in tqdm(range(1, N_EPOCHS + 1), total=N_EPOCHS, desc="Epochs"):
        gs, train_loss = train(args, model, ema_model, labeled_train_data_loader, unlabeled_train_data_loader,
                                  optimizer, e, save_path, writer, global_step, ramp_wt)
        global_step = gs
        if e > (N_EPOCHS-10) and e%2==0:
            val_loss = validate(model, ema_model, val_data_loader, e)
            #torch.save(model.state_dict(),'exp_weights/exp2/'+str(e)+'.pth')

        if train_loss < prev_best_train_loss:
            print("Yay!!! Got the train loss down...")
            train_model_path = os.path.join(model_save_dir, f'best_model_train_loss_{e}.pth')
            torch.save(model.state_dict(), train_model_path)
            prev_best_train_loss = train_loss
            if prev_best_train_loss_model_path and e<25:
                os.remove(prev_best_train_loss_model_path)
            prev_best_train_loss_model_path = train_model_path
            print(f"Saved at {train_model_path}")
            #scheduler.step(train_loss)


