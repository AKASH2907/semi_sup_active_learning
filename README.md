# semi_sup_active_learning
Official repo for AAAI-24 Paper - Semi-Supervised Active Learning for Video Action Detection

The setup can be done using anaconda3 on Ubuntu (>18) using the provided spec-file
```
    $ conda create --name <env> --file spec-file.txt
```

The dataset is expected to be in 'Datasets/UCF101' (as downloaded from the main dataset website)

Pretrained charades weights: https://github.com/piergiaj/pytorch-i3d/tree/master/models


```````````````````

Training

Run using the below commands for model training and active learning.

1. For model training 

    python semi_loc_feat_const_pa_stn_aug_add_ayush_save_weights.py --epochs 50 --bs 8 --lr 1e-4\
     --pkl_file_label train_annots_10_labeled.pkl\
     --pkl_file_unlabel train_annots_90_unlabeled.pkl\
     --wt_loc 1 --wt_cls 1 --wt_cons 0.1\
     --const_loss l2 --thresh_epoch 11 -at 2 -ema 0.99

 
2. For AL selection after having a model trained 

    python active_learning_uncertainty.py 10 15

    
3. Next model training round after AL selection 

    python semi_loc_feat_const_pa_stn_aug_add_ayush_save_weights.py --epochs 50 --bs 8 --lr 1e-4\
     --pkl_file_label train_annots_15_labeled.pkl\
     --pkl_file_unlabel train_annots_85_unlabeled.pkl\
     --wt_loc 1 --wt_cls 1 --wt_cons 0.1\
     --const_loss l2 --thresh_epoch 11 -at 2 -ema 0.99 \
     --burn_in --burn_wts main_weights/10_uncer.pth
