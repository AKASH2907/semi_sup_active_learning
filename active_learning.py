import os
import time
import numpy as np
import random
from threading import Thread
from scipy.io import loadmat, savemat
from skvideo.io import vread
import pdb
import sys 

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.utils import save_image
from models.capsules_ucf101_semi_sup_pa import CapsNet
import torch.nn.functional as F
import torch.nn.functional as F_pool
from torch.utils.data import Dataset
import time
from utils_ours import *

#import keras
#from keras import Model
#from keras.utils import print_summary
#from keras.models import load_model
#from cust_losses import *
import cv2
import pickle
from scipy.stats import norm
import cv2

'''

Loads in videos for the 24 class subset of UCF-101.

The data is assumed to be organized in a folder (dataset_dir):
-Subfolder UCF101_vids contains the videos
-Subfolder UCF101_Annotations contains the .mat annotation files

UCF101DataLoader loads in the videos and formats their annotations on seperate threads.
-Argument train_or_test determines if you want to load in training or testing data
-Argument sec_to_wait determines the amount of time to wait to load in data
-Argument n_threads determines the number of threads which will be created to load data

Calling .get_video() returns (video, bboxes, label)
-The videos are in the shape (F, H, W, 3).
-The bounding boxes are loaded in as heat maps of size (F, H, W, 1) where 1 is forground and 0 is background.
-The label is an integer corresponding to the action class.

'''




class UCF101DataLoader(Dataset):
    'Generates UCF101-24 data'
    def __init__(self, name, clip_shape, batch_size, use_random_start_frame=False):
      self._dataset_dir = 'Datasets/UCF101'      
      
      if name == 'train':
          self.vid_files = self.get_det_annots_prepared()
          #self.shuffle = True
          print("TRAINING EVAL MODE !!!!")
      else:
          print("Should not run in test mode!")
          exit()
          #self.vid_files = self.get_det_annots_prepared()
          #self.shuffle = False

      self._height = clip_shape[0]
      self._width = clip_shape[1]
      #self._channels = channels
      self._batch_size = batch_size
      self._size = len(self.vid_files)
      self.indexes = np.arange(self._size)
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.vid_files)
        self.indexes = np.arange(self._size)
        np.random.shuffle(self.indexes)
        

    def get_det_annotations(self):
        # f = loadmat(dataset_dir + 'UCF101_Annotations/trainAnnot.mat')
        # f2 = loadmat(dataset_dir + 'UCF101_Annotations/testAnnot.mat')
        f = loadmat(self._dataset_dir + '/trainAnnot.mat')
        f2 = loadmat(self._dataset_dir + '/testAnnot.mat')

        training_annotations = []
        for ann in f['annot'][0]:
            file_name = ann[1][0]

            sp_annotations = ann[2][0]
            annotations = []
            for sp_ann in sp_annotations:
                ef = sp_ann[0][0][0] - 1
                sf = sp_ann[1][0][0] - 1
                label = sp_ann[2][0][0] - 1
                bboxes = (sp_ann[3]).astype(np.int32)
                annotations.append((sf, ef, label, bboxes))
            training_annotations.append((file_name, annotations))

        testing_annotations = []
        for ann in f2['annot'][0]:
            file_name = ann[1][0]

            sp_annotations = ann[2][0]
            annotations = []
            for sp_ann in sp_annotations:
                ef = sp_ann[0][0][0] - 1
                sf = sp_ann[1][0][0] - 1
                label = sp_ann[2][0][0] - 1
                bboxes = (sp_ann[3]).astype(np.int32)
                annotations.append((sf, ef, label, bboxes))

            testing_annotations.append((file_name, annotations))
        return training_annotations, testing_annotations


    def get_det_annots_prepared(self):
        import pickle     
        training_annot_file = 'training_annots_percentOfVids_100perClass.pkl'
        with open(training_annot_file, 'rb') as tr_rid:
            training_annotations = pickle.load(tr_rid)
        print("Training samples from :", training_annot_file)
        return training_annotations
            
    
    def get_det_annots_test_prepared(self):
        test_annot_file = 'testlist_JHMDB.txt'
        with open(test_annot_file,"r") as rid:
            test_file_list = rid.readlines()
        
        for i in range(len(test_file_list)):
            test_file_list[i] = test_file_list[i].rstrip()
            
        print("Testing samples from :", test_annot_file)
        
        return test_file_list    
            
    def __len__(self):
        'Denotes the number of videos per epoch'
        return int(self._size)


    def get_item(self, index):
        v_name, anns = self.vid_files[index]
        clip, bbox_clip, label, annots = self.load_video(v_name, anns)
        if clip is None:
            print("Video none ", v_name)
            return None, None, None, None, None, None
        
        # Center crop
        frames, h, w, _ = clip.shape        
        margin_h = h - self._height
        h_crop_start = int(margin_h/2)
        margin_w = w - self._width
        w_crop_start = int(margin_w/2)
        
        clip = clip[:, h_crop_start:h_crop_start+self._height, w_crop_start:w_crop_start+self._width, :] / 255.
        bbox_clip = bbox_clip[:, h_crop_start:h_crop_start+self._height, w_crop_start:w_crop_start+self._width, :]
        
        print("Video loaded: ", v_name, " With frames: ", clip.shape[0])
        return v_name, anns, clip, bbox_clip, label, annots 
        

    def load_video(self, video_name, annotations):
        video_dir = os.path.join(self._dataset_dir, 'UCF101_Videos/%s.avi' % video_name)
        # print(video_dir)
        # print(type(video_dir))
        try:
            # print(str(video_dir))
            video = vread(str(video_dir)) # Reads in the video into shape (F, H, W, 3)
            # print(video.shape)
        except:
            # video = vread(str(video_dir), num_frames=40)
            # print('Error:', str(video_dir))
            return None, None, None, None
            # print(video.shape)

        # video = vread(str(video_dir)) # Reads in the video into shape (F, H, W, 3)
        #if video.shape[0] < 40:
            #print(str(video_dir))
            #print(video.shape[0])

        # creates the bounding box annotation at each frame
        n_frames, h, w, ch = video.shape
        bbox = np.zeros((n_frames, h, w, 1), dtype=np.uint8)
        label = -1
        
        multi_frame_annot = []
        for ann in annotations:
            start_frame, end_frame, label = ann[0], ann[1], ann[2]      # Label is from 0 in annotations
            multi_frame_annot.extend(ann[4])
            for f in range(start_frame, min(n_frames, end_frame+1)):
                try:
                    x, y, w, h = ann[3][f-start_frame]
                    bbox[f, y:y+h, x:x+w, :] = 1
                except:
                    print('ERROR LOADING ANNOTATIONS')
                    print(start_frame, end_frame)
                    print(video_dir)
                    exit()
        multi_frame_annot = list(set(multi_frame_annot))
        
        return video, bbox, label, multi_frame_annot


def get_thresholded_arr(arr, threshold = 0.3):
    # b x 1 x h x w x 1  (FG)
    # b x 1 x h x w x 25 (CLS)
    
    if arr.shape[-1] > 1:
        arr_max = (arr == np.max(arr,-1,keepdims=True)).astype(float)
        arr *= arr_max
        arr[arr>0] = 1. 
    else:
        arr[arr<=threshold] = 1e-8  
    return arr


def get_thresholded_arr_bool(arr, threshold = 0.3):

    if arr.shape[-1] > 1:
        arr_max = (arr == np.max(arr,-1,keepdims=True)).astype(float)
        arr *= arr_max
        arr[arr>0] = 1.
    else:
        arr[arr>threshold] = 1
        arr[arr<=threshold] = 0
    return arr

def get_uncertainty_logx(frame):
    
    frame_th = get_thresholded_arr_bool(frame, threshold = 0.3)
    frame_val = get_thresholded_arr(frame, threshold = 0.3)
    if frame_th.sum() == 0:
        return 0.0
    #frame_th = frame_th.astype(np.bool)
    #frame[frame_th == 0] = 1e-8
    frame_val = -np.log(frame_val)
    uncertainty = frame_val.sum() / frame_th.sum()
    #print('sum ; ', frame_th.sum())
    #print('uncer : ', uncertainty)
    #print('frame the shape ', frame_th.shape)
    #print('frame shape ', frame.shape)
    #print('frmae with the shape : ', frame[frame_th].shape)
    #print('frame sum : ', frame_th)
    #print('frame with th', frame[frame_th])
    return uncertainty




def get_det_annots_prepared_old(num):
    import pickle     
    training_annot_file = 'data_lists/train_annots_'+str(num)+'_labeled.pkl'
    #training_annot_file = 'training_annots_percentOfVids_10perClass.pkl'
    with open(training_annot_file, 'rb') as tr_rid:
        training_annotations = pickle.load(tr_rid)
    print("Training samples from :", training_annot_file)
    return training_annotations

def get_det_annots_prepared_old_unlabeled(num):
    import pickle     
    training_annot_file = 'data_lists/train_annots_'+str(num)+'_unlabeled.pkl'
    #training_annot_file = 'training_annots_percentOfVids_10perClass.pkl'
    with open(training_annot_file, 'rb') as tr_rid:
        training_annotations = pickle.load(tr_rid)
    print("Training samples from :", training_annot_file)
    return training_annotations

def get_det_annots_prepared_complete():
        import pickle
        training_annot_file = 'training_annots_percentOfVids_100perClass.pkl'
        with open(training_annot_file, 'rb') as tr_rid:
            training_annotations = pickle.load(tr_rid)
        print("Training samples from :", training_annot_file)
        return training_annotations


def global_prune(scores_per_vid, annots_per_vid, names_per_vid, corresponding_frames, corresponding_vid_idx, frames_per_vid, prune_from_percent, new_percent, label_array, label_array_count):
    old_num = int(sys.argv[1])
    new_num = int(sys.argv[2]) - int(sys.argv[1])
    total_num = old_num + new_num
    new_per = new_num/100
    old_per = old_num/100
    new_training_annotations = []
    total_vids = len(names_per_vid)
    total_new_frames = int(total_vids*new_per)   # 2 frame per vid increment rate, roughly
    curr_label_count = np.zeros((30))

    #data95 = get_det_annots_prepared_old_unlabeled(95)
    #vid_95 = []
    #for i in range(len(data95)):
    #    video, anno = data95[i]
    #    vid_95.append(video)
    
    old_data = get_det_annots_prepared_old(old_num)
    old_videos = []
    for i in range(len(old_data)):
        video, anno = old_data[i]
        old_videos.append(video)

    complete_data = get_det_annots_prepared_old_unlabeled(95)
    complete_data_dict = {}
    for i in range(len(complete_data)):
        video, ann = complete_data[i]
        complete_data_dict[video] = ann
    
    scores_per_vid = np.array(scores_per_vid)
    scores_per_vid_dec = (-1.0)*scores_per_vid
    sorted_idx = np.argsort(scores_per_vid_dec)
    
    vid_id = 0
    vid_count = 0
    new_videos = []
    while vid_count < total_new_frames:
        video_name = names_per_vid[sorted_idx[vid_id]]
        label_val = label_array[sorted_idx[vid_id]]
        label_val_count = label_array_count[label_val]

        if video_name in old_videos:
            vid_id = vid_id + 1
            continue
        
        #annots = annots_per_vid[sorted_idx[vid_id]]
        annots = complete_data_dict[video_name]
        annots[0] = list(annots[0])
        annots[0][5]=1
        annots[0] = tuple(annots[0])

        new_training_annotations.append([video_name, annots])
        old_data.append([video_name, annots])
        new_videos.append(video_name)
        vid_id += 1
        vid_count += 1
        curr_label_count[label_val] +=1

    fp = 'data_lists/train_annots_'+str(total_num)+'_labeled.pkl'
    with open(fp,'wb') as wid:
        pickle.dump(old_data, wid, pickle.HIGHEST_PROTOCOL)
    print("Saved at ", fp)

    un_data = []
    for i in range(len(complete_data)):
        video, ann = complete_data[i]
        if video in old_videos or video in new_videos:
            continue
        
        un_data.append([video, ann])
    
    unsupervise_num = 100 - int(total_num)
    fp = 'data_lists/train_annots_'+str(unsupervise_num)+'_unlabeled.pkl'
    with open(fp,'wb') as wid:
        pickle.dump(un_data, wid, pickle.HIGHEST_PROTOCOL)
    print("Saved at ", fp)

def get_thresholded_arr(arr, threshold = 0.5):
    # b x 1 x h x w x 1  (FG)
    # b x 1 x h x w x 25 (CLS)
    
    if arr.shape[-1] > 1:
        arr_max = (arr == np.max(arr,-1,keepdims=True)).astype(float)
        arr *= arr_max
        arr[arr>0] = 1. 
    else:
        arr[arr>threshold] = 1.
        arr[arr<=threshold] = 0.  
    return arr


def get_active_score(frame_var, frame_output):
    
    frame_th = get_thresholded_arr(frame_output, threshold = 0.2)
    frame_th = frame_th.astype(np.bool)
    frame_var[frame_var == 0] = 1e-8
    uncertainty = frame_var[frame_th].sum() / (frame_th.sum()+1e-8)
    return uncertainty
        

if __name__ == '__main__':
    name='train'
    clip_shape=[224,224]
    channels=3
    batch_size = 1
    select_frames = 3   # Number of frames to add
    
    prune_from_percent = '5'        # Frames to prune from (source)
    print("Pruning from frames: ", prune_from_percent)
    new_percent = str(int(prune_from_percent) + select_frames)
    print("Pruning to new: ", new_percent)
    
    vid_idx = 0 #int(sys.argv[1])
    
    #model_type = 'Multi_{}FramesRand_Globalprune8wUN_GausInterp_MaxMask1.37_Dropout_PreTrainedCharades'.format(prune_from_percent)
    epoch = 21

    #model_file_path = 'active_learning/random/10.pth'
    print('--------------------------------- single frame noise---------------------------')
    prev_percent = int(sys.argv[1])
    model_file_path = 'main_weights/'+str(prev_percent)+'.pth'
    model = CapsNet()
    model.load_previous_weights(model_file_path)
    print("Model loaded from: ", model_file_path)
    model = model.to('cuda')
    model.eval()
    model.training = False
    
    # Enable dropout in eval mode 
    #for m in model.modules():
    #    if m.__class__.__name__.startswith("Dropout"):
    #        m.train()
    
    dataloader = UCF101DataLoader(name, clip_shape, batch_size)
    
    iou_thresh = np.arange(0, 1, 0.1)
    frame_tp = np.zeros((21, iou_thresh.shape[0]))
    frame_fp = np.zeros((21, 1))
    
    clip_span = 16
    num_vids = len(dataloader)
    max_idx = min(num_vids, vid_idx+200)
    clip_batch_size = 14
    num_forward_passes = 10
    uncertainty_thresh = -np.log(0.6)
    
    print("Total vids: ", num_vids)
    new_training_annotations = []
    
    done_vids = 0
    
    scores_per_vid = []
    annots_per_vid = []
    names_per_vid = []
    corresponding_frames = []
    corresponding_vid_idx = []
    frames_per_vid = []
    label_array = []
    label_array_count = np.zeros((30))
    start_time = time.time()
    with torch.no_grad():
        for i in range(num_vids):
            v_name, anns, video, bbox, label, annots = dataloader.get_item(i)
            if video is None:
                print("Skipping video: ", v_name)
                continue
                
            num_frames = video.shape[0]
            if num_frames == 0:
                print("Video has no frames: ", v_name)
                continue
            
            #vid_scores = np.zeros((num_frames))
            vid_scores = []
            # prepare batches of this video, get results from model, stack np arrays for results 
            batches = 0
            bbox_pred_fg = np.zeros((num_frames, clip_shape[0], clip_shape[1], 1))
            bbox_pred_fg_output = np.zeros((num_frames, clip_shape[0], clip_shape[1], 1))
            while True:
                batch_frames = np.zeros((1,8,224,224,3))
                for j in range(8):
                    ind = (batches * clip_span) + (j * 2)
                    if ind >= num_frames:
                        if j > 0:
                            batch_frames[0,j] = batch_frames[0,j-1]
                    else:
                        batch_frames[0,j] = video[ind]
                
                data = np.transpose(np.array(batch_frames), [0, 4, 1, 2, 3])
                data = torch.from_numpy(data).type(torch.cuda.FloatTensor)
                action_tensor = np.ones((len(batch_frames),1), np.int) * 500
                action_tensor = torch.from_numpy(action_tensor).cuda()
                empty_vector = torch.zeros(action_tensor.shape[0]).cuda()
                
                segmentation_np = np.zeros((len(batch_frames), 1, 8, clip_shape[0], clip_shape[1]))
                segmentation_np_output = np.zeros((len(batch_frames), 1, 8, clip_shape[0], clip_shape[1]))
                data_new = torch.cat((data,data,data,data,data,data,data,data), 0) 
                random_noise = torch.rand(data_new.shape).cuda()
                random_noise = (random_noise-0.5)/10
                data_new = data_new+random_noise

                segmentation, pred, _ = model(data_new, action_tensor, empty_vector, 0, 0)
                segmentation = F.sigmoid(segmentation)
                segmentation = segmentation.cpu().data.numpy()   # B x C x F x H x W -> B x 1 x 8 x 224 x 224

                segmentation_np_output += np.mean(segmentation, axis=0).reshape(len(batch_frames), 1, 8, clip_shape[0], clip_shape[1])
                segmentation_np_output = np.transpose(segmentation_np_output, [0, 2, 3, 4, 1])
                output_fg_output = segmentation_np_output[0]      
                output_fg_output = np.repeat(output_fg_output, 2, axis=0)

                segmentation = np.var(segmentation, axis=0)  
                segmentation_np += segmentation
                segmentation_np = np.transpose(segmentation_np, [0, 2, 3, 4, 1])
                output_fg = segmentation_np[0]      # F x H x W x C
                output_fg = np.repeat(output_fg, 2, axis=0)
                
                end_idx = (batches+1) * clip_span
                if end_idx > num_frames:
                    end_idx = num_frames
                start_idx = batches * clip_span
                bbox_pred_fg[start_idx : end_idx] = output_fg[0:(end_idx - start_idx)]
                bbox_pred_fg_output[start_idx : end_idx] = output_fg_output[0:(end_idx - start_idx)]
                
                if end_idx >= num_frames:
                    break
                    
                batches += 1
            print('num of frmaes  ',num_frames)
            for f_idx in range(num_frames):
                
                output_fg = bbox_pred_fg[f_idx]
                output_fg_output = bbox_pred_fg_output[f_idx]

                if np.sum(output_fg_output) > 0.1:
                    uncertainty_score = get_active_score(output_fg, output_fg_output)
                    vid_scores.append(uncertainty_score)

            if not vid_scores:
                vid_scores.append(0.0)

            vid_scores = np.array(vid_scores)
            vid_scores_mean = vid_scores.mean()
            label_array.append(label)
            label_array_count[label] +=1 
            scores_per_vid.append(vid_scores_mean)
            annots_per_vid.append(annots)
            names_per_vid.append(v_name)
            frames_per_vid.append(num_frames)
            corresponding_frames.extend(np.arange(num_frames))
            corresponding_vid_idx.extend(np.ones((num_frames))*done_vids)
            
            done_vids += 1
            
            if i%200 == 0:
                print("Done vids: ", i+1)
            #print(v_name)
            #print(new_annotations)
            #exit()
    end_time = time.time()
    print('time taken 5 percent: ', end_time - start_time)
    print("Global pruning")
    global_prune(scores_per_vid, annots_per_vid, names_per_vid, corresponding_frames, corresponding_vid_idx, frames_per_vid, prune_from_percent, new_percent, label_array, label_array_count)
    for scores in scores_per_vid:
        print(scores)
    print("Saved vids: ", done_vids)
    print("Total vids: ", num_vids)
    exit(0)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
         
      
      
      
      
