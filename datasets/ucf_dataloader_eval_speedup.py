import numpy as np
import os.path as osp
from torch.utils.data import Dataset
import cv2
import h5py
import os
from skvideo.io import vread


class UCF101Dataset(Dataset):
    def __init__(self, mode, clip_shape, cl, file_id=None):
        self._dataset_dir =  'Datasets/UCF101'

        if mode == 'train':
            self.vid_files = self.get_det_annots_prepared(file_id)
            self.shuffle = True
            self.mode = 'train'
        else:
            self.vid_files = self.get_det_annots_test_prepared()
            self.shuffle = False
            self.mode = 'test'

        self._height = clip_shape[0]
        self._width = clip_shape[1]
        self.cl = cl
        self._size = len(self.vid_files)
        self.indexes = np.arange(self._size)
            

    def get_det_annots_prepared(self, file_id):
        import pickle

        training_annot_file = "../data_lists/data_subset_pkl_files_seed_{}/".format(str(self.subset_seed)) + file_id

        with open(training_annot_file, 'rb') as tr_rid:
            training_annotations = pickle.load(tr_rid)
        print("Training samples from :", training_annot_file)

        return training_annotations
        
        
    def get_det_annots_test_prepared(self):
        import pickle
        testing_anns = "data_lists/test_annots.pkl"
        with open(testing_anns, 'rb') as ts_rid:
            testing_annotations = pickle.load(ts_rid)

        return testing_annotations
    

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.vid_files)

    def __getitem__(self, index):
        
        v_name, anns = self.vid_files[index]
        clip, bbox_clip, label, _ = self.load_video(v_name, anns)
        
        # Center crop
        _, clip_h, clip_w, _ = clip.shape

        start_pos_h = int((clip_h - self._height) / 2)
        start_pos_w = int((clip_w - self._width) / 2)
        
        clip = clip[:, start_pos_h:start_pos_h+self._height, start_pos_w:start_pos_w+self._width, :] / 255.
        bbox_clip = bbox_clip[:, start_pos_h:start_pos_h+self._height, start_pos_w:start_pos_w+self._width, :]
        return clip, bbox_clip, label
    

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
            print('Error:', str(video_dir))
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
        #multi frame mode
        annot_idx = 0
        if len(annotations) > 1:
            annot_idx = np.random.randint(0,len(annotations))
        multi_frame_annot = []      # annotations[annot_idx][4]
        bbox_annot = np.zeros((n_frames, h, w, 1), dtype=np.uint8)
        #for ann in annotations:
        ann = annotations[annot_idx]    # For caps, use only 1 object at a time. 
        multi_frame_annot.extend(ann[4])
        start_frame, end_frame, label = ann[0], ann[1], ann[2]
        collect_annots = []
        for f in range(start_frame, min(n_frames, end_frame+1)):
            try:
                x, y, w, h = ann[3][f-start_frame]
                bbox[f, y:y+h, x:x+w, :] = 1
                if f in ann[4]:
                    collect_annots.append([x,y,w,h])
            except:
                print('ERROR LOADING ANNOTATIONS')
                print(start_frame, end_frame)
                print(video_dir)
                exit()
        
        multi_frame_annot = list(set(multi_frame_annot))
        if self.mode == 'train':
            return video, bbox_annot, label, multi_frame_annot
        else:
            return video, bbox, label, multi_frame_annot

        


def pil_to_cv(pil_image):
    """
    Returns a copy of an image in a representation suited for OpenCV
    :param pil_image: PIL.Image object
    :return: Numpy array compatible with OpenCV
    """
    return np.array(pil_image)[:, :, ::-1]


def write_video(file_path, frames, fps):
    """
    Writes frames to an mp4 video file
    :param file_path: Path to output video, must end with .mp4
    :param frames: List of PIL.Image objects
    :param fps: Desired frame rate
    """

    w, h = 224, 224
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))

    for frame in frames:
        writer.write(pil_to_cv(frame))
        # print(frame.shape)
        # writer.write(frame)

    writer.release()
    

if __name__ == '__main__':
    import imageio    
    from pathlib import Path
    from torch.utils.data import DataLoader
    mode='test'
    clip_shape=[224,224]
    channels=3
    batch_size = 1
    dataset = UCF101Dataset(mode, clip_shape, batch_size, False)
    print(len(dataset))
    

    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    print(len(train_dataloader))
    # exit()
    save_path = 'dataloader_viz/resize_erase_crop_debug/'
    Path(save_path).mkdir(parents=True, exist_ok=True)

    vid_vis_counter = 0

    for i, data in enumerate(train_dataloader):
        if i%25==0:
            print("Data iter:", i)
        orig_clip, aug_clip = data['weak_data'], data['strong_data']
        clip_mask = data['weak_mask']
        strong_mask = data['strong_mask']

        vid_class = data['action']
        vid_labeled = data['label']
        aug_probab_array = data['aug_probab']

        if orig_clip.shape[0]!=8:
            print(orig_clip.shape, aug_clip.shape)
            print(clip_mask.shape, strong_mask.shape)
        # print(vid_class, vid_labeled, aug_probab_array)
        # exit()
        # check collate function
        # if orig_clip.shape[0]!=8:
        #     print(orig_clip.shape, aug_clip.shape)
        #     print(clip_mask.shape, strong_mask.shape)
        #     print(data['label'])
        #     exit()

        # print(vid_class, vid_labeled)
        
        # # AUG-VIZ
        orig_clip = np.transpose(orig_clip.numpy(), [0, 2, 3, 4, 1])
        aug_clip = np.transpose(aug_clip.numpy(), [0, 2, 3, 4, 1])
        clip_mask = np.transpose(clip_mask.numpy(), [0, 2, 3, 4, 1])
        strong_mask = np.transpose(strong_mask.numpy(), [0, 2, 3, 4, 1])

