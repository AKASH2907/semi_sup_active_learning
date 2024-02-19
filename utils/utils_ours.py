import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import cv2

def normalize_image(pic):
    # print('type is ',type(pic))
    if pic.min() == 0 and pic.max()==0:
        return(pic)
    else:
        npic = (pic - pic.min()) / (pic.max() - pic.min())
        return npic



def show(image, title='.'):
    # display an image along with title
    # handles PIL format,and numpy arrays
    if isinstance(image, torch.Tensor) and len(image.size()) == 3:
        # image = image.numpy()
        print(image.shape)
        image = image.permute(1, 2, 0)
    # image = normalize_image(image)
    f, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.set_title(title, fontsize=30)
    plt.show()

def overlay(img,mask,orig,title='.'):
    masked = np.ma.masked_where(mask == 0, mask)
    img_masked = np.ma.masked_where(img == 0, img)

    img = img + 1
    img[img > 1] = .9


    f, ax = plt.subplots(figsize=(10, 10))
    # pic = orig
    pic = orig[:,1,:,:]
    pic = np.transpose(pic,(1,2,0))
    pic = normalize_image(pic)
    ax.imshow(pic)
    ax.imshow(img_masked,'autumn',interpolation='none', alpha=0.5)
    ax.imshow(masked, 'jet', interpolation='none', alpha=0.5)
    ax.set_title(title, fontsize=30)
    plt.show()

def side(img,mask,orig,title='.'):
    masked = np.ma.masked_where(mask == 0, mask)
    img_masked = np.ma.masked_where(img == 0, img)

    img = img + 1
    img[img > 1] = .9


    f, ax = plt.subplots(figsize=(10, 10))
    # pic = orig
    pic = orig[:,0,:,:]
    pic = np.transpose(pic,(1,2,0))
    pic = normalize_image(pic)
    ax.imshow(pic)
    ax.imshow(img_masked,'autumn',interpolation='none', alpha=0.5)
    ax.set_title(title, fontsize=30)
    plt.show()

def byside(img,mask,orig,title='.'):
    masked = np.ma.masked_where(mask == 0, mask)
    img_masked = np.ma.masked_where(img == 0, img)

    img = img + 1
    img[img > 1] = .9


    f, ax = plt.subplots(figsize=(10, 10))
    # pic = orig
    pic = orig[:,0,:,:]
    pic = np.transpose(pic,(1,2,0))
    pic = normalize_image(pic)
    ax.imshow(pic)
    # ax.imshow(img_masked,'autumn',interpolation='none', alpha=0.5)
    ax.imshow(masked, 'jet', interpolation='none', alpha=0.5)
    ax.set_title(title, fontsize=30)
    plt.show()


def overlay2(mask,orig,title='.'):
    masked = np.ma.masked_where(mask == 0, mask)
    # histogram(masked)
    # img_masked = np.ma.masked_where(img == 0, img)

    # img = img + 1
    # img[img > 1] = .9


    f, ax = plt.subplots(figsize=(10, 10))
    pic = orig
    # pic = orig[:,1,:,:]
    # pic = np.transpose(pic,(1,2,0))
    # pic = normalize_image(pic)
    ax.imshow(pic)
    # ax.imshow(img_masked,'autumn',interpolation='none', alpha=0.5)
    ax.imshow(masked, 'autumn', interpolation='none', alpha=0.5)
    ax.set_title(title, fontsize=30)
    plt.show()

def oldIOU(gt,img,orig):
    #takes ground truth, gt, and and ouput image ,img, and calculates IOU
    # make sure they are in numpy
    #test to see if they are binary - reject or fix if not

    for i in range(0,10):
        intersection = gt[i] + img[i]

        intersection[intersection < 2] = 0
        intersection[intersection > 0] = 1
        intersection_sum = intersection.sum()

        union = gt[i] + img[i]
        union[union > 1] = 1
        union_sum = union.sum()


        IOU = intersection_sum/union_sum
        # overlay(gt[i],img[i],orig[i],IOU)

    return IOU

def IOU(gt,img):
    #takes ground truth, gt, and and ouput image ,img, and calculates IOU
    # make sure they are in numpy
    #test to see if they are binary - reject or fix if not


    intersection = gt + img

    intersection[intersection < 2] = 0
    intersection[intersection > 0] = 1
    intersection_sum = intersection.sum()

    union = gt + img
    union[union > 1] = 1
    union_sum = union.sum()

    if union_sum > 0:
        IOU = intersection_sum/union_sum
    else:
        # print('union sum not positive ',union_sum)
        IOU = torch.Tensor([0])

    return IOU

def IOU2(gt,img):
    #takes ground truth, gt, and and ouput image ,img, and calculates IOU
    # make sure they are in numpy
    #test to see if they are binary - reject or fix if not


    intersection = gt + img

    intersection[intersection < 2] = 0
    intersection[intersection > 0] = 1
    intersection_sum = intersection.sum()

    union = gt + img
    union[union > 1] = 1
    union_sum = union.sum()

    if gt.sum() > 0:
        IOU = intersection_sum/union_sum
    else:
        # print('union sum not positive ',union_sum)
        IOU = float('NaN')

    return IOU

def basic_overlay(img,mask,title='.'):
    masked = np.ma.masked_where(mask == 0, mask)
    img_masked = np.ma.masked_where(img == 0, img)

    img = img + 1
    img[img > 1] = .9


    f, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(img_masked,'autumn',interpolation='none', alpha=0.5)
    ax.imshow(masked, 'jet', interpolation='none', alpha=0.5)
    ax.set_title(title, fontsize=30)
    plt.show()
def testIOU():
    a = np.zeros((10,10))
    a[3:6,3:6]= 1
    b = np.zeros((10, 10))
    b[3:6,3:6]= 1
    iou = IOU(a,b)
    basic_overlay(a,b,iou)

    a = np.zeros((10,10))
    a[3:6,3:6]= 1
    b = np.zeros((10, 10))
    b[7:9,7:9]= 1
    iou = IOU(a,b)
    basic_overlay(a,b,iou)

    a = np.zeros((10,10))
    a[3:6,3:6]= 1
    b = np.zeros((10, 10))
    b[5:8,3:6]= 1
    iou = IOU(a,b)
    basic_overlay(a,b,iou)

    a = np.zeros((10,10))
    a[3:7,3:7]= 1
    b = np.zeros((10, 10))
    b[4:6,4:6]= 1
    iou = IOU(a,b)
    basic_overlay(a,b,iou)
    
def histogram(arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu()
        arr = arr.data.numpy()
    num_bins = 200
    arr = arr.ravel()
    n, bins, patches = plt.hist(arr, num_bins, facecolor='blue', alpha=0.5)
    plt.show()

def measure_pixelwise_var_v2(pred, flip_pred, frames_cnt=5, use_sig_output=False):
    """cyclic variance
    varv3 - cyclic version
    """
    count = 0
    batch_variance = np.zeros((pred.shape[0], 1, 8, 224, 224))

    # remove the redundant frames 1dt n last of flipped map
    temp_batch_var = np.zeros((pred.shape[0], 1, 14, 224, 224))

    # use sigmoid output not logits
    # probability scores
    if use_sig_output==True:
        pred = torch.sigmoid(pred)
        flip_pred = torch.sigmoid(flip_pred)

    for zz in range(0, pred.shape[0]):
        clip = pred[zz][0]
        flip_clip = flip_pred[zz][0]

        cyclic_clip = torch.cat([clip, flip_clip[1:7]], axis=0).cpu().detach().numpy()

        # calculated variance over 14 frames
        clip_variance = np.zeros_like(temp_batch_var[0][0])
        for temp_cnt in range(temp_batch_var.shape[2]):
            # 3 frames
            if frames_cnt==3:
                if temp_cnt+1>(temp_batch_var.shape[2] - 1):
                    temp_var = np.take(cyclic_clip, indices=[temp_cnt-1, temp_cnt, 0], axis=0)
                else:
                    temp_var = np.take(cyclic_clip, indices=[temp_cnt-1, temp_cnt, temp_cnt+1], axis=0)
            # 5 frames
            if frames_cnt==5:
                if temp_cnt+1>(temp_batch_var.shape[2] - 1):
                    temp_var = np.take(cyclic_clip, indices=[temp_cnt-2, temp_cnt-1, temp_cnt, 0, 1], axis=0)
                elif temp_cnt+2>(temp_batch_var.shape[2] - 1):
                    temp_var = np.take(cyclic_clip, indices=[temp_cnt-2, temp_cnt-1, temp_cnt, temp_cnt+1, 0], axis=0)
                else:
                    temp_var = np.take(cyclic_clip, indices=[temp_cnt-2, temp_cnt-1, temp_cnt, temp_cnt+1, temp_cnt+2], axis=0)
            
            temp_var = np.var(temp_var, axis=0)
            clip_variance[temp_cnt] = temp_var

        # overlap
        for add_half in range(8):
            if add_half==0 or add_half==7:
                clip_variance[add_half] = 2* clip_variance[add_half]
            else:
                clip_variance[add_half] = clip_variance[add_half] + clip_variance[14-add_half]
        # normalize
        clip_variance = clip_variance[:8]
        clip_variance -= clip_variance.min()
        clip_variance /= (clip_variance.max() - clip_variance.min() + 1e-7)
        
        clip_variance = np.expand_dims(clip_variance, axis=0)
        batch_variance[zz] = clip_variance
    batch_variance = torch.from_numpy(batch_variance)

    return batch_variance

def dft_HighPass(pred, radius = 32):
    batch_video = pred.cpu().detach().numpy()
    batch_filtered = np.zeros((batch_video.shape[0], 1, 8, 224, 224))
    
    mask = np.zeros_like(batch_video[0][0][0])
    cy = mask.shape[0] // 2
    cx = mask.shape[1] // 2
    cv2.circle(mask, (cx,cy), radius, (255,255,255), -1)[0]
    mask = 255 - mask
    mask = cv2.GaussianBlur(mask, (19,19), 0)
    
    for zz in range(0, batch_video.shape[0]):
        clip = batch_video[zz][0]
        
        for xx in range(clip.shape[0]):
            img = clip[xx]
            
            dft = np.fft.fft2(img, axes=(0,1))
            dft_shift = np.fft.fftshift(dft)
    
            dft_shift_masked = np.multiply(dft_shift,mask) 
            back_ishift_masked = np.fft.ifftshift(dft_shift_masked)

            img_filtered = np.fft.ifft2(back_ishift_masked, axes=(0,1))
            img_filtered = np.abs(img_filtered).clip(0,1)
            
            
            batch_filtered[zz][0][xx] = img_filtered
    
    batch_filtered = torch.from_numpy(batch_filtered)
    
    return batch_filtered




def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    def warpper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.0, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0
    return warpper

def weighted_mse_loss(input, target, weight):

    return (weight * (input - target) ** 2).mean()

def weighted_mae_loss(input, target, weight):

    return (weight * (input - target)).mean()

# def mAP(gt,img):
#     #takes ground truth, gt, and and ouput image ,img, and calculates IOU
#     # make sure they are in numpy
#     #test to see if they are binary - reject or fix if not
#
#
#     intersection = gt + img
#
#     intersection[intersection < 2] = 0
#     intersection[intersection > 0] = 1
#     intersection_sum = intersection.sum()
#
#     gt_sum = gt.sum()
#
#     if gt_sum > 0:
#         overlap = intersection_sum/gt_sum
#     else:
#         # print('union sum not positive ',union_sum)
#         overlap = float('NaN')
#
#     return overlap


if __name__ == "__main__":

    import torchvision.models as models
    from tensorboardX import SummaryWriter
    from torch.autograd import Variable
    import torch
    resnet18 = models.resnet18(False)
    writer = SummaryWriter()
    for name, param in resnet18.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), 0)
    writer.export_scalars_to_json("./all_scalars.json")
    dummy_img = Variable(torch.rand(32, 3, 64, 64))
    res = resnet18(dummy_img)
    writer.add_graph(resnet18, res)
    writer.close()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--awesome', default='yes')
    # args = parser.parse_args()
    # print('bloop')
    # print(args.awesome)

    # def write_csv(self):
    #     with open('master.csv','w') as csv_file:
    #         writer = csv.writer(csv_file)
    #         for a in self.master:
    #             writer.writerow([a.get('path'),a.get('actor'),a.get('view'),a.get('label'),a.get('frame')])
    #
    # def read_csv(self):
    #     self.master = []
    #     with open('master.csv', 'r') as csv_file:
    #         reader = csv.reader(csv_file)
    #         for row in reader:
    #             self.master.append({'path': row[0], 'actor': row[1], 'view': row[2], 'label': row[3], 'frame': row[4]})


