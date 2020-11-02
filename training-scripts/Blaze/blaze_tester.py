
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from datetime import datetime
from tensorboardX import SummaryWriter
from tqdm import tqdm
from torchvision.utils import save_image, make_grid
from skimage import io
import torchvision
from torch.utils.data import RandomSampler
from torch import autograd 
import random
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import pandas as pd
from random import shuffle
import numpy as np
import cv2
from skimage.transform import resize
import pandas as pd
from tqdm import tqdm
from skimage import img_as_uint
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import time
import gc
from itertools import cycle
import matplotlib.patches as patches
from skimage.transform import resize
from completion_models import FCN3DSTN, Discriminator
import traceback
import sys
from mouth_detector import MouthDetector
from torch.utils.data.dataloader import default_collate

torch.set_num_threads(1)

torch.manual_seed(123)
np.random.seed(123)
random.seed(123)


# torch.backends.cudnn.enabled = False

device_index = sys.argv[1]

device_id = 'cuda:{}'.format(int(device_index.strip()))
device = torch.device(device_id)
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


# In[6]:
ROI_DATA_PATH = '../../../datasets/lrs_rois/'

# with open('../lrs_eval_paths.txt') as f:
#     paths = f.readlines()
# shuffle(paths)
# test_paths  = paths
# test_paths = [path.strip() for path in test_paths]
# test_paths = [path for path in test_paths if '.mp4' in path]
# # train_paths = (paths[:int(0.8*len(paths))])
# # test_paths = (paths[int(0.8*len(paths)):])
# train_paths = pd.read_csv('../lrs_train_paths.csv')['location']
# train_paths = train_paths.tolist()
# train_paths = [path for path in train_paths if '.mp4' in path]

df_test = pd.read_csv('./test_len.csv')
df_train = pd.read_csv('./train_len.csv')
test_paths = df_test['0']
train_paths = df_train['0']
train_paths = train_paths[:int(len(train_paths)*.80)]
val_paths = train_paths[int(len(train_paths)*.80):]#validation set
# import IPython; IPython.embed()
# exit()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_3d_gaussian(window_size, sigma=5):
    gauss_1 = gaussian(window_size, sigma)
    
    gaussian_kernel = torch.zeros((window_size, window_size, window_size))
    for i in range(window_size):
        for j in range(window_size):
            for k in range(window_size):
                gaussian_kernel[i,j,k] = gauss_1[i] * gauss_1[j] * gauss_1[k]
    
    gaussian_kernel = gaussian_kernel.float()
    gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0)
    
    return gaussian_kernel

class SSIM(torch.nn.Module):
    def __init__(self, device, size_average=True, window_size=11, sigma=1.5):
        super(SSIM, self).__init__()
        self.kernel = create_3d_gaussian(window_size, sigma).to(device)
        self.size_average = size_average
        
    def forward(self, img1, img2):
        mu1 = F.conv3d(img1, self.kernel, padding = self.kernel.shape[2]//2, groups = 1)
        mu2 = F.conv3d(img2, self.kernel, padding = self.kernel.shape[2]//2, groups = 1)

        mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2

        sigma1_sq = F.conv3d(img1**2, self.kernel, padding = self.kernel.shape[2]//2, groups = 1) - mu1_sq
        sigma2_sq = F.conv3d(img2**2, self.kernel, padding = self.kernel.shape[2]//2, groups = 1) - mu2_sq    
        sigma12 = F.conv3d(img1*img2, self.kernel, padding = self.kernel.shape[2]//2, groups = 1) - mu1_mu2

        L = 1
        C1, C2 = (0.01*L)**2, (0.03*L)**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return torch.mean(ssim_map) + 1 # ssim varies from -1 to 1
        else:
            ssim_map
            

def sort_paths(paths):
    paths_ = [(int(name.split('.')[0]), name) for name in paths]
    paths_ = sorted(paths_, key=lambda x:x[0])
    paths_ = [path[1] for path in paths_]
    return paths_

def read_roi_frames(path, root_dir='train'): 
    subdir = path.split('/')[-2] 
    dir_ = path.split('/')[-1].split('.mp4')[0] 
    path_ = os.path.join(ROI_DATA_PATH, root_dir,subdir, dir_) 
    frames = [] 
    img_names = sort_paths(os.listdir(path_))
    counter = 0
    for img in img_names: 
        frames.append(io.imread(os.path.join(path_, img))) 
        counter += 1 
        if counter == 31:
            break
    return np.array(frames) 


################################################ Dataloader
class LRSVideoHandler(object):
    def __init__(self, filepaths):
#         self.root_dir = root_dir
        self.paths = filepaths
    
    def read_video_audio(self, video_path, audio_path=None, audio=False):

        frames = [] 
#         print('fetching frames from ', video_path)
#         exit()
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
#                 frame = cv2.resize(frame, (64, 64), cv2.INTER_CUBIC)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = resize(frame, (128, 128))
                
                frames.append(frame)

            else:
#                 print('ret is false')
                break
            if len(frames) == 32:
                break
        cap.release()
#         import ipdb; ipdb.set_trace()
        frames = np.array(frames)
        frames = (frames - frames.min()) / (frames.max() - frames.min())
#         print('frames fetched', frames.shape)
        #         video_frames = torch.FloatTensor(list(clip.iter_frames()))
        video_frames = torch.from_numpy(frames)
#         print('elapsed: ', time() - t2)
        if audio:
            waveform, sample_rate = torchaudio.load_wav(audio_path)

            specgram = torchaudio.transforms.MelSpectrogram()(waveform)
        
#         print('elapsed : ', time() - t1)
        
            return specgram, video_frames
        else:
            return video_frames , len(frames)

    def save_wave(self, video_name):
        try:
            in_stream = ffmpeg.input(video_name)
            out_stream = ffmpeg.output(in_stream, video_name.split('.mp')[0] + '.wav')
            out_stream = ffmpeg.overwrite_output(out_stream)
            ffmpeg.run(out_stream)
        except:
            logging.basicConfig(level=logging.DEBUG, filename='./dl_error_logs/{}.log'.format(video_name.split('/')[-1].split('.')[0]), filemode='w')

            logging.error(video_name)

    def save_wavefiles(self, num_p=6):
 
        from multiprocessing import Pool
        with Pool(num_p) as p:
            r = list(tqdm(p.imap(self.save_wave, (self.mp4_filenames)), total=len(self.mp4_filenames)))
            
    

def reversed_frames(frames):
    out = torch.zeros_like(frames)
    out[0] = frames[0].clone()
    out[-1] = frames[-1].clone()
    for i in range(1, len(frames)-1):
        out[frames.shape[0]-1-i] = frames[i].clone()
    return out


def get_padded_frames(in_frames, out_frames, device='cuda', frames_len=32, mask=True):
    in_shape = in_frames.shape
    DIM = in_frames.shape[-1]
    common_shape = (frames_len, 3, DIM, DIM)
    common = torch.rand(*common_shape)

    in_frames_new = torch.ones(frames_len, 4, DIM, DIM)
    in_frames_new[:, :3, :, :] = common.clone()
    in_frames_new[:, -1, :, :] = 0
    
    out_frames_new = common.clone()
    
    if frames_len > in_frames.shape[0]:
        end_index = in_frames.shape[0]
    else:
        end_index = frames_len
    
    out_frames_new[:end_index, :, :, :] = out_frames[:end_index].clone().to(device)
    in_frames_new[:end_index, :, :, :] = in_frames[:end_index].clone().to(device)
    
    return out_frames_new, in_frames_new 




####################################################

############## LOAD files


# In[9]:


def m_strip(val):
    return val.strip()



train_vhandler = LRSVideoHandler(train_paths)
test_vhandler = LRSVideoHandler(test_paths)

def run_corruption_with_mask( frames, p=0.5):

    count = 0
    no_corrupt_frames  = int(len(frames) * p_corr)
    frame_indices = list(range(len(frames)))
    indices = random.sample(frame_indices, no_corrupt_frames)

    for index in indices:
        frames[index,:3] = torch.rand(*frames[index,:3].shape)
        frames[index,-1] = torch.zeros(*frames[index,-1].shape)

    return frames

def run_corruption_prefix_suffix(frames, c_type='suffix'):
    global p_corr
    num_frames = int(len(frames) * p_corr)
    indices = list(range(len(frames)))
    if c_type == 'prefix':
        indices = indices[:num_frames]
    else:
        indices = indices[len(indices)-num_frames:]
    for index in indices:
        frames[index,:3] = torch.rand(*frames[index,:3].shape)
        frames[index,-1] = torch.zeros(*frames[index,-1].shape)
    return frames

batch_size = 1

###################################### MODEL


test_mode = False
p_corr = 0.40
corr_name = 'random'
# In[17]:

################################# CREATE DIRS
root_dir = './Viseme_train_outputs_fcn3d_stn_plain_roi_blaze_cnn_TEST_FULL_DATASET'
now = str(datetime.now()) + '_TEST_FULL_{}_{}'.format(int(p_corr*100), corr_name)
if test_mode:
    now += '_test' 
    
subdir = os.path.join(root_dir, now)


log_dir = os.path.join(subdir, 'logs')
traindir = os.path.join(subdir, 'train_outputs')
testdir = os.path.join(subdir, 'test_outputs')
modeldir = os.path.join(subdir, 'models')
out_dirs = [root_dir, subdir, log_dir, traindir, testdir, modeldir]
for dir_ in out_dirs:
    if not os.path.exists(dir_):
        os.makedirs(dir_)


############################################## Train and Test functions
def normalize_img(in_arr):
    in_arr = in_arr - in_arr.min() / (in_arr.max() - in_arr.min())
    in_arr *=255
    return in_arr

def save_sample(x, y, m, o_x, o_m, dir_, name):
    
#     print(x.shape, y.shape, m.shape, o_x.shape, o_m.shape)
    out_x = normalize_img(make_grid(x[0,:3].permute(1,0,2,3)).permute(1,2,0)).cpu().numpy().astype(np.uint8)
    mask_x = make_grid(x[0,-1].unsqueeze(0).repeat(3,1,1,1).permute(1,0,2,3)).permute(1,2,0).cpu().numpy().astype(np.uint8) * 255

    out_y = normalize_img(make_grid(y[0].permute(1,0,2,3))).permute(1,2,0).cpu().numpy().astype(np.uint8)

    out_m = normalize_img(make_grid(m[0].permute(1,0,2,3))).permute(1,2,0).cpu().numpy().astype(np.uint8)

    o_rois = normalize_img(make_grid(o_m[0].permute(1,0,2,3))).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
    
    o_frames = normalize_img(make_grid(o_x[0].permute(1,0,2,3))).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
    
    io.imsave(os.path.join(dir_, name+'_in.png'), out_x)
    io.imsave(os.path.join(dir_, name+'_in_mask.png'), mask_x)
    io.imsave(os.path.join(dir_, name+'_gt.png'), out_y)
    io.imsave(os.path.join(dir_, name+'_gt_roi.png'), out_m)
    io.imsave(os.path.join(dir_, name+'_out_rois.png'), o_rois)
    io.imsave(os.path.join(dir_, name+'_out.png'), o_frames)

def save_model(optimizerG,  model,  epoch, batch):
    torch.save({'epoch': epoch, 'batch': batch, 'model_state_dict': model.state_dict(),
                 'optimizerG': optimizerG.state_dict()}, os.path.join(modeldir, '{}_{}.pt'.format(epoch, batch)))

# save_sample(x, y, m , out, out_m, subdir, 'test')

train_path_len = len(train_paths)
test_path_len = len(test_paths)
val_path_len = len(val_paths)

def prepare_batch(paths, detector):
    count = 0 
    file_count = 0
    x_list = []
    y_list = []
    roi_list = []
    file_paths = []
    while count < batch_size:
        indices = list(range(len(paths)))
        vid_path = paths.pop(random.choice(indices))
        frames, len_ =  train_vhandler.read_video_audio(os.path.join('../',vid_path))
        frames_y_tensor = (frames).float()
        file_count += 1
        if len_ < 30:
#             print('Reading video ', vid_path)
            continue
#         print('Reading video ', vid_path)
#         print(frames_y_tensor.shape, len_)
        try:
            mouth_frames, flag = get_roi_frames(detector, frames.permute(0,3,1,2))

        except Exception as e:
            print('error in ', vid_path)
            traceback.print_exc(file=sys.stdout)
            print(e)
#             exit()
            continue
        if flag:
           continue  
        file_paths.append(vid_path)
        padded_m_frames = get_padded_roi((mouth_frames), len(mouth_frames))
        x,y = process_xy(frames_y_tensor)
        x = x.permute(1,0,2,3).unsqueeze(0).to(device)
        y = y.permute(1,0,2,3).unsqueeze(0).to(device)
        m = torch.from_numpy(padded_m_frames).permute(3,0,1,2).unsqueeze(0).to(device)

        x = x[:,:,:,::2,::2]
        y = y[:,:,:,::2,::2]

        x_list.append(x)
        y_list.append(y)
        roi_list.append(m)
        count += 1
    return  x_list, y_list, roi_list, file_count, file_paths

        
def test_step(model, epoch, batch, test_paths, detector, summary_writer, p_corr):
    print('Testing ############', p_corr)
    
    criterion_mse = nn.MSELoss()
    criterion_ssim = SSIM(device).to(device)
    
    batch_frame_loss_l1 = []
    batch_roi_loss_l1 = []
    
    batch_frame_loss_mse = []
    batch_roi_loss_mse = []
    
    
    batch_frame_loss_mse_int8 = []
    batch_roi_loss_mse_int8 = []
    
    batch_frame_loss_ssim = []
    batch_roi_loss_ssim = []
    
    batch_frame_loss_psnr = []
    batch_roi_loss_psnr = []
#     batch_rois_loss_psnr = []
    
    batch_total_loss = []
    test_steps = len(test_paths) + 1
    global_file_count = 0
    test_paths_copy = test_paths[:]
    working_paths = []
#     for index in tqdm(range((test_steps))):
    index = 0
    while len(test_paths) > 0:
        
        x_list, y_list, roi_list, file_count, file_paths = prepare_batch(test_paths, detector)
        
        global_file_count += file_count
        x = torch.cat(x_list, dim=0)
        y = torch.cat(y_list, dim=0)
        m = torch.cat(roi_list, dim=0)
        working_paths.extend(file_paths)
        if test_mode:
            print(file_paths)
#         print(x.shape, y.shape, m.shape)
        model.eval()
        with torch.no_grad():
            out_frames, out_rois = model(x)
            if test_mode:
                print('out shapes' , out_frames.shape, out_rois.shape)
        
        loss_frames = criterion(out_frames, y)
        loss_rois = criterion(out_rois, m)
        batch_frame_loss_l1.append(loss_frames.item())
        batch_roi_loss_l1.append(loss_rois.item())
        
#         print(out_frames.shape, y.shape)
        loss_ssim_frames = criterion_ssim(out_frames.mean(1).unsqueeze(1), y.mean(1).unsqueeze(1))
        loss_ssim_rois = criterion_ssim(out_rois.mean(1).unsqueeze(1), m.mean(1).unsqueeze(1))
        batch_frame_loss_ssim.append(loss_ssim_frames.item())
        batch_roi_loss_ssim.append(loss_ssim_rois.item())
        
        mse_loss_frames = criterion_mse(out_frames, y)
        mse_loss_rois = criterion_mse(out_rois, m)
        batch_frame_loss_mse.append(mse_loss_frames.item())
        batch_roi_loss_mse.append(mse_loss_rois.item())
        
        mse_loss_frames_int8 = criterion_mse(out_frames*255., y*255.)
        mse_loss_rois_int8 = criterion_mse(out_rois*255., m*255.)
        batch_frame_loss_mse_int8.append(mse_loss_frames_int8.item())
        batch_roi_loss_mse_int8.append(mse_loss_rois_int8.item())
        
        psnr_val_frames = 20 * (torch.log10(255./ torch.sqrt(mse_loss_frames_int8)))
        psnr_val_rois = 20 * (torch.log10(255./ torch.sqrt(mse_loss_rois_int8)))
        batch_frame_loss_psnr.append(psnr_val_frames.item())
        batch_roi_loss_psnr.append(psnr_val_rois.item())
        
        
        total_loss = loss_frames + loss_rois 
        

        summary_writer.add_scalar('test frames loss', loss_frames.item())
        summary_writer.add_scalar('test rois loss', loss_rois.item())
        summary_writer.add_scalar('total test loss', total_loss.item())
        
#         batch_frame_loss_l1.append(loss_frames.item())
#         batch_roi_loss_l1.append(loss_rois.item())
#         batch_total_loss.append(total_loss.item())
        if index % 200 == 0 : 
            with open(os.path.join(log_dir, 'test.csv'), 'a+') as f:
                f.writelines('frame_loss:{:.5f}'.format(loss_frames.item()) + \
                     ',' + 'roi_loss:{:.5f}'.format(loss_rois.item()) + ',' + 'total_loss:{:.5f}\n'.format(total_loss.item()))
        
        
        
        if (index % 1000 == 0):
            corr = str(int(p_corr*100))
            save_sample(x, y, m, out_frames, out_rois, testdir, name='{}_{}_{}'.format(epoch, index, corr))
            if test_mode:
                break
        index += 1
        if index % 500 == 0:
            print('File ', index)
#     print('{}/{}'.format(index, len(test_paths)))
    print('Avg Test Frame Loss L1', np.average(batch_frame_loss_l1))
    print('Avg Test ROI Loss L1', np.average(batch_roi_loss_l1))
#     print('Avg Test Total Loss', total_loss.item())

#     print('Avg Test MSE (int 8 ) Loss', np.average(batch_frame_loss_mse_int8))
    
    print('Avg Test Frame MSE Loss', np.average(batch_frame_loss_mse))
    print('Avg Test ROI MSE Loss', np.average(batch_roi_loss_mse))
    
    print('Avg Test Frame MSE Loss (int 8)', np.average(batch_frame_loss_mse_int8))
    print('Avg Test ROI MSE Loss (int 8)', np.average(batch_roi_loss_mse_int8))
    
    print('Avg Test Frame SSIM Loss', np.average(batch_frame_loss_ssim))
    print('Avg Test ROI SSIM Loss', np.average(batch_roi_loss_ssim))
    
    print('Avg Test Frame PSNR Loss', np.average(batch_frame_loss_psnr))
    print('Avg Test ROI PSNR Loss', np.average(batch_roi_loss_psnr))
    print(global_file_count)
#     print('Avg Test PSNR Loss', psnr_val.item())`
    import IPython; IPython.embed()
    
#     return np.average(batch_frame_loss), np.average(batch_roi_loss), np.average(batch_total_loss)

pid = os.getpid()
prev_mem = 0

def get_padded_roi(mouth_frames, v_len):
    out = torch.rand(32, 64, 64,3)
    v_len = len(mouth_frames)
    out[:v_len,:,:,:] = torch.from_numpy(mouth_frames / 255.).float() 
    return out.numpy()
    
def get_roi_frames(blaze_detector, frames):
    img = (frames) * 255
    detections = blaze_detector.net.predict_on_batch(img)
#         for d in detections:
#             print(d.shape)
    mouth_regions, flag = blaze_detector.batch_mouth_detection(img.permute(0,2,3,1), detections)
    return np.array(mouth_regions), flag    

def process_xy(v):
    global p_corr
    if test_mode:
        print(p_corr)
    v_out = torch.clone(v).permute(0, 3, 1, 2)
    v = torch.cat([v, torch.ones(v.shape[0], v.shape[1], v.shape[2], 1)], dim=-1)
    in_frames = v.permute(0, 3, 1, 2)
    out_frames, in_frames = get_padded_frames(in_frames, v_out)
    in_frames = run_corruption_with_mask(in_frames, p_corr)
    
#     in_frames  = run_corruption_prefix_suffix(in_frames)
    return in_frames, out_frames

def check_memory(train_index):
    global pid, prev_mem
    cur_mem = (int(open('/proc/%s/statm'%pid, 'r').read().split()[1])+0.0)/256
    add_mem = cur_mem - prev_mem
    prev_mem = cur_mem
    print("train instances: %s, added mem: %sM"%(train_index, add_mem))


def train_step(model,  optimizerG, train_paths, test_dataloader, detector, summary_writer):
    global p_corr
    batch_f_loss = []
    batch_r_loss = []
    batch_t_loss = []
    file_count = 0
    epoch = 0
    
    global_file_count = 0


    num_iters = len(train_paths) * 10
    
    for train_index in (range(num_iters)):
        
            
        x_list, y_list, roi_list, file_count, file_paths = prepare_batch(train_paths, detector)
        x = torch.cat(x_list, dim=0)
        y = torch.cat(y_list, dim=0)
        m = torch.cat(roi_list, dim=0)
        
        if test_mode:
            print(x.shape, y.shape, m.shape)
        
        global_file_count +=  file_count 
        if global_file_count >= len(train_paths):
            epoch += 1
            global_file_count = file_count % len(train_paths)
        
#         print(x.shape, y.shape, m.shape)
        t_time = time.time()
        
        
        model.train()
        roi_disc.train()
        faces_disc.train()
        for k in range(1):
            optimizerG.zero_grad()
            out_frames, out_rois = model(x)
            
            loss_frames = criterion(out_frames, y) * 10.
            loss_rois = criterion(out_rois, m) * 20.
            

            total_loss = loss_frames + loss_rois 
        
            total_loss.backward()
            optimizerG.step()
        
        print('Batch : {}/{}, Epoch:{}'.format(train_index, len(train_paths), epoch))
        g_losses_msg = 'Train Frame Loss : {:.5f}'.format(loss_frames.item())
        g_losses_msg +=  ' Train ROI Loss : {:.5f}'.format(loss_rois.item())
        g_losses_msg += 'Total loss gen : {:.5f}\n'.format(total_loss.item()) 
        
        print(g_losses_msg)
        print()
        if train_index % 500 == 0:
            with open(os.path.join(log_dir, 'train.csv'), 'a+') as f:
                f.writelines('frame_loss:{:.5f}'.format(loss_frames.item()) + \
                         ',' + 'roi_loss:{:.5f}'.format(loss_rois.item()) + ',' + 'total_loss:{:.5f}\n'.format(total_loss.item()))
            
#         summary_writer.add_scalar('train frame loss', loss_frames.item())
#         summary_writer.add_scalar('train rois loss', loss_rois.item())
#         summary_writer.add_scalar('total train loss', total_loss.item())

            
        gc.collect()

        print('Train step time:', time.time()-t_time)

        if (train_index == len(train_paths)//2) or (train_index % 2000 == 0):
            start = time.time()
            test_frame_loss, test_rois_loss, test_total_loss = test_step(model,  epoch, train_index, test_paths, detector, summary_writer, p_corr)
            print('Test Time:{:.3f}'.format(time.time() - start))
            summary_writer.add_scalar('avg test frames loss', test_frame_loss)
            summary_writer.add_scalar('avg test rois loss', test_rois_loss)
            summary_writer.add_scalar('avg total test loss', test_total_loss)
        
        if (train_index   == (len(train_paths)//2)) or (train_index % 2000 == 0):
            corr = str(int(p_corr*100))
            save_sample(x, y, m, out_frames, out_rois, traindir, name='{}_{}_{}'.format(epoch, train_index, corr))
        
#         if train_index == len(train_dataloader)-1 :
#             pass
            
        if train_index % 2000 == 0: 
            save_model(optimizerG, model, epoch, train_index)
        if train_index % 2500 == 0  and (p_corr < 0.75):
                p_corr += 0.05
                
        if test_mode:
            break
    save_model(optimizerG,  model, epoch, train_index)

################################# Final Training Loop


n_epochs = 50
criterion = nn.L1Loss().to(device)
print('Loading model')
model = FCN3DSTN().to(device)
roi_disc = Discriminator().to(device)
faces_disc = Discriminator()
mouth_detector = MouthDetector(device)
lr= 1e-4

# optimizerD = optim.Adam(list(roi_disc.parameters())+list(faces_disc.parameters()), lr=lr, betas=(0.9, 0.999), weight_decay=0.001)
optimizerG = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.001)
model_path_stn = '../BlazeFace-PyTorch/Viseme_train_outputs_fcn3d_stn_plain_roi_blaze_cnn/2020-09-15 06:26:30.327477_scratch/models/5_102000.pt'

model.load_state_dict(torch.load(model_path_stn, map_location=lambda storage, loc: storage)['model_state_dict'])
print('model loaded')
print('starting test')
# for i in range(n_epochs):
#     print('EPOCH: ', i, '#######################################################')
start = time.time()

summary_writer = SummaryWriter(logdir=log_dir)
test_path_len = len(test_paths)
test_paths = test_paths.tolist()
print('Legths : ', 'train : ', train_path_len, 'test' , test_path_len, 'validation', val_path_len)

epoch = 0
batch = 0

test_step(model, epoch, batch, test_paths, mouth_detector, summary_writer, p_corr)

print('Total Time Time : {:3f}'.format(time.time()- start))

        


