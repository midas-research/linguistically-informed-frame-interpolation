# from dataloader import *
import os
# DEVICE_ID_0 = 3
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
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from random import shuffle
import numpy as np
import pandas as pd
import cv2
from skimage.transform import resize


# with open('./lrs_paths.txt') as f:
#     paths = f.readlines()
# shuffle(paths)
# train_paths = (paths[:int(0.8*len(paths))])
# test_paths = (paths[int(0.8*len(paths)):])

with open('./lrs_eval_paths.txt') as f:
    paths = f.readlines()
shuffle(paths)
test_paths  = paths
test_paths = [path.strip() for path in test_paths]
test_paths = [path for path in test_paths if '.mp4' in path]
# train_paths = (paths[:int(0.8*len(paths))])
# test_paths = (paths[int(0.8*len(paths)):])
train_paths = pd.read_csv('lrs_train_paths.csv')['location']
train_paths = train_paths.tolist()
train_paths = [path for path in train_paths if '.mp4' in path]



class LRSVideoHandler(object):
    def __init__(self, filepaths):
#         self.root_dir = root_dir
        self.paths = filepaths
    
    def read_video_audio(self, video_path, audio_path=None, audio=False):

        frames = [] 
#         print('fetching frames from ', video_path)
#         exit()
        cap = cv2.VideoCapture(video_path.strip())
        
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
#                 frame = cv2.resize(frame, (64, 64), cv2.INTER_CUBIC)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = resize(frame, (64, 64)) 
                
                frames.append(frame)
            else:
#                 print('ret is false')
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
            return video_frames

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
            
class VideoDatasetAE(Dataset):
    def __init__(self, vhandler, vpaths, apaths=None, c_p=0.40, c_vis=False, dyn=True, DIM=64, frame_len=32, mask= True, corr_type='noise_mask', audio=False,res=1):
        super(VideoDatasetAE, self).__init__()
        self.vpaths = vpaths  # video paths
        self.apaths = apaths  # audio paths
        self.cr_vis = c_vis
        self.p_corr = c_p
        self.dyn = dyn
        self.corr_type = corr_type
        self.frame_dims = frame_len
        self.vhandler = vhandler
        self.mask = mask
        self.audio = audio
        self.res = res
        self.c_type = 'suffix'
        
        print('Corruption type', self.c_type)
        
        if self.mask:
            self.channels = 4

            self.EOS_FRAME = torch.rand(self.channels, DIM, DIM)
            self.SOS_FRAME = torch.rand(self.channels, DIM, DIM)
            self.PAD_FRAME = torch.rand(self.channels, DIM, DIM)

            self.EOS_FRAME[-1, :, :] = torch.zeros(DIM,DIM )
            self.SOS_FRAME[-1, :, :] = torch.zeros(DIM,DIM )
            self.PAD_FRAME[-1, :, :] = torch.zeros(DIM,DIM )
    
        else:
            self.channels = 3
        
        
            self.EOS_FRAME = torch.rand(self.channels, DIM, DIM)
            self.SOS_FRAME = torch.rand(self.channels, DIM, DIM)
            self.PAD_FRAME = torch.rand(self.channels, DIM, DIM)
        
        
#         assert len(self.vpaths) == len(self.apaths)
        
    def set_device(self, device):
        self.device = device

    def run_corruption_with_mask(self, frames, p=0.5, vis=True, device=None):
        
        count = 0
        no_corrupt_frames  = int(len(frames) * self.p_corr)
        frame_indices = list(range(len(frames)))
        indices = random.sample(frame_indices, no_corrupt_frames)
        
        for index in indices:
            frames[index,:3] = torch.rand(*frames[index,:3].shape)
            frames[index,-1] = torch.zeros(*frames[index,-1].shape)
        
        return frames
            
    def run_corruption_mid_first_last(self, frames):
        frames = frames.permute(1, 0, 2, 3)
        new_frames = torch.rand(*frames.shape)
        new_frames[0] = frames[0]
        new_frames[frames.shape[0]//2] = frames[frames.shape[0]//2]
        new_frames[-1] = frames[-1]
        return new_frames.permute(1,0,2,3)

    def run_corruption(self, frames, p=0.5, vis=True):
        # print(frames.shape)
        frames = frames.permute(1,0, 2, 3)

        # exit()
        count = 0
        no_corrupt_frames = int(len(frames) * self.p_corr)
        print('corrupting : {} frames'.format(no_corrupt_frames))
        exit()
#        no_corrupt_frames = 27
        frame_indices = list(range(len(frames)))
        done_frames = []
        done = False
        while not done:
        #for i in range(no_corrupt_frames):
            index = random.choice(frame_indices)
            if index not in done_frames:
                done_frames.append(index)
                #frame_indices.remove(frame_indices.index(index))
                frames[index] = torch.rand(*frames[index].shape)
                count +=1 
                if count == (no_corrupt_frames):
                   done = True

        if vis:
            grid_img = make_grid(frames)
            plt.imshow(grid_img.permute(1,2,0)); plt.show()

        return frames.permute(1, 0, 2, 3)
    
    def run_corruption_prefix_suffix(self, frames, c_type):
        
        num_frames = int(len(frames) * self.p_corr)
        indices = list(range(len(frames)))
        if c_type == 'prefix':
            indices = indices[:num_frames]
        else:
            indices = indices[len(indices)-num_frames:]
        for index in indices:
            frames[index,:3] = torch.rand(*frames[index,:3].shape)
            frames[index,-1] = torch.zeros(*frames[index,-1].shape)
        return frames
    
    def run_corruption_repeat(self, frames, vis=False):
        # print(frames.shape)
        frames = frames.permute(1,0, 2, 3)
        # exit()
        no_corrupt_frames = int(round((len(frames)-1) * self.p_corr))

        frame_indices = list(range(1, len(frames)))
        done = False
        done_frames = []
        count = 0
        while not done:

        #for i in range(no_corrupt_frames):
            index = random.choice(frame_indices)
            #frame_indices.remove(frame_indices.index(index))
            if index not in done_frames:
                done_frames.append(index)
                frames[index] = frames[index-1].clone()
                count += 1
            if count == no_corrupt_frames:
                done = True
        if vis:
            grid_img = make_grid(frames)
            plt.imshow(grid_img.permute(1,2,0)); plt.show()

        return frames.permute(1,0, 2, 3)
    
    
    def run_corruption_prefix_suffix(self, frames, c_type):
        
        num_frames = int(len(frames) * self.p_corr)
        indices = list(range(len(frames)))
        if c_type == 'prefix':
            indices = indices[:num_frames]
        else:
            indices = indices[len(indices)-num_frames:]
        for index in indices:
            frames[index,:3] = torch.rand(*frames[index,:3].shape)
            frames[index,-1] = torch.zeros(*frames[index,-1].shape)
        return frames
    

    def __getitem__(self, index):
        
        vpath = self.vpaths[index]
#         t = time()
#         print(vpath)
        if self.audio:
            print('audio')
            apath = self.apaths[index]
            a, v = self.vhandler.read_video_audio(vpath, apath)
        else: 
            v = self.vhandler.read_video_audio(vpath)
#         print(time() - t)
        # exit()
        v = v.float().permute(3,0,1,2) 

        v_out = torch.clone(v).permute(1,0,2,3)
#         print('out shape ', v_out.shape)
#         v_out = generate_padded_frames_dataset(v_out, self.EOS_FRAME[:3], self.SOS_FRAME[:3], self.PAD_FRAME[:3], 3, device='cpu', frames_len=self.frame_dims, mask=False)
    
        v = torch.cat([v, torch.ones(1, v.shape[1], v.shape[2], v.shape[3])], dim=0)
# #         print(v.shape)
#         v_in = generate_padded_frames_dataset(v.permute(1,0,2, 3),self.EOS_FRAME, self.SOS_FRAME, self.PAD_FRAME, self.channels, device='cpu', frames_len=self.frame_dims, mask=self.mask)
        v_out , v_in = get_padded_frames(v.permute(1,0,2, 3), v_out, frames_len=self.frame_dims, device='cpu')
#         print('padded', v_in.shape)
        
#         v_in = self.run_corruption_prefix_suffix(v_in, self.c_type)
#         v_in = self.run_corruption_prefix_suffix(v_in, self.c_type)
#         if self.corr_type == 'noise_mask':
        v_in = self.run_corruption_with_mask(v_in, vis=self.cr_vis)
# #             print('running corr mask ', v_in.shape)
#         elif self.corr_type == 'noise': 
#             v_in = self.run_corruption(v_in, vis=self.cr_vis)
#         elif self.corr_type == 'fml':
#             v_in = self.run_corruption_mid_first_last(v_in)
#         elif self.corr_type == 'rep':
#             v_in = self.run_corruption_repeat(v_in)
#         print(v_in.shape)
#         v_rev = reversed_frames(v_in)
        v_rev = reversed_frames(v_in).permute(1,0,2,3)
        v_in = v_in[:,:,::self.res,::self.res].permute(1,0,2,3)
        v_out = v_out[:,:,::self.res,::self.res].permute(1,0,2,3)
        
        if self.audio:
            return a, vin, v_out, v_rev, vpath
        else:
            return v_in, v_out, v_rev, vpath
        
    def __len__(self):
        return len(self.vpaths)
    

def reversed_frames(frames):
#     print('reversing ####', frames.shape, out.shape)
    out = torch.zeros_like(frames)
#     print()
    for i in range(0, len(frames)):
        out[i] = frames[-(i+1)].clone()
    
#     import IPython; IPython.embed()
#     out[0] = frames[0].clone()
#     out[-1] = frames[-1].clone()
#     for i in range(1, len(frames)-1):
#         out[frames.shape[0]-1-i] = frames[i].clone()
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



class FCN3DBNSkip(nn.Module):
    def __init__(self, in_channels=4):
        
        super(FCN3DBNSkip, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv3d(64, 128, 3, padding=1, stride=2)
        self.conv3 = nn.Conv3d(128, 128, 3, padding=1, stride=2)
        self.conv4 = nn.Conv3d(128, 256, 3, padding=1, stride=2)
        self.conv5 = nn.Conv3d(256, 512, 3, padding=1, stride=2)
        self.conv6 = nn.ConvTranspose3d(512, 256, 3, padding=1, output_padding=1, stride=2)
        self.conv7 = nn.ConvTranspose3d(256+256 , 128, 3, padding=1, output_padding=1, stride=2)
        self.conv8 = nn.ConvTranspose3d(128+128, 64, 3, padding=1, output_padding=1, stride=2)
        self.conv9 = nn.ConvTranspose3d(64+128, 3, 3, padding=1, output_padding=1, stride=2)

        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(256)
        self.bn5 = nn.BatchNorm3d(512)
        self.bn6 = nn.BatchNorm3d(256)
        self.bn7 = nn.BatchNorm3d(128)
        self.bn7 = nn.BatchNorm3d(128)
        self.bn8 = nn.BatchNorm3d(64)
    def forward(self, x):
#         print(self.training)
        out1 = F.dropout3d(self.bn1(F.leaky_relu(self.conv1(x))), p=0.4, training=self.training)
#         print('out1 ', out1.shape)
        out2 = F.dropout3d(self.bn2(F.leaky_relu(self.conv2(out1))), p=0.3, training=self.training)
#         print('out2 ', out2.shape)
        out3 = F.dropout3d(self.bn3(F.leaky_relu(self.conv3(out2))), p=0.3, training=self.training)
#         print('out3 ', out3.shape)
        out4 = F.dropout3d(self.bn4(F.leaky_relu(self.conv4(out3))), p=0.3, training=self.training)
#         print('out4 ', out4.shape)
        out5 = F.dropout3d(self.bn5(F.leaky_relu(self.conv5(out4))), p=0.3, training=self.training)
#         print('out5 ', out5.shape)
        out6 = F.dropout3d(self.bn6(F.leaky_relu(self.conv6(out5))), p=0.3, training=self.training)
#         print('out6 ', out6.shape)
        out6 = torch.cat([out4, out6], dim=1)
        out7 = F.dropout3d(self.bn7(F.leaky_relu(self.conv7(out6))), p=0.3, training=self.training)
        out7 = torch.cat([out3, out7], dim=1)
#         print('out7 ', out7.shape)
        out8 = F.dropout3d(self.bn8(F.leaky_relu(self.conv8(out7))), p=0.3, training=self.training)
        out8 = torch.cat([out2, out8], dim=1)
#         print('out8 ', out8.shape )
        out = F.sigmoid(self.conv9(out8))
        
        return out

class FCN3DBN(nn.Module):
    def __init__(self, in_channels=3):
        
        super(FCN3DBN, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv3d(64, 128, 3, padding=1, stride=2)
        self.conv3 = nn.Conv3d(128, 128, 3, padding=1, stride=2)
        self.conv4 = nn.Conv3d(128, 256, 3, padding=1, stride=2)
        self.conv5 = nn.Conv3d(256, 512, 3, padding=1, stride=2)
        self.conv6 = nn.ConvTranspose3d(512, 256, 3, padding=1, output_padding=1, stride=2)
        self.conv7 = nn.ConvTranspose3d(256, 128, 3, padding=1, output_padding=1, stride=2)
        self.conv8 = nn.ConvTranspose3d(128, 64, 3, padding=1, output_padding=1, stride=2)
        self.conv9 = nn.ConvTranspose3d(64, 1, 3, padding=1, output_padding=1, stride=2)

        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(128)
        self.bn4 = nn.BatchNorm3d(256)
        self.bn5 = nn.BatchNorm3d(512)
        self.bn6 = nn.BatchNorm3d(256)
        self.bn7 = nn.BatchNorm3d(128)
        self.bn7 = nn.BatchNorm3d(128)
        self.bn8 = nn.BatchNorm3d(64)
    def forward(self, x):
#         print(self.training)
        out = F.dropout3d(self.bn1(F.leaky_relu(self.conv1(x))), p=0.4, training=self.training)
        out = F.dropout3d(self.bn2(F.leaky_relu(self.conv2(out))), p=0.3, training=self.training)
        out = F.dropout3d(self.bn3(F.leaky_relu(self.conv3(out))), p=0.3, training=self.training)
        out = F.dropout3d(self.bn4(F.leaky_relu(self.conv4(out))), p=0.3, training=self.training)
        out = F.dropout3d(self.bn5(F.leaky_relu(self.conv5(out))), p=0.3, training=self.training)

        out = F.dropout3d(self.bn6(F.leaky_relu(self.conv6(out))), p=0.3, training=self.training)
        out = F.dropout3d(self.bn7(F.leaky_relu(self.conv7(out))), p=0.3, training=self.training)
        out = F.dropout3d(self.bn8(F.leaky_relu(self.conv8(out))), p=0.3, training=self.training)
        out = F.sigmoid(self.conv9(out))
        
        return out

class FCN3D(nn.Module):
    def __init__(self, in_channels=4):
        
        super(FCN3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, 8, 3, padding=1)
        self.conv2 = nn.Conv3d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv5 = nn.Conv3d(64, 64, 3, padding=1)
#         self.conv6 = nn.Conv3d(64, 128, 3, padding=1)
#         self.conv7 = nn.Conv3d(128, 128, 3, padding=1)
        self.conv8 = nn.Conv3d(64, 128, 3, padding=1)
        self.conv9 = nn.Conv3d(128, 3, 3, padding=1)
    
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x))
        out = F.leaky_relu(self.conv2(out))
        out = F.leaky_relu(self.conv3(out))
        out = F.leaky_relu(self.conv4(out))
        out = F.leaky_relu(self.conv5(out))
        
#         out = F.leaky_relu(self.conv6(out))
#         out = F.leaky_relu(self.conv7(out))
        out = F.leaky_relu(self.conv8(out))
        out = F.sigmoid(self.conv9(out))
        
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = []
        self.model += [
            nn.Conv3d(3, 64, 3, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.4, inplace=True),

            nn.Conv3d(64, 128, 3, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.4, inplace=True),

            nn.Conv3d(128, 128, 3, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.4, inplace=True),

            nn.Conv3d(128, 128, 3, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.4, inplace=True),

            FlattenLayer(),

            nn.Linear(4096, 500),
            nn.LeakyReLU(),
            nn.Dropout(p=0.4, inplace=True),

            nn.Linear(500, 1),
            nn.Sigmoid(),
            # nn.Dropout(p=0.2, inplace=True),

            # PrintLayer()
        ]

        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        out = self.model(x)
        return out


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    
    def forward(self, x):
        return x.view(x.shape[0], -1)


class VNetDec(nn.Module):
    def __init__(self, in_channels=0):
        super(VNetDec, self).__init__()
        self.model = []
        self.model += [
            nn.Conv3d(2048, 1024, 3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(1024),

            nn.Dropout3d(p=0.3, inplace=True),
            nn.ConvTranspose3d(1024, 512, 3, padding=1,
                               output_padding=(0, 1, 1), stride=(1, 2, 2)),
            nn.LeakyReLU(),
            nn.BatchNorm3d(512),

            nn.Dropout3d(p=0.3, inplace=True),
            nn.ConvTranspose3d(512, 256, 3, padding=1,
                               output_padding=(0, 1, 1), stride=(1, 2, 2)),
            nn.LeakyReLU(),
            nn.BatchNorm3d(256),

            nn.Dropout3d(p=0.3, inplace=True),
            nn.ConvTranspose3d(256, 128, 3, padding=(
                1, 1, 1), output_padding=(0, 1, 1), stride=(1, 2, 2)),
            nn.LeakyReLU(),
            nn.BatchNorm3d(128),

            # PrintLayer(),

            nn.Dropout3d(p=0.3, inplace=True),
            nn.ConvTranspose3d(128, 3, 3, padding=(1, 1, 1),
                               output_padding=(0, 3, 3), stride=(1, 4, 4)),
            nn.Sigmoid(),
            #                         nn.Dropout3d(p=0.3, inplace=True),
        ]

        self.model = nn.Sequential(*self.model)
        for m in self.modules():
            if isinstance(m,nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                print('Initializing', m)
                nn.init.xavier_normal_(m.weight)
    def forward(self, v_x):
        out = self.model(v_x)
        return out

#####################################################################################################################################################


class Args():
    def __init__(self, learning_rate_g, learning_rate_d, iter, optim, base_model='resnet', test_mode=False):
        self.learning_rate_g = learning_rate_g
        self.learning_rate_d = learning_rate_d
        
        self.root_dir = base_model + '_FCN3d_LRS_FULL_TRAIN'
        self.log_dir = 'logs'
        self.model_dir = 'models'
        self.sub_ = '_fcn3d_plain'
        if test_mode:
            self.sub_ += '_test_mode'
        self.sub_dir = os.path.join(self.root_dir, str(datetime.now())+self.sub_)
        self.train_out = 'train'
        self.test_out = 'test'

        self.iters = iter
        self.save_iter = 100
        self.save_output_iter = 50
        self.optimizer = optim
        self.test_iter = 100
        self.batch_size = 10
        self.base_model = base_model



    def get_params(self):
        return {'learning_rate_g': self.learning_rate_g, 'learning_rate_d': self.learning_rate_d, 'optimizer':self.optimizer}

    def create_dirs(self):
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        if not os.path.exists(self.sub_dir):
            os.makedirs(self.sub_dir)
        if not os.path.exists(os.path.join(self.sub_dir, self.log_dir)):
            os.makedirs(os.path.join(self.sub_dir, self.log_dir))
        if not os.path.exists(os.path.join(self.sub_dir, self.model_dir)):
            os.makedirs(os.path.join(self.sub_dir, self.model_dir))
        if not os.path.exists(os.path.join(self.sub_dir, self.train_out)):
            os.makedirs(os.path.join(self.sub_dir, self.train_out))
        if not os.path.exists(os.path.join(self.sub_dir, self.test_out)):
            os.makedirs(os.path.join(self.sub_dir, self.test_out))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
class Trainer():
    def __init__(self, iters=100, g_lr=1e-3, d_lr=1e-3, optim='Adam', device=True, test_mode=True, parallel_gpus=True):
        
        self.args = Args(g_lr, d_lr, iters, optim, test_mode=test_mode)
        
        self.parallel_gpu = parallel_gpus
        self.device = torch.device('cuda' if device else 'cpu')
        self.generator = FCN3D().to(self.device)
        print('Generator params ', count_parameters(self.generator))

        self.criterion = nn.MSELoss()
        self.gen_opt = self.get_opt(self.generator, g_lr)
        self.summary_writer = SummaryWriter(os.path.join(self.args.sub_dir, self.args.log_dir))
        self.test_mode = test_mode

        self.set_dataloaders()
        self.args.create_dirs()
        

    def set_base_model(self):
        if self.args.base_model != 'resnet':
            
            self.vgg16 = torchvision.models.vgg16(pretrained=True)
            self.new_base = (list(self.vgg16.children())[:-1])
            self.new_base = nn.Sequential(*self.new_base)
            for p in self.new_base.parameters():
                p.requires_grad = False
            self.new_base = self.new_base[0].to(device)

        else:
            
            resnet152 = torchvision.models.resnet152(pretrained=True) 
            modules=list(resnet152.children())[:-2] 
            self.new_base=nn.Sequential(*modules).to(self.device) 
            for p in self.new_base.parameters(): 
                p.requires_grad = False
                
    def calc_gradient_penalty(self, netD, real_data, fake_data, device, channels=3, LAMBDA=5.0):
        #print real_data.size()
        alpha = torch.rand(real_data.shape[0], channels, 1, 1, 1)
        # print(alpha.shape)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(device)
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        # print(interpolates.shape)
        
        interpolates = interpolates.to(device)
        # interpolates.required_grad = True
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty

    def get_opt(self, model, lr):
        
        if self.args.optimizer == 'Adam':
            return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
        
        elif self.args.optimizer == 'RMSprop':
            return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=0.01)

    def set_dataloaders(self, noise_type='noise'):



        self.vhandler_val = LRSVideoHandler(test_paths)
        self.vhandler_train = LRSVideoHandler(train_paths)
        
        self.vdset_train = VideoDatasetAE(self.vhandler_train, self.vhandler_train.paths)
            
        self.vdset_val = VideoDatasetAE(self.vhandler_val,self.vhandler_val.paths)

        val_sampler = RandomSampler(self.vdset_val)

        self.train_dataloader = DataLoader(self.vdset_train, batch_size=self.args.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.vdset_val, batch_size=self.args.batch_size, sampler=val_sampler)


    def save_model(self, index):
        torch.save(self.generator.state_dict(), os.path.join(self.args.sub_dir, self.args.model_dir, '{}_g.pt'.format(index)))

    def save_out(self, out, loc, index):
        if not os.path.exists(os.path.join(self.args.sub_dir, loc, str(index))):
            os.makedirs(os.path.join(self.args.sub_dir, loc, str(index)))
        for i, o in enumerate(out):
            o = o.cpu().detach().permute(1,2,3,0)
            # print(o.shape)
            # exit()
            for f_index, frame in enumerate(o):

                io.imsave(os.path.join(self.args.sub_dir, loc, str(index),'{}_{}.png'.format(i, f_index)), frame)
        
    def save_img_grid(self, out, loc, index, name='in'):
        loc = os.path.join(self.args.sub_dir, loc, '{}_{}.png'.format(index, name))
        for i, o in enumerate(out):
#             print(o.shape,'in shape')
#             exit()
            o = make_grid(o.detach().permute(1, 0, 2, 3)[:,:3,:,:], normalize=True, scale_each=True)
            save_image(o, loc)


    def save_params(self):
        torch.save({
            'generator_class': FCN3DBNSkip,
            # 'discriminator_class': Discriminator,
            'g_lr': self.args.learning_rate_g,
            # 'd_lr': self.args.learning_rate_d,
            'g_optimizer': self.gen_opt,
            # 'd_optimizer': self.disc_opt 
            }, os.path.join(self.args.sub_dir, self.args.model_dir, 'params'))

    def show_image_grid(self, out):
        out = out[0]
        out = make_grid(out.permute(1, 0 , 2, 3))
        plt.imshow(out.permute(1,2,0)); plt.show()

    def train_step(self):

        

        for batch_idx, (v_in,v_out, _, _) in enumerate(self.train_dataloader):
            
#             print( v_in.shape, v_out.shape)
            if batch_idx == 0 and self.index == 0:
                self.val_dataloader.dataset.p_corr = 0.4
                self.train_dataloader.dataset.p_corr = 0.4

            self.generator.train()
            v_in = v_in.to(self.device)
            v_out = v_out.to(self.device)

            if self.test_mode:
                print(v_in.shape, v_out.shape)

   
            self.gen_opt.zero_grad()
            out = self.generator(v_in)

            loss = 100. * self.criterion(out, v_out)
            loss.backward()
            self.gen_opt.step()
            print('#########################################')
            print('Iter:', self.index, 'Batch idx: {}/{}'.format(batch_idx,len(self.train_dataloader)),  'AE Loss:', loss.item())
            print('#########################################')
            
            self.summary_writer.add_scalar('G Loss', loss.item())
            
            if batch_idx % self.args.save_output_iter == 0:
                loc = 'train'
                p_cr = int(self.train_dataloader.dataset.p_corr * 100)
                self.save_img_grid(out, loc, self.index, name='{}_{}_out'.format(batch_idx, p_cr))
                self.save_img_grid(v_in, loc, self.index, name='{}_{}_in'.format(batch_idx, p_cr))
                self.save_img_grid(v_out, loc, self.index, name='{}_{}_gt'.format(batch_idx, p_cr))
#                 exit()
            if batch_idx  % self.args.save_iter == 0:
                self.save_model(str(self.index)+'_'+str(batch_idx))
                self.test_step(batch_idx)
            if batch_idx % 250 == 0  and (self.train_dataloader.dataset.p_corr < 0.75):
                self.val_dataloader.dataset.p_corr += 0.05
                self.train_dataloader.dataset.p_corr += 0.05
                #pass
            if self.test_mode:
                break

        return v_in, out

    
    def test_step(self, bid):
        print('Testing')
        self.generator.eval()
        outs = []
        test_losses = []
        batch_idx = 0
        for (v_in,v_out, _, _) in tqdm((self.val_dataloader)):
#             print('Testing shapes ', v_in.shape, v_out.shape)
            
            batch_idx += 1
            v_in = v_in.to(self.device)
            v_out = v_out.to(self.device)
          
            
            if self.test_mode:
                print(v_in.shape, v_out.shape)

            with torch.no_grad():
                out = self.generator(v_in)
                t_loss = self.criterion(out, v_out)
                test_losses.append(t_loss.item())
            # if batch_idx  == 3:
            #     break
        print('TEST LOSS:', np.average(test_losses))
        self.summary_writer.add_scalar('Test Loss', np.average(test_losses))

        loc = 'test'
        p_cr = int(self.val_dataloader.dataset.p_corr * 100)
        self.save_img_grid(out, loc, self.index, name='{}_{}_{}_out'.format(self.index, bid, p_cr))
        print('saving_test _image')
        self.save_img_grid(v_in, loc, self.index, name='{}_{}_{}_in'.format(self.index, bid, p_cr))
        self.save_img_grid(v_out, loc, self.index, name='{}_{}_{}_gt'.format(self.index, bid, p_cr))
            # outs.append(out)
            # break
        return test_losses

    def train(self):
        self.save_params()
        if self.test_mode:
            self.args.test_iter = 1
            self.args.save_output_iter = 1
            self.args.save_iter = 1
        for self.index in range(self.args.iters):
            in_img, out = self.train_step()
            # print(out.shape)
            if self.index % self.args.test_iter == 0 and self.index !=0:
                pass
                #self.test_step()

            if self.index % self.args.save_iter == 0:
                self.save_model(self.index)
            
            if self.index % self.args.save_output_iter == 0:
                loc = 'train'

            if self.test_mode:
                exit()
    


# if __name__ == '__main__':
trainer = Trainer(device=True, test_mode=False, optim='Adam') # test mode true checks if a single iteration is working or not
trainer.train()
