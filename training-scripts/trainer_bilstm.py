import os
# DEVICE_ID_0 = 3
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
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
#import ipdb
import numpy as np
from torch.utils.data import DataLoader, Dataset
import cv2
import torchaudio
import math
from time import time
import sys
from random import shuffle
from skimage.transform import resize
import pytorch_ssim 
import IPython

torch.cuda.empty_cache()
torch.set_num_threads(10)


device_index = sys.argv[1]

device_id = 'cuda:{}'.format(int(device_index.strip()))
device = torch.device(device_id)
# device = torch.device('cpu')


torch.manual_seed(123)
np.random.seed(123)
random.seed(123)



# with open('./lrs_eval_paths.txt') as f:
#     paths = f.readlines()
# shuffle(paths)
# test_paths  = paths
# test_paths = [path.strip() for path in test_paths]
# test_paths = [path for path in test_paths if '.mp4' in path]
# train_paths = (paths[:int(0.8*len(paths))])
# test_paths = (paths[int(0.8*len(paths)):])
train_paths = pd.read_csv('lrs_train_paths.csv')['location']
train_paths = train_paths.tolist()
train_paths = [path for path in train_paths if '.mp4' in path]
train_paths = train_paths[:int(len(train_paths)*.80)]
test_paths = train_paths[int(len(train_paths)*.80):]#validation set


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

class LRSVideoHandler(object):
    def __init__(self, filepaths):
#         self.root_dir = root_dir
        self.paths = filepaths
    
    def read_video_audio(self, video_path, audio_path=None, audio=False):

        frames = [] 

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
    def __init__(self, vhandler, vpaths, apaths=None, c_p=0.75, c_vis=False, dyn=True, DIM=64, frame_len=32, mask= True, corr_type='noise_mask', audio=False,res=1):
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
        self.c_type = 'prefix'
        
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

        count = 0
        no_corrupt_frames = int(len(frames) * self.p_corr)
        # print('corrupting
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

#        for i in range(no_corrupt_frames):
#            index = random.choice(frame_indices)
            # print(index, frame_indices)
#            frames[index] = torch.rand(*frames[index].shape)
        
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
        if self.corr_type == 'noise_mask':
            v_in = self.run_corruption_with_mask(v_in, vis=self.cr_vis)
#             print('running corr mask ', v_in.shape)
        elif self.corr_type == 'noise': 
            v_in = self.run_corruption(v_in, vis=self.cr_vis)
        elif self.corr_type == 'fml':
            v_in = self.run_corruption_mid_first_last(v_in)
        elif self.corr_type == 'rep':
            v_in = self.run_corruption_repeat(v_in)
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




train_vhandler = LRSVideoHandler(train_paths)
test_vhandler = LRSVideoHandler(test_paths)


batch_size = 5

dset_train = VideoDatasetAE(train_vhandler, train_vhandler.paths)
dset_test = VideoDatasetAE(test_vhandler, test_vhandler.paths)
# train_sampler = RandomSampler(dset_train)
# test_sampler = RandomSampler(dset_test)

from itertools import cycle

train_dataloader = (DataLoader(dset_train, batch_size=batch_size,shuffle=True, num_workers=1))
test_dataloader = (DataLoader(dset_test, batch_size=batch_size, shuffle=True, num_workers=1))

# IPython.embed()


###################################### MODEL

import torch
import torch.nn as nn 
import torch.nn.functional as F


class ConvTransLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias, stride=1, output_padding=1):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvTransLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.stride = stride
#         self.h_stride = h_stride
        self.hidden_conv = nn.ConvTranspose2d(self.hidden_dim, out_channels=self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias,
                              stride=self.stride,
                              output_padding=output_padding
                                    )
        self.cur_conv = nn.ConvTranspose2d(self.hidden_dim, out_channels= self.hidden_dim*4,
                      kernel_size=self.kernel_size,
                      padding=self.padding,
                      bias=self.bias,
                      stride=self.stride,
                      output_padding=output_padding
                                           
                            )
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias, 
                              stride=1,
#                               output_padding=output_padding
                             )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, input_tensor, cur_state):
        h,c_cur  = cur_state
#         print('input: ', input_tensor.shape, 'hidden :',  h.shape)
        combined = torch.cat([input_tensor, h], dim=1).to(device)
#         print(combined.shape)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
#         print('split shapes ', cc_i.shape, cc_f.shape, cc_o.shape, cc_g.shape)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = F.leaky_relu(cc_g)
#         print(f.shape, c_cur.shape, i.shape, g.shape)
        c_next = f * c_cur + i * g
        h_next = o * F.leaky_relu(c_next)
#         print('done')
        o_next = self.hidden_conv(h_next)
#         c_next = self.cur_conv(c_next)
#         h_next = self.hidden_conv(h_next)
        return h_next, c_next, o_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias, stride=1, h_stride=1):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
#         print(self.input_dim, self.hidden_dim)
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.stride = stride
#         self.h_stride = h_stride
        self.hidden_conv = nn.Conv2d(self.hidden_dim, out_channels=self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias,
                              stride=self.stride
                                    )
        self.cur_conv = nn.Conv2d(self.hidden_dim, out_channels= self.hidden_dim,
                      kernel_size=self.kernel_size,
                      padding=self.padding,
                      bias=self.bias,
                      stride=self.stride
                            )
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias, 
                              stride=1
                             )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, input_tensor, cur_state):
        h,c_cur  = cur_state
#         print('input: ', input_tensor.shape, 'hidden :',  h.shape, c_cur.shape)
#         print(input_tensor.device, h.device)
        combined = torch.cat([input_tensor, h], dim=1).to(device)
#         print(self.input_dim, self.hidden_dim)
#         print('combined.shape', combined.shape)
        combined_conv = self.conv(combined)
        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
#         print('split shapes ', cc_i.shape, cc_f.shape, cc_o.shape, cc_g.shape)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
#         print(f.shape, cur_conved.shape, i.shape, g.shape)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
#         print('done')
        o_next = self.hidden_conv(h_next)
#         print('H: {}  C : {} O: {}'.format(h_next.shape, c_next.shape, o_next.shape))
        return h_next, c_next, o_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width).to(device),
                torch.zeros(batch_size, self.hidden_dim, height, width).to(device))

class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, stride=1, h_stride=1, output_padding=0,
                 batch_first=True, bias=True, return_all_layers=False, transposed=False):
        super(ConvLSTM, self).__init__()
#         print(self.device)
        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.stride = stride
        self.output_padding = output_padding
        self.transposed = transposed
#         print(self.hidden_dim)
        cell_list = []
        for i in range(0, self.num_layers):
            if i == self.num_layers-1 or self.num_layers==1:
                stride = self.stride
                output_padding = self.output_padding
            else:
                stride = 1
                output_padding = 0
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            if not transposed:
                cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias,
                                          stride=stride,

                                         ))
            else:
                cell_list.append(ConvTransLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias,
                                          stride=stride,
                                          output_padding=output_padding,
                                         ))


        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
#         print(self.num_layers)
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        b, _, _, h, w = input_tensor.size()
        hidden_h = h
        hidden_w = w

        
        if hidden_state is not None:
            hidden_state = self._init_hidden_withval(hidden_state)
#             raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(hidden_h, hidden_w))
#             print('hidden generated ' , hidden_state[0][0].shape)
        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            hidden_inner = []
            for t in range(seq_len):
#                 print(t, layer_idx)
                h, c, o = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(o)
                hidden_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c, o])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            hi = self.cell_list[i].init_hidden(batch_size, image_size)
#             print(hi[0].shape, hi[1].shape)
            init_states.append(hi)
        return init_states
    
    def _init_hidden_withval(self, hidden_state):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(hidden_state[0])
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param
    


class SeqEncoder(nn.Module):
    def __init__(self, in_channels=4):
        super(SeqEncoder, self).__init__()
                #     16   32    32    64    64  
        self.f_maps = [64, 128, 32, 16, 16, 64]
        self.conv_lstm1 = ConvLSTM(in_channels, self.f_maps[0], (3,3), 1, 2, batch_first=True, bias=True, return_all_layers=False)
        self.hidden_conv1 = nn.Conv2d(self.f_maps[0], self.f_maps[1], 3, stride=2, padding=1)
        self.cur_conv1 = nn.Conv2d(self.f_maps[0], self.f_maps[1], 3, stride=2, padding=1)
        
        self.conv_lstm2 = ConvLSTM(self.f_maps[0], self.f_maps[1], (3,3), 1, 2, batch_first=True, bias=True, return_all_layers=False)
        
        self.hidden_conv2 = nn.Conv2d(self.f_maps[1], self.f_maps[2], 3, stride=2, padding=1)
        self.cur_conv2 = nn.Conv2d(self.f_maps[1], self.f_maps[2], 3, stride=2, padding=1)
        self.conv_lstm3 = ConvLSTM(self.f_maps[1], self.f_maps[2], (3,3), 1, 1, batch_first=True, bias=True, return_all_layers=False)
        
    def forward(self, x):
#         print('Seq enc', x.shape)
        out, l_state = self.conv_lstm1(x)
        out[0] = F.leaky_relu(out[0])

        h_next = F.leaky_relu(self.hidden_conv1(F.leaky_relu(l_state[0][0])))
        c_next = F.leaky_relu(self.cur_conv1(F.leaky_relu(l_state[0][1])))        
        out, l_state = self.conv_lstm2(F.leaky_relu(out[0]), [(h_next, c_next)])

        h_next = F.leaky_relu(self.hidden_conv2(F.leaky_relu(l_state[0][0])))
        c_next = F.leaky_relu(self.cur_conv2(F.leaky_relu(l_state[0][1])))
#         print('####################')
        out, l_state = self.conv_lstm3(F.leaky_relu(out[0]), [(h_next, c_next)])
    
        return out, l_state
    

class SeqDecoder3(nn.Module):
    def __init__(self, device=None):
        super(SeqDecoder3, self).__init__()
#         print('Decoder device', self.device)
                #     16   32    32    64    64  
        self.f_maps = [64, 3, 32, 16, 16, 64]
        self.conv_lstm1 = ConvLSTM(32, self.f_maps[2], (3,3), 1, stride=2, batch_first=True, bias=True, return_all_layers=False, transposed=True, output_padding=1)
        self.hidden_conv1 = nn.ConvTranspose2d(self.f_maps[2], self.f_maps[1], 3, stride=2, padding=1, output_padding=1)
        self.cur_conv1 = nn.ConvTranspose2d(self.f_maps[2], self.f_maps[1], 3, stride=2, padding=1, output_padding=1)
        
        self.conv_lstm2 = ConvLSTM(self.f_maps[2], 3, (3,3), 2, stride=2, batch_first=True, bias=True, return_all_layers=False, transposed=True, output_padding=1)
        
 
        self.conv1 = nn.Conv2d(64, 32, 3, padding=1, stride=1)
        self.conv2 = nn.ConvTranspose2d(32, 64, 3, padding=1, stride=1)
        self.dropout0 = nn.Dropout2d()
        self.dropout1 = nn.Dropout2d()
        self.batchnorm0 = nn.BatchNorm2d(self.f_maps[1])
        self.batchnorm1 = nn.BatchNorm2d(self.f_maps[1])
                         
        
    def forward(self, x1, x2, z):
#         print('Decoder', x1.shape, x2.shape, z.shape)
        x = torch.cat([x1, x2], dim=1)
#         print(x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
#         x = x.unsqueeze(0)
#         print(x.shape)
        
        x1 = x[:, :x.shape[1]//2, :, :]
        x2 = x[:,  x.shape[1]//2:, :, :]
#         print(x1.shape, x2.shape, z.shape)
#         print('Z:', z.shape, 'X1: ',  x1.shape, x2.shape)
        out, l_state = self.conv_lstm1(z, [(x1, x2)])
        
#         print('LSTM 1 passed')
#         print(out[0][0].shape, l_state[0][0].shape)
#         print(l_state[0][0].shape)
        h_next = self.dropout0(self.batchnorm0(F.leaky_relu(self.hidden_conv1(F.leaky_relu(l_state[0][0])))))
        c_next = self.dropout1(self.batchnorm1(F.leaky_relu(self.cur_conv1(F.leaky_relu(l_state[0][1])))))        
#         print('L ', l_state[0][1].shape, l_state[0][0].shape)
#         import IPython; IPython.embed()
#         print('out[0].shape', out[0].shape, h_next.shape, c_next.shape)
        out, l_state = self.conv_lstm2(F.leaky_relu(out[0]), [(h_next, c_next)])
        
        
        out[0] = torch.sigmoid(out[0])
        
        return out, l_state
    
    
class SeqDecoder(nn.Module):
    def __init__(self, device=None):
        super(SeqDecoder, self).__init__()
#         print('Decoder device', self.device)
                #     16   32    32    64    64  
        self.f_maps = [32, 3, 32, 16, 16, 64]
        self.conv_lstm1 = ConvLSTM(3, self.f_maps[2], (3,3), 1, stride=2, batch_first=True, bias=True, return_all_layers=False, transposed=True, output_padding=1)
        self.hidden_conv1 = nn.ConvTranspose2d(self.f_maps[2], self.f_maps[1], 3, stride=2, padding=1, output_padding=1)
        self.cur_conv1 = nn.ConvTranspose2d(self.f_maps[2], self.f_maps[1], 3, stride=2, padding=1, output_padding=1)
        
        self.conv_lstm2 = ConvLSTM(self.f_maps[2], 3, (3,3), 1, stride=2, batch_first=True, bias=True, return_all_layers=False, transposed=True, output_padding=1)
        
        self.hidden_conv2 = nn.Conv2d(self.f_maps[2], self.f_maps[2], 3, stride=2, padding=1)
        self.cur_conv2 = nn.Conv2d(self.f_maps[2], self.f_maps[2], 3, stride=2, padding=1)
        self.conv_lstm3 = ConvLSTM(self.f_maps[1], self.f_maps[2], (3,3), 1, 1, batch_first=True, bias=True, return_all_layers=False)
        
        self.conv1 = nn.ConvTranspose2d(64, 32, 3, padding=1, stride=2, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(32, 64, 3, padding=1, stride=2, output_padding=1)
        self.dropout0 = nn.Dropout2d()
        self.dropout1 = nn.Dropout2d()
        self.batchnorm0 = nn.BatchNorm2d(self.f_maps[1])
        self.batchnorm1 = nn.BatchNorm2d(self.f_maps[1])
                         
        
    def forward(self, x1, x2, z):
#         print('Decoder', x1.shape, x2.shape, z.shape)
        x = torch.cat([x1, x2], dim=1)
#         print(x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
#         x = x.unsqueeze(0)
#         print(x.shape)
        
        x1 = x[:, :x.shape[1]//2, :, :]
        x2 = x[:,  x.shape[1]//2:, :, :]
        
#         print('Z:', z.shape, 'X1: ',  x1.shape)
        out, l_state = self.conv_lstm1(z, [(x1, x2)])
        
#         print('LSTM 1 passed')
#         print(out[0][0].shape, l_state[0][0].shape)

        h_next = self.dropout0(self.batchnorm0(F.leaky_relu(self.hidden_conv1(F.leaky_relu(l_state[0][0])))))
        c_next = self.dropout1(self.batchnorm1(F.leaky_relu(self.cur_conv1(F.leaky_relu(l_state[0][1])))))        
#         print('L ', l_state[0][1].shape, l_state[0][0].shape)
#         import IPython; IPython.embed()
#         print('out[0].shape', out[0].shape, h_next.shape, c_next.shape)
        out, l_state = self.conv_lstm2(F.leaky_relu(out[0]), [(h_next, c_next)])
        
        out[0] = torch.sigmoid(out[0])
# #         print(out[0].shape, l_state[0][0].shape)
#         h_next = F.leaky_relu(self.hidden_conv2(F.leaky_relu(l_state[0][0])))
#         c_next = F.leaky_relu(self.cur_conv2(F.leaky_relu(l_state[0][1])))
# #         print('####################')
#         out, l_state = self.conv_lstm3(F.leaky_relu(out[0]), [(h_next, c_next)])
    
#         return out, l_state
        
        return out, l_state
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.f_encoder = SeqEncoder()
#         self.b_encoder = SeqEncoder()
        
        self.decoder = SeqDecoder3()
    
    def forward(self, x_, x_rev):
        x_f, state_f = self.f_encoder(x_)
        x_b, state_b = self.f_encoder(x_rev)
        
#         print('##################',state_f[-1][-1].shape, state_b[-1][-1].shape, x_f[0].shape, x_b[0].shape)
        out, state = self.decoder(state_f[-1][-1], state_b[-1][-1], x_f[0] + x_b[0])
#         out, state = self.decoder(state_f[-1][-1], state_b[-1][-1], z)
#         import IPython; IPython.embed()
#         out, state = self.decoder(x_f[0], x_b[0])
        return out, state

#############################################
iters = 20

test_mode = True

# In[17]:

################################# CREATE DIRS
root_dir = 'train_outputs_lstm_recon'
now = str(datetime.now()) + '_resumed_scratch'
if test_mode:
    now += '_test' 
    
subdir = os.path.join(root_dir, now)


logdir = os.path.join(subdir, 'logs')
traindir = os.path.join(subdir, 'train_outputs')
testdir = os.path.join(subdir, 'test_outputs')
modeldir = os.path.join(subdir, 'models')
out_dirs = [root_dir, subdir, logdir, traindir, testdir, modeldir]
for dir_ in out_dirs:
    if not os.path.exists(dir_):
        os.makedirs(dir_)

def save_sample(vin, v_rev, v_out, out,  dir_, name):
    def normalize_img(in_arr):
        in_arr = in_arr - in_arr.min() / (in_arr.max() - in_arr.min())
        in_arr *=255
        return in_arr
    
#     print(x.shape, y.shape, m.shape, o_x.shape, o_m.shape)
    out_vin = normalize_img(make_grid(vin[:3].permute(1,0,2,3)).permute(1,2,0)).cpu().numpy().astype(np.uint8)
    out_vrev = normalize_img(make_grid(v_rev[:3].permute(1,0,2,3)).permute(1,2,0)).cpu().numpy().astype(np.uint8)
    mask_x = make_grid(vin[-1].unsqueeze(0).repeat(3,1,1,1).permute(1,0,2,3)).permute(1,2,0).cpu().numpy().astype(np.uint8) * 255
    mask_x_rev = make_grid(v_rev[-1].unsqueeze(0).repeat(3,1,1,1).permute(1,0,2,3)).permute(1,2,0).cpu().numpy().astype(np.uint8) * 255

    out_y = normalize_img(make_grid(v_out.permute(1,0,2,3))).permute(1,2,0).cpu().numpy().astype(np.uint8)

    o_frames = normalize_img(make_grid(out)).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
    
    
    io.imsave(os.path.join(dir_, name+'_in.png'), out_vin)
    io.imsave(os.path.join(dir_, name+'_in_rev.png'), out_vrev)
    io.imsave(os.path.join(dir_, name+'_in_mask.png'), mask_x)
    io.imsave(os.path.join(dir_, name+'_in_mask_rev.png'), mask_x_rev)
    io.imsave(os.path.join(dir_, name+'_gt.png'), out_y)
    io.imsave(os.path.join(dir_, name+'_out.png'), o_frames)

def save_img_grid(out, loc, index, name='in'):
    loc_img = os.path.join( loc, '{}_{}.png'.format(index, name))
    loc_mask = os.path.join( loc, '{}_{}_mask.png'.format(index, name))
#     print(out.shape)
    for i, o in enumerate(out):
        out = make_grid(o[:,:3,:,:].detach(), normalize=True, scale_each=True)
        if 'in' in name:
            mask = make_grid(o[:,-1,:,:].unsqueeze(1).detach())
            save_image(mask, loc_mask)
        save_image(out, loc_img)


def save_model(optimizer, model, epoch, batch):
    torch.save({'epoch': epoch, 'batch': batch, 'model_state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, os.path.join(modeldir, '{}_{}.pt'.format(epoch, batch)))


summary_writer = SummaryWriter(log_dir=logdir)
num_train_batches = len(train_dataloader)
num_test_batches = len(test_dataloader)

print('train batches :', num_train_batches)
print('test batches :' , num_test_batches)

# train_dataloader = cycle(iter(train_dataloader))
# test_dataloader = cycle(iter(test_dataloader))


model = Generator().to(device)
# model_path = '/train_outputs_lstm_recon/2020-08-30 18:40:25.370046_resumed_scratch/models/0_1500.pt'
# model.load_state_dict(torch.load(model_path)['model_state_dict'])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.SmoothL1Loss()

def count_parameters(model):
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

count_parameters(model)

def test_step(index, model):
    test_losses = []
    for batch,(vin, vout, v_rev, paths) in enumerate(test_dataloader):
        vin = vin.to(device).permute(0,2,1,3,4)
        vout = vout.to(device).permute(0,2,1,3,4)
        v_rev = v_rev.to(device).permute(0,2,1,3,4)

        model.eval()
        curr_cent = int(test_dataloader.dataset.p_corr*100)
        with torch.no_grad():
            out, state = model(vin, v_rev)
            loss = criterion(out[0], vout)
#         print('test vid path', paths[0])
        save_sample(vin[0].permute(1,0,2,3), v_rev[0].permute(1,0,2,3), vout[0].permute(1,0,2,3), out[0][0], testdir, 'train_{}_{}_{}'.format(curr_cent, train_index, batch))
        test_losses.append(loss.item())
        summary_writer.add_scalar('test sample loss', loss.item())
        
    return test_losses
        


for train_index in range(iters):
    for batch, (vin, vout, v_rev, path) in enumerate(train_dataloader):
         
        vin = vin.to(device).permute(0,2,1,3,4)
        vout = vout.to(device).permute(0,2,1,3,4)
        v_rev = v_rev.to(device).permute(0,2,1,3,4)
        print(vin.shape, vout.shape, v_rev.shape)
        print()
        

        model.train()
        optimizer.zero_grad()
        out, state = model(vin, v_rev)
        loss = criterion(out[0], vout)
        loss.backward()
        optimizer.step()

        summary_writer.add_scalar('train_loss', loss.item())
        print('Train batch loss :', loss.item(), 'Epoch :', train_index, 'Batch : {}/{}'.format(batch, num_train_batches))


        if batch % 250 == 0:
            print('Saving images')

            curr_cent = int(train_dataloader.dataset.p_corr*100)
            save_sample(vin[0].permute(1,0,2,3), v_rev[0].permute(1,0,2,3), vout[0].permute(1,0,2,3), out[0][0], traindir, 'train_{}_{}_{}'.format(curr_cent, train_index, batch))


        if batch % 250 == 0  and (train_dataloader.dataset.p_corr < 0.75):
            train_dataloader.dataset.p_corr += 0.05
            test_dataloader.dataset.p_corr += 0.05

        if batch % 500 == 0:
            test_loss = test_step(train_index, model)
            save_model(optimizer, model, train_index, batch)
            print('TEST LOSS : ', test_loss)
        if test_mode:
            break
    if test_mode:
        break
    
