import torch
import torch.nn as nn
import torch.nn.functional as  F

class FCN3DSTN(nn.Module):
    def __init__(self, in_channels=4):
        
        super(FCN3DSTN, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, 8, 3, padding=1)
        self.conv2 = nn.Conv3d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv4 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv5 = nn.Conv3d(64, 64, 3, padding=1)
#         self.conv6 = nn.Conv3d(64, 128, 3, padding=1)
#         self.conv7 = nn.Conv3d(128, 128, 3, padding=1)
        self.conv8 = nn.Conv3d(64, 128, 3, padding=1)
        self.conv9 = nn.Conv3d(128, 3, 3, padding=1)
    
        self.fc_loc = nn.Sequential(
            nn.Conv3d(3, 128, 1, stride=4, padding=2),
            nn.LeakyReLU(),
            nn.Conv3d(128, 3, 3, stride=4, padding=1),
            nn.LeakyReLU(),
            FlattenLayer(),
#             PrintLayer(),
            nn.Linear( 225, 512),
            nn.ReLU(True),
            nn.Linear(512, 3 * 4)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0], dtype=torch.float))
        
    def stn(self, x):

        theta = self.fc_loc(x)
        theta = theta.view(-1, 3, 4)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
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
        
        out_roi = self.stn(out)
        
        return out, out_roi
    
class FCN3D(nn.Module):
    def __init__(self, in_channels=4):
        
        super(FCN3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv3d(64, 128, 3, padding=1)
        self.decode_frames = [nn.Conv3d(128, 64, 3, padding=1),
                              nn.LeakyReLU(),
                              nn.Conv3d(64, 128, 3, padding=1),
                              nn.LeakyReLU(),
                              nn.Conv3d(128, 3, 3, padding=1),
                              nn.Sigmoid()
                             ]
        self.decode_frames = nn.Sequential(*self.decode_frames)
        
        self.decode_rois = [
            nn.Conv3d(128, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(64, 128, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(128, 3, 3, padding=1),
            nn.Sigmoid()
        ]
        
        self.decode_rois = nn.Sequential(*self.decode_rois)
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out_frames = self.decode_frames(out)
        out_rois = self.decode_rois(out)
        
        return out_frames, out_rois



class FCN3DBN(nn.Module):
    def __init__(self, in_channels=4):
        
        super(FCN3DBN, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv3d(32, 64, 3, padding=1, stride=2)
        self.conv4 = nn.Conv3d(64, 128, 3, padding=1, stride=2)
        self.conv5 = nn.Conv3d(128, 256, 3, padding=1, stride=2)
        self.conv6 = nn.ConvTranspose3d(256, 256, 3, padding=1, output_padding=1, stride=2)
        self.conv7 = nn.ConvTranspose3d(256, 128, 3, padding=1, output_padding=1, stride=2)
        self.conv8 = nn.ConvTranspose3d(128, 64, 3, padding=1, output_padding=1, stride=2)
        self.conv9 = nn.ConvTranspose3d(64, 3, 3, padding=1, output_padding=1, stride=2)
        
        
        self.m_conv6 = nn.ConvTranspose3d(256, 256, 3, padding=1, output_padding=1, stride=2)
        self.m_conv7 = nn.ConvTranspose3d(256, 128, 3, padding=1, output_padding=1, stride=2)
        self.m_conv8 = nn.ConvTranspose3d(128, 64, 3, padding=1, output_padding=1, stride=2)
        self.m_conv9 = nn.ConvTranspose3d(64, 3, 3, padding=1, output_padding=1, stride=2)

        self.bn1 = nn.BatchNorm3d(16)
        self.bn2 = nn.BatchNorm3d(32)
        self.bn3 = nn.BatchNorm3d(64)
        self.bn4 = nn.BatchNorm3d(128)
        self.bn5 = nn.BatchNorm3d(256)
        self.bn6 = nn.BatchNorm3d(256)
        self.bn7 = nn.BatchNorm3d(128)
        self.bn7 = nn.BatchNorm3d(128)
        self.bn8 = nn.BatchNorm3d(64)
        
        self.m_bn6 = nn.BatchNorm3d(256)
        self.m_bn7 = nn.BatchNorm3d(128)
        self.m_bn7 = nn.BatchNorm3d(128)
        self.m_bn8 = nn.BatchNorm3d(64)
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                print('initializing ', m)
                nn.init.kaiming_normal_(m.weight)
    def forward(self, x):
#         print(self.training)
        out = F.dropout3d(self.bn1(F.leaky_relu(self.conv1(x))), p=0.4, training=self.training)
        out = F.dropout3d(self.bn2(F.leaky_relu(self.conv2(out))), p=0.3, training=self.training)
        out = F.dropout3d(self.bn3(F.leaky_relu(self.conv3(out))), p=0.3, training=self.training)
        out = F.dropout3d(self.bn4(F.leaky_relu(self.conv4(out))), p=0.3, training=self.training)
        out = F.dropout3d(self.bn5(F.leaky_relu(self.conv5(out))), p=0.3, training=self.training)
        
        
        out2 = F.dropout3d(self.m_bn6(F.leaky_relu(self.m_conv6(out))), p=0.3, training=self.training)
        out2 = F.dropout3d(self.m_bn7(F.leaky_relu(self.m_conv7(out2))), p=0.3, training=self.training)
        out2 = F.dropout3d(self.m_bn8(F.leaky_relu(self.m_conv8(out2))), p=0.3, training=self.training)
        out2 = F.sigmoid(self.conv9(out2))
        
        out = F.dropout3d(self.bn6(F.leaky_relu(self.conv6(out))), p=0.3, training=self.training)
        out = F.dropout3d(self.bn7(F.leaky_relu(self.conv7(out))), p=0.3, training=self.training)
        out = F.dropout3d(self.bn8(F.leaky_relu(self.conv8(out))), p=0.3, training=self.training)
        out = F.sigmoid(self.conv9(out))
        
        
        return out, out2
    
class TransformerFCN3d(nn.Module):
    def __init__(self, in_channels=4):
        
        super(TransformerFCN3d, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv3d(64, 128, 3, padding=1)
        self.decode_frames = [nn.Conv3d(128, 64, 3, padding=1),
                              nn.LeakyReLU(),
                              nn.Conv3d(64, 128, 3, padding=1),
                              nn.LeakyReLU(),
                              nn.Conv3d(128, 3, 3, padding=1),
                              nn.Sigmoid()
                             ]
        self.decode_frames = nn.Sequential(*self.decode_frames)
        

        self.fc_loc = nn.Sequential(
            nn.Linear(3 * 64 *64 * 32, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 4)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
#         xs = self.decode_frames(x)
#         print(xs.shape)
        xs = x.view(x.shape[0], -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 3, 4)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out_frames = self.decode_frames(out)
        out_rois = self.stn(out_frames)
        
        return out_frames, out_rois

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.view(x.shape[0], -1)
    

class Print(nn.Module):
    def __init__(self, tensor=False):
        super().__init__()
        self.display_tensor = tensor
    def forward(self, x):
        print(x.shape)
        if self.display_tensor:
            print(x)
        return x

class FCN3DBNSTN(nn.Module):
    def __init__(self, in_channels=4):
        
        super(FCN3DBNSTN, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv3d(32, 64, 3, padding=1, stride=2)
        self.conv4 = nn.Conv3d(64, 128, 3, padding=1, stride=2)
        self.conv5 = nn.Conv3d(128, 512, 3, padding=1, stride=2)
        self.conv6 = nn.ConvTranspose3d(512, 256, 3, padding=1, output_padding=1, stride=2)
        self.conv7 = nn.ConvTranspose3d(256, 128, 3, padding=1, output_padding=1, stride=2)
        self.conv8 = nn.ConvTranspose3d(128, 64, 3, padding=1, output_padding=1, stride=2)
        self.conv9 = nn.ConvTranspose3d(64, 3, 3, padding=1, output_padding=1, stride=2)
        
        
        self.m_conv6 = nn.ConvTranspose3d(512, 256, 3, padding=1, output_padding=1, stride=2)
        self.m_conv7 = nn.ConvTranspose3d(256, 128, 3, padding=1, output_padding=1, stride=2)
        self.m_conv8 = nn.ConvTranspose3d(128, 64, 3, padding=1, output_padding=1, stride=2)
        self.m_conv9 = nn.ConvTranspose3d(64, 64, 3, padding=1, output_padding=1, stride=2)
        self.m_conv9 = nn.ConvTranspose3d(64, 3, 3, padding=1, output_padding=1, stride=2)

        self.bn1 = nn.BatchNorm3d(16)
        self.bn2 = nn.BatchNorm3d(32)
        self.bn3 = nn.BatchNorm3d(64)
        self.bn4 = nn.BatchNorm3d(128)
        self.bn5 = nn.BatchNorm3d(512)
        self.bn6 = nn.BatchNorm3d(256)
        self.bn7 = nn.BatchNorm3d(128)
        self.bn7 = nn.BatchNorm3d(128)
        self.bn8 = nn.BatchNorm3d(64)
        
        self.m_bn6 = nn.BatchNorm3d(256)
        self.m_bn7 = nn.BatchNorm3d(128)
        self.m_bn7 = nn.BatchNorm3d(128)
        self.m_bn8 = nn.BatchNorm3d(64)
        self.fc_loc = nn.Sequential(
            nn.Conv3d(3, 3, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(3, 3, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            Flatten(),
#             Print(),
            nn.Linear(6144 , 512),
            nn.ReLU(True),
            nn.Linear(512, 3 * 4)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0], dtype=torch.float))
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                print('initializing conv', m)
                nn.init.xavier_uniform_(m.weight)
            
            if isinstance(m, nn.BatchNorm3d):
                print('initializing batchnorm ', m)
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
#                 m.bias.data.fill_(0.01)
    def stn(self, x):
#         xs = self.decode_frames(x)
#         print(xs.shape)
#         print(x.shape)
        
#         xs = x.view(x.shape[0], -1)
#         print(xs.shape, self.fc_loc[0].weight.shape)
        theta = self.fc_loc(x)
        theta = theta.view(-1, 3, 4)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
    
    def forward(self, x):
#         print(self.training)
        out = F.dropout3d(self.bn1(F.leaky_relu(self.conv1(x))), p=0.4, training=self.training)
        out = F.dropout3d(self.bn2(F.leaky_relu(self.conv2(out))), p=0.3, training=self.training)
        out = F.dropout3d(self.bn3(F.leaky_relu(self.conv3(out))), p=0.3, training=self.training)
        out = F.dropout3d(self.bn4(F.leaky_relu(self.conv4(out))), p=0.3, training=self.training)
        out = F.dropout3d(self.bn5(F.leaky_relu(self.conv5(out))), p=0.3, training=self.training)
        
#         out2 = self.stn(out2)
#         out2 = F.dropout3d(self.m_bn6(F.leaky_relu(self.m_conv6(out))), p=0.3, training=self.training)
#         out2 = F.dropout3d(self.m_bn7(F.leaky_relu(self.m_conv7(out2))), p=0.3, training=self.training)
#         out2 = F.dropout3d(self.m_bn8(F.leaky_relu(self.m_conv8(out2))), p=0.3, training=self.training)
#         out_rois = F.sigmoid(self.conv9(out2))
        
        out = F.dropout3d(self.bn6(F.leaky_relu(self.conv6(out))), p=0.3, training=self.training)
        out = F.dropout3d(self.bn7(F.leaky_relu(self.conv7(out))), p=0.3, training=self.training)
        out = F.dropout3d(self.bn8(F.leaky_relu(self.conv8(out))), p=0.3, training=self.training)
        out_frames = F.sigmoid(self.conv9(out))
        
        print('out_frames.shape', out_frames.shape)
        out_rois = self.stn(out_frames)
        
        
        return out_frames, out_rois
    
    
class FCN3DBN2Stream(nn.Module):
    def __init__(self, in_channels=4):
        
        super(FCN3DBNSTN2Stream, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv3d(32, 64, 3, padding=1, stride=2)
        self.conv4 = nn.Conv3d(64, 128, 3, padding=1, stride=2)
        self.conv5 = nn.Conv3d(128, 512, 3, padding=1, stride=2)
        self.conv6 = nn.ConvTranspose3d(512, 256, 3, padding=1, output_padding=1, stride=2)
        self.conv7 = nn.ConvTranspose3d(256, 128, 3, padding=1, output_padding=1, stride=2)
        self.conv8 = nn.ConvTranspose3d(128, 64, 3, padding=1, output_padding=1, stride=2)
        self.conv9 = nn.ConvTranspose3d(64, 3, 3, padding=1, output_padding=1, stride=2)
        
        
        self.m_conv6 = nn.ConvTranspose3d(512, 256, 3, padding=1, output_padding=1, stride=2)
        self.m_conv7 = nn.ConvTranspose3d(256, 128, 3, padding=1, output_padding=1, stride=2)
        self.m_conv8 = nn.ConvTranspose3d(128, 64, 3, padding=1, output_padding=1, stride=2)
        self.m_conv9 = nn.ConvTranspose3d(64, 64, 3, padding=1, output_padding=1, stride=2)
        self.m_conv9 = nn.ConvTranspose3d(64, 3, 3, padding=1, output_padding=1, stride=2)

        self.bn1 = nn.BatchNorm3d(16)
        self.bn2 = nn.BatchNorm3d(32)
        self.bn3 = nn.BatchNorm3d(64)
        self.bn4 = nn.BatchNorm3d(128)
        self.bn5 = nn.BatchNorm3d(512)
        self.bn6 = nn.BatchNorm3d(256)
        self.bn7 = nn.BatchNorm3d(128)
        self.bn7 = nn.BatchNorm3d(128)
        self.bn8 = nn.BatchNorm3d(64)
        
        self.m_bn6 = nn.BatchNorm3d(256)
        self.m_bn7 = nn.BatchNorm3d(128)
        self.m_bn7 = nn.BatchNorm3d(128)
        self.m_bn8 = nn.BatchNorm3d(64)
        self.fc_loc = nn.Sequential(
            nn.Conv3d(512, 128, 1, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(128, 3, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            Flatten(),
            Print(),
            nn.Linear(432 , 512),
            nn.ReLU(True),
            nn.Linear(512, 3 * 4)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0], dtype=torch.float))
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                print('initializing conv', m)
                nn.init.xavier_uniform_(m.weight)
            
            if isinstance(m, nn.BatchNorm3d):
                print('initializing batchnorm ', m)
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
#                 m.bias.data.fill_(0.01)
    def stn(self, x):
#         xs = self.decode_frames(x)
#         print(xs.shape)
#         print(x.shape)
        
#         xs = x.view(x.shape[0], -1)
#         print(xs.shape, self.fc_loc[0].weight.shape)
        theta = self.fc_loc(x)
        theta = theta.view(-1, 3, 4)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
    
    def forward(self, x):
#         print(self.training)
        out = F.dropout3d(self.bn1(F.leaky_relu(self.conv1(x))), p=0.4, training=self.training)
        out = F.dropout3d(self.bn2(F.leaky_relu(self.conv2(out))), p=0.3, training=self.training)
        out = F.dropout3d(self.bn3(F.leaky_relu(self.conv3(out))), p=0.3, training=self.training)
        out = F.dropout3d(self.bn4(F.leaky_relu(self.conv4(out))), p=0.3, training=self.training)
        out = F.dropout3d(self.bn5(F.leaky_relu(self.conv5(out))), p=0.3, training=self.training)
        
#         out2 = self.stn(out)
        out2 = F.dropout3d(self.m_bn6(F.leaky_relu(self.m_conv6(out))), p=0.3, training=self.training)
        out2 = F.dropout3d(self.m_bn7(F.leaky_relu(self.m_conv7(out2))), p=0.3, training=self.training)
        out2 = F.dropout3d(self.m_bn8(F.leaky_relu(self.m_conv8(out2))), p=0.3, training=self.training)
        out_rois = F.sigmoid(self.conv9(out2))
        
        out = F.dropout3d(self.bn6(F.leaky_relu(self.conv6(out))), p=0.3, training=self.training)
        out = F.dropout3d(self.bn7(F.leaky_relu(self.conv7(out))), p=0.3, training=self.training)
        out = F.dropout3d(self.bn8(F.leaky_relu(self.conv8(out))), p=0.3, training=self.training)
        out_frames = F.sigmoid(self.conv9(out))
        
#         print('out_frames.shape', out_frames.shape)
#         out_rois = self.stn(out_frames)
        
        
        return out_frames, out_rois
    
class FCN3DBNSTN2Stream(nn.Module):
    def __init__(self, in_channels=4):
        
        super(FCN3DBNSTN2Stream, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, 3, padding=1, stride=2)
        self.conv3 = nn.Conv3d(32, 64, 3, padding=1, stride=2)
        self.conv4 = nn.Conv3d(64, 128, 3, padding=1, stride=2)
        self.conv5 = nn.Conv3d(128, 512, 3, padding=1, stride=2)
        self.conv6 = nn.ConvTranspose3d(512, 256, 3, padding=1, output_padding=1, stride=2)
        self.conv7 = nn.ConvTranspose3d(256, 128, 3, padding=1, output_padding=1, stride=2)
        self.conv8 = nn.ConvTranspose3d(128, 64, 3, padding=1, output_padding=1, stride=2)
        self.conv9 = nn.ConvTranspose3d(64, 3, 3, padding=1, output_padding=1, stride=2)
        
        
        self.m_conv6 = nn.ConvTranspose3d(512, 256, 3, padding=1, output_padding=1, stride=2)
        self.m_conv7 = nn.ConvTranspose3d(256, 128, 3, padding=1, output_padding=1, stride=2)
        self.m_conv8 = nn.ConvTranspose3d(128, 64, 3, padding=1, output_padding=1, stride=2)
        self.m_conv9 = nn.ConvTranspose3d(64, 64, 3, padding=1, output_padding=1, stride=2)
        self.m_conv9 = nn.ConvTranspose3d(64, 3, 3, padding=1, output_padding=1, stride=2)

        self.bn1 = nn.BatchNorm3d(16)
        self.bn2 = nn.BatchNorm3d(32)
        self.bn3 = nn.BatchNorm3d(64)
        self.bn4 = nn.BatchNorm3d(128)
        self.bn5 = nn.BatchNorm3d(512)
        self.bn6 = nn.BatchNorm3d(256)
        self.bn7 = nn.BatchNorm3d(128)
        self.bn7 = nn.BatchNorm3d(128)
        self.bn8 = nn.BatchNorm3d(64)
        
        self.m_bn6 = nn.BatchNorm3d(256)
        self.m_bn7 = nn.BatchNorm3d(128)
        self.m_bn7 = nn.BatchNorm3d(128)
        self.m_bn8 = nn.BatchNorm3d(64)
        self.fc_loc = nn.Sequential(
            nn.Conv3d(512, 128, 1, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(128, 3, 3, stride=1, padding=1),
            nn.LeakyReLU(),
            Flatten(),
            Print(),
            nn.Linear(432 , 512),
            nn.ReLU(True),
            nn.Linear(512, 3 * 4)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0], dtype=torch.float))
        
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                print('initializing conv', m)
                nn.init.xavier_uniform_(m.weight)
            
            if isinstance(m, nn.BatchNorm3d):
                print('initializing batchnorm ', m)
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
#                 m.bias.data.fill_(0.01)
    def stn(self, x):
#         xs = self.decode_frames(x)
#         print(xs.shape)
#         print(x.shape)
        
#         xs = x.view(x.shape[0], -1)
#         print(xs.shape, self.fc_loc[0].weight.shape)
        theta = self.fc_loc(x)
        theta = theta.view(-1, 3, 4)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x
    
    def forward(self, x):
#         print(self.training)
        out = F.dropout3d(self.bn1(F.leaky_relu(self.conv1(x))), p=0.4, training=self.training)
        out = F.dropout3d(self.bn2(F.leaky_relu(self.conv2(out))), p=0.3, training=self.training)
        out = F.dropout3d(self.bn3(F.leaky_relu(self.conv3(out))), p=0.3, training=self.training)
        out = F.dropout3d(self.bn4(F.leaky_relu(self.conv4(out))), p=0.3, training=self.training)
        out = F.dropout3d(self.bn5(F.leaky_relu(self.conv5(out))), p=0.3, training=self.training)
        
        out2 = self.stn(out)
        out2 = F.dropout3d(self.m_bn6(F.leaky_relu(self.m_conv6(out2))), p=0.3, training=self.training)
        out2 = F.dropout3d(self.m_bn7(F.leaky_relu(self.m_conv7(out2))), p=0.3, training=self.training)
        out2 = F.dropout3d(self.m_bn8(F.leaky_relu(self.m_conv8(out2))), p=0.3, training=self.training)
        out_rois = F.sigmoid(self.conv9(out2))
        
        out = F.dropout3d(self.bn6(F.leaky_relu(self.conv6(out))), p=0.3, training=self.training)
        out = F.dropout3d(self.bn7(F.leaky_relu(self.conv7(out))), p=0.3, training=self.training)
        out = F.dropout3d(self.bn8(F.leaky_relu(self.conv8(out))), p=0.3, training=self.training)
        out_frames = F.sigmoid(self.conv9(out))
        
#         print('out_frames.shape', out_frames.shape)
#         out_rois = self.stn(out_frames)
        
        
        return out_frames, out_rois
    
    
    
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

            nn.Conv3d(128, 128, 3, padding=1, stride=4),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.4, inplace=True),

            nn.Conv3d(128, 128, 3, padding=1, stride=2),
            nn.LeakyReLU(),
            nn.Dropout2d(p=0.4, inplace=True),

            FlattenLayer(),

            nn.Linear(512, 300),
            nn.LeakyReLU(),
            nn.Dropout(p=0.4, inplace=True),

            nn.Linear(300, 1),
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
