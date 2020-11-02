from blazeface import BlazeFace
import torch
from skimage.transform import resize
import numpy as np

class MouthDetector():

    def __init__(self, device):
        self.net = BlazeFace().to(device)
        self.net.load_weights("blazeface.pth")
        self.net.load_anchors("anchors.npy")

        self.mouth_region_size = (64,64)
        self.img_dims = (128, 128)

    def plot_detections(self, img, detections, with_keypoints=True):
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.grid(False)
        ax.imshow(img/255.)
        
        if isinstance(detections, torch.Tensor):
            detections = detections.cpu().numpy()

        if detections.ndim == 1:
            detections = np.expand_dims(detections, axis=0)

        print("Found %d faces" % detections.shape[0])
            
        for i in range(detections.shape[0]):
            ymin = detections[i, 0] * img.shape[0]
            xmin = detections[i, 1] * img.shape[1]
            ymax = detections[i, 2] * img.shape[0]
            xmax = detections[i, 3] * img.shape[1]

            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=1, edgecolor="r", facecolor="none", 
                                     alpha=detections[i, 16])
            ax.add_patch(rect)
            print(ymin, ymax, xmin, xmax)
            if with_keypoints:
                for k in range(2,3):
                    kp_x = detections[i, 4 + k*2    ] * img.shape[1]
                    kp_y = detections[i, 4 + k*2 + 1] * img.shape[0]
                    circle = patches.Circle((kp_x, kp_y), radius=0.5, linewidth=1, 
                                            edgecolor="lightskyblue", facecolor="none", 
                                            alpha=detections[i, 16])
                    ax.add_patch(circle)
            
        plt.show()



    def mouth_detection(self, img, detections, with_keypoints=True, img_dims=(128,128)):
        # fig, ax = plt.subplots(1, figsize=(10, 10))
        # ax.grid(False)
        # ax.imshow(img/255.)
        
        if isinstance(detections, torch.Tensor):
            detections = detections.cpu().numpy()

        if detections.ndim == 1:
            detections = np.expand_dims(detections, axis=0)

        print("Found %d faces" % detections.shape[0])
        i = 0 # first face detection
        k = 2 # nose keypoint
        # for i in range(detections.shape[0]): #for all faces
        ymin = detections[i, 0] * img_dims[0]
        xmin = detections[i, 1] * img_dims[1]
        ymax = detections[i, 2] * img_dims[0]
        xmax = detections[i, 3] * img_dims[1]

        # print(xmin, xmax, ymin, ymax)
        # for k in range(2,3):  #for all keypoints
        kp_x = detections[i, 4 + k*2    ] * img_dims[1]
        kp_y = detections[i, 4 + k*2 + 1] * img_dims[0]

        print('########')
        print(kp_y, kp_x)

        mouth_region = img[int(kp_y):int(ymax), int(xmin):int(xmax)]
        return mouth_region

    def batch_mouth_detection(self, frames, detections, with_keypoints=True, img_dims= (128, 128)):

        """
            return mouth regions for a batch of frames along with status if any frame was skipped while keypoint finding
            mouth_regions: mouth rois
            flag: boolean if a frame is skipped ; True if a frame is skipped else False 
        """
        resize_frames = []
        for frame in frames:
            if frame.shape[0] !=self.img_dims[0] or frame.shape[1] != self.img_dims[1]:
                frame = resize(frame, self.img_dims)
                resize_frames.append(frame)
            else:
                resize_frames.append(frame)
        # print(len(resize_frames))

#         frames = torch.from_numpy(np.array(resize_frames))

#         if isinstance(detections, torch.Tensor):
#             detections = detections.cpu().numpy()

        # if len(detections) == 2:
        #     detections = np.expand_dims(detections, axis=1)

        # print("Found %d faces" % detections.shape[0])
        i = 0 # first face detection
        k = 2 # nose keypoint
        # for i in range(detections.shape[0]): #for all faces
        # print(len(detections))
        # print('########')
        # print(kp_y, kp_x)
        mouth_regions = []
        for index, img in enumerate(frames):
            if len(detections[index]) > 0:
                try:
                    ymin = detections[index][i, 0] * img_dims[0]
                    xmin = detections[index][i, 1] * img_dims[1]
                    ymax = detections[index][i, 2] * img_dims[0]
                    xmax = detections[index][i, 3] * img_dims[1]

                    # print(xmin, xmax, ymin, ymax)
                    # for k in range(2,3):  #for all keypoints
                    kp_x = detections[index][i, 4 + k*2    ] * img_dims[1]
                    kp_y = detections[index][i, 4 + k*2 + 1] * img_dims[0]

                    mouth_region = img[int(kp_y):int(ymax), int(xmin):int(xmax)]
                    mouth_regions.append(resize(mouth_region.cpu().numpy(), self.mouth_region_size))
                except IndexError:
                    flag = True
                    break
            else:
                flag = True

        if len(frames) == len(mouth_regions):
            flag = False   
        else: 
            flag = True
        # print(len(mouth_region_size))
        return np.array(mouth_regions), flag


    
########################## old code


# class VideoHandler(object):
#     def __init__(self, filepaths):
#         self.paths = filepaths
#         self.mp4_filenames = filepaths
#         self.blaze_detector = MouthDetector()
# #         self.mouth_extractor = FaceROIExtractor()
    
#     def read_video_audio_dyn(self, video_path):
#         # print(video_path, audio_path)
#         clip = VideoFileClip(video_path, verbose=False)
#         video_frames = torch.FloatTensor(list(clip.iter_frames()))
#         # video_frames = torch.FloatTensor(list(imageio.get_reader(video_path, 'ffmpeg')))

#         # waveform, sample_rate = torchaudio.load(audio_path)
#         waveform = torch.from_numpy(
#             clip.audio.to_soundarray()).float().permute(1, 0)
#         specgram = torchaudio.transforms.MelSpectrogram()(waveform)
#         return specgram, video_frames

#     def read_video_audio_blaze_roi(self, video_path, frame_len, subdir='train', audio_path=None):

#         frames = [] 
#         mouth_frames  = []
#         mouth_indices = []
#         video_frames = []
#         cap = cv2.VideoCapture(video_path)
#         frame_counter = 0
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if ret:
#                 frame = resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (256,256))*255
                
#                 frames.append(frame.astype(np.uint8))
#                 frame_counter += 1
#             else:
#                 break
#             if frame_counter == 31: # frame length
#                 break
#         cap.release()

#         frames = np.array(frames)
#         frames = (frames - frames.min()) / (frames.max() - frames.min())
#         img = torch.from_numpy(frames).permute(0,3,1,2) * 255
#         detections = self.blaze_detector.net.predict_on_batch(img)
#         mouth_regions, flag = self.blaze_detector.batch_mouth_detection(img.permute(0,2,3,1), detections)
        
#         if not flag:
#             mouth_regions /= 255.
#         return frames, mouth_regions, flag        





# class MouthDetector():

#     def __init__(self):
#         self.net = BlazeFace().to(device)
#         self.net.load_weights("blazeface.pth")
#         self.net.load_anchors("anchors.npy")

#         self.mouth_region_size = (64,64)
#         self.img_dims = (128, 128)

#     def plot_detections(self, img, detections, with_keypoints=True):
#         fig, ax = plt.subplots(1, figsize=(10, 10))
#         ax.grid(False)
#         ax.imshow(img/255.)
        
#         if isinstance(detections, torch.Tensor):
#             detections = detections.cpu().numpy()

#         if detections.ndim == 1:
#             detections = np.expand_dims(detections, axis=0)

#         print("Found %d faces" % detections.shape[0])
            
#         for i in range(detections.shape[0]):
#             ymin = detections[i, 0] * img.shape[0]
#             xmin = detections[i, 1] * img.shape[1]
#             ymax = detections[i, 2] * img.shape[0]
#             xmax = detections[i, 3] * img.shape[1]

#             rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
#                                      linewidth=1, edgecolor="r", facecolor="none", 
#                                      alpha=detections[i, 16])
#             ax.add_patch(rect)
#             print(ymin, ymax, xmin, xmax)
#             if with_keypoints:
#                 for k in range(2,3):
#                     kp_x = detections[i, 4 + k*2    ] * img.shape[1]
#                     kp_y = detections[i, 4 + k*2 + 1] * img.shape[0]
#                     circle = patches.Circle((kp_x, kp_y), radius=0.5, linewidth=1, 
#                                             edgecolor="lightskyblue", facecolor="none", 
#                                             alpha=detections[i, 16])
#                     ax.add_patch(circle)
            
#         plt.show()



#     def mouth_detection(self, img, detections, with_keypoints=True, img_dims=(128,128)):
#         # fig, ax = plt.subplots(1, figsize=(10, 10))
#         # ax.grid(False)
#         # ax.imshow(img/255.)
        
#         if isinstance(detections, torch.Tensor):
#             detections = detections.cpu().numpy()

#         if detections.ndim == 1:
#             detections = np.expand_dims(detections, axis=0)

#         print("Found %d faces" % detections.shape[0])
#         i = 0 # first face detection
#         k = 2 # nose keypoint
#         # for i in range(detections.shape[0]): #for all faces
#         ymin = detections[i, 0] * img_dims[0]
#         xmin = detections[i, 1] * img_dims[1]
#         ymax = detections[i, 2] * img_dims[0]
#         xmax = detections[i, 3] * img_dims[1]

#         # print(xmin, xmax, ymin, ymax)
#         # for k in range(2,3):  #for all keypoints
#         kp_x = detections[i, 4 + k*2    ] * img_dims[1]
#         kp_y = detections[i, 4 + k*2 + 1] * img_dims[0]

#         print('########')
#         print(kp_y, kp_x)

#         mouth_region = img[int(kp_y):int(ymax), int(xmin):int(xmax)]
#         return mouth_region

#     def batch_mouth_detection(self, frames, detections, with_keypoints=True, img_dims= (128, 128)):

#         """
#             return mouth regions for a batch of frames along with status if any frame was skipped while keypoint finding
#             mouth_regions: mouth rois
#             flag: boolean if a frame is skipped ; True if a frame is skipped else False 
#         """
#         resize_frames = []
#         for frame in frames:
#             if frame.shape[0] !=self.img_dims[0] or frame.shape[1] != self.img_dims[1]:
#                 frame = resize(frame, self.img_dims)
#                 resize_frames.append(frame)
#             else:
#                 resize_frames.append(frame.numpy())
#         # print(len(resize_frames))

#         frames = torch.from_numpy(np.array(resize_frames))

#         if isinstance(detections, torch.Tensor):
#             detections = detections.cpu().numpy()

#         # if len(detections) == 2:
#         #     detections = np.expand_dims(detections, axis=1)

#         # print("Found %d faces" % detections.shape[0])
#         i = 0 # first face detection
#         k = 2 # nose keypoint
#         # for i in range(detections.shape[0]): #for all faces
#         # print(len(detections))
#         # print('########')
#         # print(kp_y, kp_x)
#         mouth_regions = []
#         for index, img in enumerate(frames):
#             if len(detections[index]) > 0:
#                 try:
#                     ymin = detections[index][i, 0] * img_dims[0]
#                     xmin = detections[index][i, 1] * img_dims[1]
#                     ymax = detections[index][i, 2] * img_dims[0]
#                     xmax = detections[index][i, 3] * img_dims[1]

#                     # print(xmin, xmax, ymin, ymax)
#                     # for k in range(2,3):  #for all keypoints
#                     kp_x = detections[index][i, 4 + k*2    ] * img_dims[1]
#                     kp_y = detections[index][i, 4 + k*2 + 1] * img_dims[0]

#                     mouth_region = img[int(kp_y):int(ymax), int(xmin):int(xmax)]
#                     mouth_regions.append(resize(mouth_region, self.mouth_region_size))
#                 except IndexError:
#                     flag = True
#                     break
#             else:
#                 flag = True

#         if len(frames) == len(mouth_regions):
#             flag = False   
#         else: 
#             flag = True
#         # print(len(mouth_region_size))
#         return np.array(mouth_regions), flag