import torch
import torch.nn as nn
import numpy as np
import os

from torch.nn.functional import mse_loss
from args import get_parser
import pickle
from model import get_model
from torchvision import transforms

from mura_dataloader import MURA_dataset, transform, inverse_transform
from utils.output_utils import prepare_output
from PIL import Image
import time
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import image
import pandas as pd
from flask import Flask, request, abort, jsonify
from flask_cors import CORS
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from PIL import Image
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
import numpy as np

from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage import measure



from util import tensor2im

from gan_model import Generator, Discriminator, Encoder

app = Flask(__name__)
CORS(app)

requestFolderName = 'demo_images'

def test():
    return 'test success !'

def anomaly_test():
    print('Anomaly')

    category = 'hand'
    img_file = requestFolderName+'/image-16.jpg'

    print('saved ',img_file, ' category: ',category)
    count = 16


    files = []
    for i in range(65):
        files.append(img_file)

    data = {'0': np.array(files)}

    mura_valid_df = pd.DataFrame(data)
    print(mura_valid_df.head())
    transforms = transform(False, True, True, True, True, True, True, False)
    transforms = inverse_transform(False, True, True, True, True, True, True, False)

    # resize image to 256 X 256 to construct the output image

    noresize_transform = transform(False, False, False, True, True, True, True, False)
    img = cv2.imread(img_file)
    print(img.shape)
    img = noresize_transform(img)
    print(img.shape)

    transforms1 = transform(False, True, False, False, False, False, True, False)
    resized_input_img = transforms1(img)

    # transforms2 = transform(False, True, False, False, False, False, True, False)
    # resized_input_img = transforms2(img)



    # rotation, hflip, resize, totensor, normalize, centercrop, to_pil, gray

    # valid_dataset = MURA_dataset(mura_valid_df, '/content/drive/Shared drives/MeanSquare-Drive/Advanced-DeepLearning/', transforms)
    valid_dataset = MURA_dataset(mura_valid_df, '', transforms)
    valid_dataloader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=64, shuffle=True,
                                                   num_workers=0,
                                                   drop_last=False)
    if category == 'hand':
        out = 'models/XR_HAND/'
    else:
        out = 'models/XR_ELBOW/'

    max_auc = 0
    latent_dim = 128
    channels = 3
    batch_size = 64

    generator = Generator(dim=64, zdim=latent_dim, nc=channels)
    discriminator = Discriminator(dim=64, zdim=latent_dim, nc=channels, out_feat=True)
    encoder = Encoder(dim=64, zdim=latent_dim, nc=channels)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    device = 'cpu'
    generator.load_state_dict(torch.load(out + 'G_epoch5000.pt', map_location=torch.device('cpu')))
    discriminator.load_state_dict(torch.load(out + 'D_epoch5000.pt', map_location=torch.device('cpu')))

    generator.to(device)
    encoder.to(device)
    discriminator.to(device)

    with torch.no_grad():
        labels = torch.zeros(size=(len(valid_dataloader.dataset),),
                             dtype=torch.long, device=device)

        scores = torch.empty(
            size=(len(valid_dataloader.dataset),),
            dtype=torch.float32,
            device=device)
        for i, (imgs, lbls) in enumerate(valid_dataloader):
            print('imgs. shape ', imgs.shape)
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            labels[i * batch_size:(i + 1) * batch_size].copy_(lbls)
            emb_query = encoder(imgs)
            print('emb_query. shape ', emb_query.shape)

            fake_imgs = generator(emb_query)
            emb_fake = encoder(fake_imgs)

            image_feats = discriminator(imgs)
            recon_feats = discriminator(fake_imgs)

            diff = imgs - fake_imgs

            image1_tensor = diff[0]
            im = tensor2im(imgs)

            im2 = tensor2im(fake_imgs)
            print(lbls)

            im3 = tensor2im(diff)
            # plt.figure(1)
            # plt.subplot(311)
            # plt.title('Real image')
            # plt.imshow(im)

            # plt.subplot(312)
            # plt.title('Fake img')
            # plt.imshow(im2)
            # plt.show()

            img = cv2.GaussianBlur(im3, (5, 5), 0)
            img_gray = rgb2gray(img)
            #plt.imshow(img_gray)
            thresh = threshold_otsu(img_gray)
            binary = img_gray > thresh

            #plt.imshow(binary)
            im_rgb = np.array(Image.fromarray(binary).convert('RGB'))
            mask = binary.copy()
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0

            mask3 = np.stack((mask, mask, mask), axis=2)

            all_labels = measure.label(mask)
            all_labels[all_labels >= 1] = 255
            all_labels[all_labels < 1] = 0
            all_labels3 = np.stack((all_labels, all_labels, all_labels), axis=2)

            #             kernel = np.ones((6, 6), np.uint8)

            #             # Using cv2.erode() method
            #             image = cv2.erode(Image.fromarray(mask3), kernel, cv2.BORDER_REFLECT)

            black_pixels_mask = np.all(mask3 == 1, axis=2)
            non_black_pixels_mask = np.any(mask3 > [0, 0, 0], axis=-1)

            all_labels3[non_black_pixels_mask] = [255, 0, 0]

            # plt.subplot(313)
            # plt.title('Difference')
            # plt.imshow(im3)
            # plt.show()
            #
            # plt.subplot(321)
            # plt.title('colored mask')
            # plt.imshow(all_labels3)
            # plt.show()

            gray = cv2.cvtColor(im3, cv2.COLOR_BGR2GRAY)

            # Find Canny edges
            edged = cv2.Canny(gray, 30, 200)

            # Finding Contours
            # Use a copy of the image e.g. edged.copy()
            # since findContours alters the image
            contours, hierarchy = cv2.findContours(edged,
                                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            # plt.subplot(322)
            # plt.imshow(edged)
            # plt.title('Edged')
            # plt.show()

            print("Number of Contours found = " + str(len(contours)))

            # Draw all contours
            # -1 signifies drawing all contours
            print('im3: ', im3.shape)
            backtorgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
            print('contours: ',len(contours))
            img_contours = np.zeros(backtorgb.shape)


            cv2.drawContours(img_contours, contours, -1, (220, 0, 0), 1)
            resized_output_image = cv2.resize(img_contours, (256, 256))

            cv2.imshow('output blue', resized_output_image)
            cv2.waitKey(0)



            cv2.imwrite('output_files/output-image-' + str(count) + '.jpg', resized_output_image)
            #Image.fromarray(resized_output_image).save('output_files/output-image-' + str(count) + '.jpg')
            print('resize: ',resized_output_image.shape, np.asarray(resized_input_img).shape)

            mix_img = cv2.addWeighted(np.asarray(resized_input_img), 0.3, resized_output_image, 0.7, 0,dtype=cv2.CV_32F)
            #Image.fromarray(mix_img).save('output_files/mix-image-' + str(count) + '.jpg')
            cv2.imwrite('output_files/mix-image-' + str(count) + '.jpg', mix_img)


            # plt.subplot(323)
            # plt.title('contour')
            # plt.imshow(gray)

            # plt.show()

            thresh = 50
            ret, thresh_img = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

            contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            print('contours second time : ',len(contours))

            backtorgb1 = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)

            cv2.drawContours(backtorgb1, contours, -1, (0, 255, 0), 1)

            #backtorgb = cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)

            cv2.imshow('output',backtorgb1)
            cv2.waitKey(0)

           # Image.fromarray(backtorgb1).save('output_files/image-' + str(count) + '.jpg')
           # cv2.imwrite('output_files/cv-image-' + str(count) + '.jpg', backtorgb1)
            #break

            image_distance = torch.mean(torch.pow(imgs - fake_imgs, 2), dim=[1, 2, 3])
            feat_distance = torch.mean(torch.pow(image_feats - recon_feats, 2), dim=1)
            print(emb_query.shape, emb_fake.shape)
            z_distance = mse_loss(emb_query, emb_fake)  # mse_loss(emb_query, emb_fake)
            # print z_distance
            print('z_distance=', z_distance)
            # print('hiiiiiiiii')
            scores[i * batch_size:(i + 1) * batch_size].copy_(feat_distance)
            print('feat_distance ',feat_distance[0])
            break


    output = {}
    output['status'] = 'done'
    return 'done'



if __name__=='__main__':
    print('hello')
    anomaly_test()
