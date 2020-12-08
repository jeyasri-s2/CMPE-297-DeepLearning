from torch.utils.data import DataLoader, Dataset
import cv2
import os
import torchvision

class MURA_dataset(Dataset):
    '''
    Dataset class for MURA dataset
    Args:
        - df: Dataframe with the first columns contains the path to the images
        - root_dir: string contains path of  root directory
        - transforms: Pytorch transform operations
    '''

    def __init__(self, df, root_dir, transforms=None):
        #print("I am calling Mura dataset")
        self.df = df
        self.root_dir = root_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.df.iloc[idx, 0])
        #print('img_name ',img_name)
        img = cv2.imread(img_name)
        #print('img shape ',img.shape)

        if self.transforms:
            img = self.transforms(img)
        #print('img shape after reshape : ',img.shape)

        if 'negative' in img_name: label = 0
        else: label = 1

        return img, label

def transform(rotation, hflip, resize, totensor, normalize, centercrop, to_pil, gray):
    options = []
    if to_pil:
        options.append(torchvision.transforms.ToPILImage())
    if gray:
        options.append(torchvision.transforms.Grayscale())
    if rotation:
        options.append(torchvision.transforms.RandomRotation(20))
    if hflip:
        options.append(torchvision.transforms.RandomHorizontalFlip())
    if centercrop:
        options.append(torchvision.transforms.CenterCrop(256))
    if resize:
        options.append(torchvision.transforms.Resize((32,32)))
    if totensor:
        options.append(torchvision.transforms.ToTensor())
    # if True:
    #     options.append(transforms.Lambda(lambda x: (x - x.min())/(x.max()-x.min())))
    if normalize:
        options.append(torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = torchvision.transforms.Compose(options)
    return transform

def inverse_transform(rotation, hflip, resize, totensor, normalize, centercrop, to_pil, gray):
    options = []
    if to_pil:
        options.append(torchvision.transforms.ToPILImage())
    if gray:
        options.append(torchvision.transforms.Grayscale())
    if rotation:
        options.append(torchvision.transforms.RandomRotation(-20))
    if hflip:
        options.append(torchvision.transforms.RandomHorizontalFlip())
    if centercrop:
        options.append(torchvision.transforms.CenterCrop(256))
    if resize:
        options.append(torchvision.transforms.Resize((32,32)))
    if totensor:
        options.append(torchvision.transforms.ToTensor())
    # if True:
    #     options.append(transforms.Lambda(lambda x: (x - x.min())/(x.max()-x.min())))
    if normalize:
        options.append(torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = torchvision.transforms.Compose(options)
    return transform