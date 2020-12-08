import torch
import torchvision.datasets as dsets
from torchvision import transforms
from data_mura import customDf, MURA_dataset


class Data_Loader():
    def __init__(self, train, dataset, mura_class, mura_type, image_path, image_size, batch_size, shuffle=True):
        self.train = train
        self.dataset = dataset
        self.mura_class = mura_class
        self.mura_type = mura_type
        self.path = image_path
        self.imsize = image_size
        self.batch = batch_size
        self.shuffle = shuffle

    def transform(self, rotation, hflip, resize, totensor, normalize, centercrop, to_pil, gray):
        options = []
        if to_pil:
            options.append(transforms.ToPILImage())
        if gray:
            options.append(transforms.Grayscale())
        if rotation:
            options.append(transforms.RandomRotation(20))
        if hflip:
            options.append(transforms.RandomHorizontalFlip())
        if centercrop:
            options.append(transforms.CenterCrop(256))
        if resize:
            options.append(transforms.Resize((self.imsize,self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        # if True:
        #     options.append(transforms.Lambda(lambda x: (x - x.min())/(x.max()-x.min())))
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        transform = transforms.Compose(options)
        return transform

    def load_mura(self, studyClass, studyType):
        # train = MURASubset(filenames=splitter.data_train.path, patients=splitter.data_train.patient,
        #                    transform=composed_transforms, true_labels=np.zeros(len(splitter.data_train.path)))
        transforms = self.transform(False, True, True, True, True, True, True, False)
        mura_df = customDf(self.path+'/MURA-v1.1/train_image_paths.csv', studyClass, studyType)
        dataset = MURA_dataset(mura_df, self.path, transforms)
        return dataset

    def load_mura_valid(self, studyClass, studyType):
        transforms = self.transform(False, True, True, True, True, True, True, False)
        mura_df = customDf(self.path+'/MURA-v1.1/valid_image_paths.csv', studyClass, studyType)
        dataset = MURA_dataset(mura_df, self.path, transforms)
        return dataset

    def loader(self):
        if self.dataset == 'mura':
            if self.train:
                dataset = self.load_mura(self.mura_class, self.mura_type)
            else:
                dataset = self.load_mura_valid(self.mura_class, self.mura_type)

        print("Dataset length: ", len(dataset))


        # train_loader = DataLoader(train, batch_size=run_params['batch_size'], shuffle=True, num_workers=num_workers,
        #                               worker_init_fn=loader_init_fn, drop_last=model_class in [DCGAN])

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=self.batch,
                                              shuffle=self.shuffle,
                                              num_workers=4,
                                              drop_last=False)
        return loader
