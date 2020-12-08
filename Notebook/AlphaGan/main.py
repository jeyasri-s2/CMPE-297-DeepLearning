
from parameter import *
from trainer import Trainer
from tester import Tester
from alpha_trainer import alpha_Trainer
# from tester import Tester
from data_loader import Data_Loader
from torch.backends import cudnn
from utils import make_folder
import os
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main(config):
    # For fast training
    cudnn.benchmark = True


    # Data loader
    data_loader = Data_Loader(config.train, config.dataset, config.mura_class, config.mura_type,
            config.image_path, config.imsize, config.batch_size, shuffle=config.train)
    """
    train = MURASubset(filenames=splitter.data_train.path, patients=splitter.data_train.patient,
                       transform=composed_transforms, true_labels=np.zeros(len(splitter.data_train.path)))
    validation = MURASubset(filenames=splitter.data_val.path, true_labels=splitter.data_val.label,
                            patients=splitter.data_val.patient, transform=composed_transforms_val)
    test = MURASubset(filenames=splitter.data_test.path, true_labels=splitter.data_test.label,
                      patients=splitter.data_test.patient, transform=composed_transforms_val)

    train_loader = DataLoader(train, batch_size=run_params['batch_size'], shuffle=True, num_workers=num_workers,
                              worker_init_fn=loader_init_fn, drop_last=model_class in [DCGAN])
    val_loader = DataLoader(validation, batch_size=run_params['batch_size'], shuffle=True, num_workers=num_workers,
                            drop_last=model_class in [DCGAN])
    test_loader = DataLoader(test, batch_size=run_params['batch_size'], shuffle=True, num_workers=num_workers,
                             drop_last=model_class in [DCGAN])
    """

    # for batch_data in tqdm(data_loader, desc='Training', total=len(data_loader)):
    #     print('batch_data ',batch_data)

    # Create directories if not exist
    make_folder(config.model_save_path, config.version)
    make_folder(config.sample_path, config.version)
    make_folder(config.log_path, config.version)
    make_folder(config.attn_path, config.version)


    if config.train:
        if config.model=='sagan':
            trainer = Trainer(data_loader.loader(), config)
        elif config.model == 'qgan':
            trainer = qgan_trainer(data_loader.loader(), config)
        elif config.model == 'alpha':
            trainer = alpha_Trainer(data_loader.loader(), config)
        print('calling the trainer')
        trainer.train()
    else:
        tester = Tester(data_loader.loader(), config)
        tester.test()

if __name__ == '__main__':
    config = get_parameters()
    #print(config)
    main(config)
