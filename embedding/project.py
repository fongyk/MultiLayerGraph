# tensorboard --logdir=your_log_dir --host=your_host_ip

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import numpy as np

writer = SummaryWriter('your_log_dir', flush_secs=10)

datasets = {
    'oxford5k': {
        'num': 5064,
        'feature': 'oxford5k_feature.npy',
        'label': 'oxford5k_label.npy',
        'image_txt': 'oxford5k_image_list.txt',},
    'paris6k': {
        'num': 6393,
        'feature': 'paris6k_feature.npy',
        'label': 'paris6k_label.npy',
        'image_txt': 'paris6k_image_list.txt',},
}

img_transform = transforms.Compose([
        transforms.Resize((25, 25)),
        transforms.ToTensor(),
    ])

for dataset in datasets.keys():
    feature = np.load(datasets[dataset]['feature'])
    print(feature.shape)
    label = np.load(datasets[dataset]['label'])
    label_img = torch.randn(datasets[dataset]['num'], 3, 25, 25)
    with open(datasets[dataset]['image_txt'], 'r') as fr:
        images = fr.readlines()
        for i, img in enumerate(images):
            img = img.strip()
            img = Image.open(img)
            label_img[i] = img_transform(img)
    writer.add_embedding(feature, metadata=label, label_img=label_img, global_step=1, tag=dataset)
writer.close()
