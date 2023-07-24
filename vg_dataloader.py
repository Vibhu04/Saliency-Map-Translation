import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import ijson
import random
from PIL import Image


def get_numeric_part(filename):
    # Assuming the numeric part is the first part before the '.jpg' extension
    return int(filename.split('.')[0])


class VG_Dataset(Dataset):
    def __init__(self, img_dir, annot_file):
        self.img_dir = img_dir
        self.annot_file = annot_file
        self.img_transform = transforms.Compose([
            transforms.Resize((224, 224)),
        ])
        self.region_transform = transforms.Compose([
            transforms.Resize((100, 100)),
        ])

        self.image_names = os.listdir(self.img_dir)
        self.image_names = sorted(self.image_names, key=get_numeric_part)[:500]
        self.num_images = len(self.image_names)
        self.regions = []
        self.descr = []

        counter = 0
        with open(self.annot_file, 'rb') as annot:
            for obj in ijson.items(annot, 'item'):
                if str(obj['id']) + '.jpg' in self.image_names:
                    self.regions.append(obj['regions'])
                    counter += 1
                else:
                    continue
                if counter == self.num_images:
                    break




    def __len__(self):

        return self.num_images


    def __getitem__(self, idx):

        while True:
            img_path = os.path.join(self.img_dir, self.image_names[idx])
            try:
                image = Image.open(img_path)
                break
            except:
                idx = random.randint(self.num_images - 1)
                input("Found")

        tensor_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        ])
        image = tensor_transform(image)

        regions = self.regions[idx]
        num_regions = len(regions)
        random_region = regions[random.randint(0, num_regions - 1)]

        x, y, w, h = random_region['x'], random_region['y'], random_region['width'], random_region['height']

        x = max(0, x)
        y = max(0, y)

        region = image[:, y:y+h, x:x+w]

        caption = random_region['phrase']

        image = self.img_transform(image)
        region = self.region_transform(region)

        return image, region, caption
