import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from PIL import Image
import csv


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
        self.image_names = sorted(self.image_names, key=get_numeric_part)
        self.num_images = len(self.image_names)
        self.regions = []
        self.descr = []

        self.load_annotations()



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

        print(img_path)
        tensor_transform = transforms.Compose([
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        ])
        image = tensor_transform(image)

        regions = self.regions[idx]
        num_regions = len(regions)
        random_region = regions[random.randint(0, num_regions - 1)]

        x, y, w, h = int(random_region[1]), int(random_region[2]), int(random_region[3]), int(random_region[4])

        print(random_region[0])

        x = max(0, x)
        y = max(0, y)

        region = image[:, y:y+h, x:x+w]

        caption = random_region[5]

        image = self.img_transform(image)
        region = self.region_transform(region)

        return image, region, caption


    def load_annotations(self):

        counter = 0
        img_idx = 0
        cur_img = int(self.image_names[img_idx].strip('.jpg'))
        with open(self.annot_file, "r", newline="", encoding="utf-8") as csv_file:
            reader = csv.reader(csv_file)
            regions = []
            for row in reader:
                csv_cur_img = int(row[0])
                if csv_cur_img != cur_img:
                    self.regions.append(regions)
                    regions = []
                    counter += 1
                    img_idx += 1
                    cur_img = int(self.image_names[img_idx].strip('.jpg'))
                    while csv_cur_img != cur_img:
                        self.image_names.remove(self.image_names[img_idx])
                        cur_img = int(self.image_names[img_idx].strip('.jpg'))
                else:
                    regions.append(row)

            self.regions.append(regions)
            counter += 1

        print(counter)
        print(len(self.image_names))
        print(len(self.regions))


