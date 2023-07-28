import ijson
import os
import random
import csv


annot_file = 'region_descriptions.json'
image_names = os.listdir('VG')
num_images = 5

def get_numeric_part(filename):
    # Assuming the numeric part is the first part before the '.jpg' extension
    return int(filename.split('.')[0])


image_names = sorted(image_names, key=get_numeric_part)

csv_file_path = "output.csv"

existing_regions = []

counter = 0

with open("regions.csv", "a", newline="", encoding="utf-8") as csv_file:
    fieldnames = ["image_id", "x", "y", "width", "height", "phrase"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    with open(annot_file, 'rb') as annot:
        for obj in ijson.items(annot, 'item'):
            if str(obj['id']) + '.jpg' in image_names:
                if num_images > len(obj['regions']):
                    continue
                regions = []
                img_indexes = random.sample(range(0, len(obj['regions']) - 1), num_images)
                for idx in img_indexes:
                    obj['regions'][idx].pop('region_id')
                    obj['regions'][idx]['phrase'] = obj['regions'][idx]['phrase'].replace("\n", " ")
                    regions.append(obj['regions'][idx])

                for item in regions:
                    writer.writerow(item)

                print(counter, str(obj['id']) + '.jpg')
                counter += 1
            else:
                print(str(obj['id']) + '.jpg', 'doesnt exist')


