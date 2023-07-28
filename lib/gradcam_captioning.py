import os
import torch
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2
from vgg_model import VGG
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from labels import labels






def get_heatmap(img, model, preds):

    pred = preds.argmax(dim=1)

    print(labels[pred])

    # get the gradient of the output with respect to the parameters of the model
    preds[:, pred].backward()
    # pull the gradients out of the model
    gradients = model.get_activations_gradient()
    # pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    # get the activations of the last convolutional layer
    activations = model.get_activations(img).detach()
    # weight the channels by corresponding gradients
    for i in range(512):
        activations[:, i, :, :] *= pooled_gradients[i]
    # average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap, 0)
    # normalize the heatmap
    heatmap /= torch.max(heatmap)

    return heatmap



def get_nth_contour(contours, n):

    contour_areas = [cv2.contourArea(contour) for contour in contours]
    sorted_areas = sorted(contour_areas, reverse=True)
    nth_largest_area = sorted_areas[n-1]
    nth_largest_contour = None
    for contour in contours:
        if cv2.contourArea(contour) == nth_largest_area:
            nth_largest_contour = contour
            break

    return nth_largest_contour





def get_smooth_masked(img_name, heatmap):

    img = cv2.imread(img_name, cv2.IMREAD_COLOR)

    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    threshold_value = 150  # Adjust this threshold value based on your heatmap
    _, mask = cv2.threshold(heatmap, threshold_value, 255, cv2.THRESH_BINARY)
    # Define the colors to keep
    color1 = np.array([0, 0, 255])  # (0, 0, 255) - Red
    color2 = np.array([0, 255, 255])  # (0, 255, 255) - Yellow

    # Create a mask for color1 and color2
    mask1 = cv2.inRange(mask, color1, color1)
    mask2 = cv2.inRange(mask, color2, color2)


    blue_contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_blue_ctr = get_nth_contour(blue_contours, 1)

    light_blue_contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    light_blue_contours = list(light_blue_contours)

    # Initialize variables for the closest contour
    closest_contour = None
    closest_distance = float('inf')

    # Iterate over the list of contours
    for contour in light_blue_contours:
        # Iterate over each point in contour c
        for point_c in largest_blue_ctr:
            # Iterate over each point in the current contour
            for point in contour:
                # Calculate the Euclidean distance between the points
                distance = np.linalg.norm(point_c - point)

                # Check if the distance is smaller than the current closest distance
                if distance < closest_distance:
                    closest_distance = distance
                    closest_contour = contour


    x, y, w, h = cv2.boundingRect(closest_contour)

    # Create a mask for the largest contour
    largest_mask = np.zeros_like(mask)

    cv2.drawContours(largest_mask, [closest_contour], 0, 255, -1)
    cv2.drawContours(largest_mask, [largest_blue_ctr], 0, 255, -1)

    largest_mask = largest_mask[:,:,0]

    bounded_img = img[y:y+h,x:x+w]

    inv_mask = cv2.bitwise_not(largest_mask).astype(np.uint8)
    inv_mask = inv_mask[:, :, np.newaxis]
    color_inv_mask = np.repeat(inv_mask, 3, axis=2)

    average_color = cv2.mean(bounded_img)
    average_color = np.uint8(average_color[:3])

    color = np.ones_like(color_inv_mask) * average_color
    color *= color_inv_mask

    masked_img = (img * (largest_mask[:, :, np.newaxis] / 255)).astype(np.uint8)
    masked_img += color[:, :, ::-1]
    final_img = masked_img[y:y+h,x:x+w]


    return final_img




def draw_heatmap_on_image(img_name, heatmap):

    img = cv2.imread(img_name)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = heatmap * 0.4 + img
    cv2.imwrite('map.jpg', superimposed_img)




def get_bbox_coords(heatmap):

    heatmap_scaled = (heatmap * 255).astype(np.uint8)
    threshold_value = 200  # Adjust this threshold value based on your heatmap
    _, thresholded = cv2.threshold(heatmap_scaled, threshold_value, 255, cv2.THRESH_BINARY)

    plt.imshow(thresholded)
    plt.show()
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_bbox = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > max_area:
            max_area = area
            max_bbox = (x, y, w, h)
    bbox_img = cv2.cvtColor(heatmap, cv2.COLOR_GRAY2RGB)
    if max_bbox is not None:
        x, y, w, h = max_bbox
        cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 255, 0), 1)  # Green bounding box with thickness 2
    plt.imshow(bbox_img)
    plt.axis("off")
    plt.show()

    return max_bbox




def main():


    # use the ImageNet transformation
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # define a 1 image dataset
    dataset = datasets.ImageFolder(root='images', transform=transform)

    # define the dataloader to load that single image
    dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)


    vgg = VGG()

    vgg.eval()


    img_classes = sorted(os.listdir('images'))

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    prompt = "the object in the image is"

    for batch_idx, batch_data in enumerate(dataloader):

        img, _ = batch_data

        preds = vgg(img)

        heatmap = get_heatmap(img, vgg, preds)
        heatmap = np.array(heatmap)


        img_class = img_classes[batch_idx]
        img_name = os.listdir(os.path.join('images', img_class))[0]
        img_path = os.path.join('images', img_class, img_name)
        orig_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        orig_img = orig_img[:, :, ::-1]


        processed_img = get_smooth_masked(img_path, heatmap)
        cv2.imwrite("cropped_image.png", processed_img)

        draw_heatmap_on_image(img_path, heatmap)

        raw_image = Image.open("cropped_image.png")

        # conditional image captioning
        inputs = processor(raw_image, prompt, return_tensors="pt")

        out = model.generate(**inputs)
        print(processor.decode(out[0], skip_special_tokens=True))

        inputs = processor(raw_image, return_tensors="pt")

        out = model.generate(**inputs)
        print(processor.decode(out[0], skip_special_tokens=True))

        print(img_name, "done")
        input()







if __name__ == '__main__':
    main()



