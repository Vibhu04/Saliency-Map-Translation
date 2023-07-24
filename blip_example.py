from PIL import Image
import requests
from transformers import AutoProcessor, BlipModel
import numpy as np
import matplotlib.pyplot as plt

blip_model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = np.array(Image.open(requests.get(url, stream=True).raw))

plt.imshow(image)
plt.show()

image = np.expand_dims(image, axis=0)
print(image.shape)

inputs = processor(
    text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
)

blip_model.eval()

outputs = blip_model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)

print(probs)