import numpy as np
import cv2
import os
from matplotlib import pyplot as plt


DIR = r"C:\Users\HP\PycharmProjects\Color_Transfer"
PROTOTXT = os.path.join(DIR, r"Models/colorization_deploy_v2 (1).prototxt")
POINTS = os.path.join(DIR, r"Models/pts_in_hull.npy")
MODEL = os.path.join(DIR, r"Models/colorization_release_v2.caffemodel")


S=input("Enter the path to the File :: ")
IMAGE_PATH = fr"{S}"

# Load the Model
print("Load model")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

# Load centers for ab channel quantization used for rebalancing.
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Load the input image
image = cv2.imread(IMAGE_PATH)

# Check if the image was loaded successfully
if image is None:
    print(f"Error: Unable to load image at {IMAGE_PATH}")
else:
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    print("Colorizing the image")
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    colorized = (255 * colorized).astype("uint8")

    # Display images using matplotlib
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.title("Colorized")
    plt.imshow(cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
