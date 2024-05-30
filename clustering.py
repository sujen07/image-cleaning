import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('images/47.jpg')

# Check if image is loaded
if image is None:
    print("Image not found. Please check the file path.")
else:
    # Convert the image to a NumPy array
    x, y, _ = image.shape

    # Reshape the image into a 2D array
    image_reshaped = image.reshape((x * y, 3))

    # Create a KMeans object with 4 clusters
    kmeans = KMeans(n_clusters=4)

    # Fit the KMeans object to the data
    kmeans.fit(image_reshaped)

    # Predict the cluster labels for each pixel
    labels = kmeans.predict(image_reshaped)

    # Reshape the labels array into the original image shape
    segmented_image = labels.reshape(x, y)

    # Convert cluster labels to a visible image format
    segmented_image = np.uint8(255 * segmented_image / segmented_image.max())

    # Display the segmented image using matplotlib
    plt.imshow(segmented_image, cmap='gray')
    plt.title('Segmented Image')
    plt.show()
