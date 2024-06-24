import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from sklearn.manifold import TSNE
import re
from scipy.spatial.distance import euclidean

# Constants
BATCH_SIZE = 32
scaler = StandardScaler()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def numerical_sort(value):
    """Helper function to extract numerical values from a string."""
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else 0

def sorted_numerical_filenames(folder_path):
    """Sort filenames in a folder numerically."""
    filenames = os.listdir(folder_path)
    sorted_filenames = sorted(filenames, key=numerical_sort)
    return sorted_filenames

def load_model():
    """Load and prepare the pre-trained ResNet50 model."""
    model = resnet50(pretrained=True)
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()
    return model

def extract_features(model, images):
    """Extract features from a list of images using the given model."""
    features = []
    for img in images:
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = model(img_tensor)
        features.append(feature.cpu().numpy().flatten())
    return np.array(features)

def get_images_from_folder(folder_path):
    """Load images from the specified folder."""
    image_paths = []
    for filename in sorted_numerical_filenames(folder_path)[:50]:
        if filename.endswith(".jpg"):
            file_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(file_path).convert('RGB')
                image_paths.append((file_path, img))
            except Exception as e:
                print(f"Error loading image {file_path}: {e}")
    return image_paths

def plot_k_distance(sorted_distances, knee_locator):
    """Plot the K-distance graph to determine the optimal epsilon value."""
    plt.figure(figsize=(10, 5))
    plt.plot(sorted_distances)
    plt.vlines(knee_locator.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.title("K-distance Graph")
    plt.xlabel("Points sorted by distance")
    plt.ylabel("Epsilon (distance to 3rd nearest neighbor)")
    plt.show()

def plot_clusters(features_tsne, clusters):
    """Plot the TSNE clusters."""
    plt.scatter(features_tsne[:,0], features_tsne[:,1], c=clusters)
    plt.colorbar()
    plt.show()


def calculate_image_distance(image_path1, image_path2):
    """Calculate the Euclidean distance between two images."""
    model = load_model()

    img1 = Image.open(image_path1).convert('RGB')
    img2 = Image.open(image_path2).convert('RGB')

    features1 = extract_features(model, [img1])[0]
    features2 = extract_features(model, [img2])[0]

    distance = euclidean(features1, features2)
    return distance

def main(folder_path):
    # Load model
    model = load_model()

    # Get images
    images = get_images_from_folder(folder_path)

    # Extract features
    features = extract_features(model, [img for path, img in images])

    # Scale features
    scaled_features = features #scaler.fit_transform(features)

    # Print features after scaling
    print("Features after scaling:")
    for i, (path, _) in enumerate(images):
        print(f"{path}: {scaled_features[i]}")

    # DBSCAN clustering
    dbscan = DBSCAN(eps=10, min_samples=1)
    clusters = dbscan.fit_predict(scaled_features)

    # Group images by clusters
    cluster_images = {i: [] for i in range(-1, max(clusters) + 1)}
    for idx, cluster in enumerate(clusters):
        cluster_images[cluster].append(images[idx])
    return cluster_images

if __name__ == "__main__":
     folder_path = 'album_images'
     cluster_imgs = main(folder_path)
