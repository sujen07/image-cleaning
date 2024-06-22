import cv2
import numpy as np
import dlib

# Load the face detector and the landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load source and target images
source_image = cv2.imread("source.jpg")
target_image = cv2.imread("target.jpg")

# Function to extract landmarks
def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    if len(faces) == 0:
        return None
    landmarks = predictor(gray, faces[0])
    return np.array([(p.x, p.y) for p in landmarks.parts()])

# Get landmarks
source_landmarks = get_landmarks(source_image)
target_landmarks = get_landmarks(target_image)

# Function to extract Harris corners
def get_harris_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    harris_corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    keypoints = np.argwhere(harris_corners > 0.01 * harris_corners.max())
    return keypoints[:, [1, 0]]  # convert to (x, y) format

# Get Harris corners
source_corners = get_harris_corners(source_image)
target_corners = get_harris_corners(target_image)

# Combine landmarks and corners
source_points = np.vstack((source_landmarks, source_corners))
target_points = np.vstack((target_landmarks, target_corners))

min_length = min(len(source_points), len(target_points))
source_points = source_points[:min_length]
target_points = target_points[:min_length]

# Compute the similarity transform
M, inliers = cv2.estimateAffinePartial2D(source_points, target_points, method=cv2.RANSAC)

# Apply the transformation if it exists
if M is not None:
    aligned_image = cv2.warpAffine(source_image, M, (target_image.shape[1], target_image.shape[0]))
    cv2.imwrite("aligned_image.jpg", aligned_image)
else:
    print("Similarity transform could not be computed.")

# Function to compute mean values for each color channel
def compute_channel_means(image):
    mean_values = cv2.mean(image)
    return np.array(mean_values[:3])  # Ignore the alpha channel if present

# Compute mean values for each color channel in both images
mean_source = compute_channel_means(aligned_image)
mean_target = compute_channel_means(target_image)

# Perform color correction
corrected_image = aligned_image + (mean_target - mean_source)

# Clip values to ensure they remain in the valid range [0, 255]
corrected_image = np.clip(corrected_image, 0, 255).astype(np.uint8)

# Save the corrected image
corrected_image_path = 'corrected_image.jpg'
cv2.imwrite(corrected_image_path, corrected_image)

print(f"Color corrected image saved to {corrected_image_path}")

def get_face_landmarks_and_hull(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        raise Exception("No face detected")

    landmarks = predictor(gray, faces[0])
    points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
    points = np.array(points, dtype=np.int32)

    return points

def create_focus_mask(landmarks, image_shape):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    # Indices for important features: eyes, nose, mouth
    indices = {
        #'left_eye': list(range(36, 42)),
        #'right_eye': list(range(42, 48)),
        #'nose': list(range(27, 36)),
        #'mouth': list(range(48, 61))
        'whole_face': list(range(17,60)) + list(range(0,6)) + list(range(11, 17))
    }
    
    for key in indices:
        region_points = landmarks[indices[key]]
        hull = cv2.convexHull(region_points)
        cv2.fillConvexPoly(mask, hull, 255)

    return mask

# Get face landmarks
source_landmarks = get_face_landmarks_and_hull(corrected_image)
target_landmarks = get_face_landmarks_and_hull(target_image)

# Create focused masks
source_mask = create_focus_mask(source_landmarks, corrected_image.shape)

# Normalize mask to range [0, 1]
source_mask = source_mask / 255.0

# Apply Gaussian blur to the mask
source_mask = cv2.GaussianBlur(source_mask, (7, 7), 0)

# Blend images using alpha mask
source_mask = np.repeat(source_mask[:, :, np.newaxis], 3, axis=2)  # Make mask 3 channels

# Composite image using source_mask to blend
composite_img = (source_mask * corrected_image + (1 - source_mask) * target_image).astype(np.uint8)

final_img = cv2.imread("nara_shyam1.jpg")
top,right,bottom,left = (554, 464, 644, 374)
final_img[top:bottom, left:right] = composite_img

# Save or display result
cv2.imwrite('composite_image.jpg', final_img)
cv2.imshow('Composite Image', final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
