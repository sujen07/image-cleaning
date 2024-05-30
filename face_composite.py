import cv2
import face_recognition
import os
from pick_best_pic import *
import pdb

# Function to load images from folder
def load_images_from_folder(folder, ims):
    images = []
    for filename in ims:
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append((filename, img))
    return images

# Function to detect and encode faces
def get_face_encodings_and_locations(image):
    # Convert image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_image)
    
    # Encode faces
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    return face_encodings, face_locations

def smile_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    if len(faces) == 0:
        return 0

    # Assuming the first detected face is the target
    face = faces[0]
    landmarks = predictor(gray, face)

    # Extract coordinates of the mouth region
    mouth_points = []
    for i in range(48, 68):  # 48-67 are mouth points in the 68 point model
        mouth_points.append((landmarks.part(i).x, landmarks.part(i).y))
    
    mouth_points = np.array(mouth_points)
    
    # Calculate the width and height of the mouth
    mouth_width = np.linalg.norm(mouth_points[6] - mouth_points[0])  # Distance between corners of the mouth
    mouth_height = np.linalg.norm(mouth_points[3] - mouth_points[9])  # Distance between top and bottom of the mouth
    
    # Calculate smile intensity based on width and height
    smile_intensity = mouth_width * mouth_height
    
    # Analyze curvature: higher curvature implies a better smile
    upper_lip_curve = np.polyfit(mouth_points[:7, 0], mouth_points[:7, 1], 2)  # Upper lip
    lower_lip_curve = np.polyfit(mouth_points[7:, 0], mouth_points[7:, 1], 2)  # Lower lip
    
    upper_lip_curvature = upper_lip_curve[0]
    lower_lip_curvature = lower_lip_curve[0]
    
    curvature_score = (upper_lip_curvature - lower_lip_curvature) * -1  # Inverse curvature
    
    # Combine the features into a final smile score
    raw_score = smile_intensity + curvature_score
    
    return raw_score

def composite_score(images, faces):
    smiles = []
    eyes = []
    blurs = []
    brightness = []
    contrasts = []

    for image, face in zip(images, faces):
        _, image = image
        top,right, bottom, left = face
        face_image = image[top:bottom, left:right]

        smiles.append(smile_score(face_image))
        eyes.append(eyes_open_score(face_image))
        blurs.append(blur_score(face_image))
        b_score, c_score = lighting_score(face_image)
        brightness.append(b_score)
        contrasts.append(c_score)

    return \
        (np.array(smiles) / max(smiles) ) * 0.3 + \
        (np.array(eyes) / max(eyes) ) * 0.4 + \
        (np.array(blurs) / max(blurs) ) * 0.1 + \
        (np.array(brightness) / max(brightness) ) * 0.05 + \
        (np.array(contrasts) / max(contrasts) ) * 0.05



folder = 'album_images'
ims = ['76.jpg', '75.jpg']  # Add more image filenames as needed

# Load images
images = load_images_from_folder(folder, ims)

# Detect and encode faces in all images
image_face_data = {}
for filename, image in images:
    encodings, locations = get_face_encodings_and_locations(image)
    image_face_data[filename] = {
        'encodings': encodings,
        'locations': locations
    }

# Initialize the dictionary with faces from the first image
first_image_encodings = image_face_data[ims[0]]['encodings']
first_image_locations = image_face_data[ims[0]]['locations']

face_matches_dict = {}

for i, encoding1 in enumerate(first_image_encodings):
    matches = [first_image_locations[i]]  # Start with the location from the first image
    match_found_in_all_images = True
    
    for filename in ims[1:]:
        current_encodings = image_face_data[filename]['encodings']
        current_locations = image_face_data[filename]['locations']
        
        match_found = False
        for j, encoding2 in enumerate(current_encodings):
            match = face_recognition.compare_faces([encoding1], encoding2, tolerance=0.2)  # Adjust tolerance as needed
            if match[0]:
                matches.append(current_locations[j])
                match_found = True
                break
        
        if not match_found:
            match_found_in_all_images = False
            break
    
    if match_found_in_all_images:
        face_matches_dict[f"Face {i}"] = matches

# Print the dictionary of matches
print("Matches dictionary:")
for key, value in face_matches_dict.items():
    print(f"{key}: {value}")

for face in face_matches_dict:

    faces = face_matches_dict[face]

    comp_score = composite_score(images, faces)

    best_face_ind = np.argmax(comp_score)

    top,right,bottom,left = face_matches_dict[face][best_face_ind]
    face_image = images[best_face_ind][1][top:bottom, left:right]
    face_matches_dict[face] = face_image





superior_image = face_matches_dict['Face 2']
top,right,bottom,left = (1420, 2459, 1687, 2192)
inferior_image = images[0][1][top:bottom, left:right].copy()
superior_image = cv2.resize(superior_image, (inferior_image.shape[1], inferior_image.shape[0]))


# Function to get facial landmarks

def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        points = [(p.x, p.y) for p in landmarks.parts()]
        return points
    return None

# Get landmarks for both faces
inferior_landmarks = get_landmarks(inferior_image)
superior_landmarks = get_landmarks(superior_image)

# Function to align the superior face to the inferior face
def align_faces(inferior_points, superior_points, superior_image):
    # Convert points to NumPy arrays
    inferior_points = np.array(inferior_points, dtype=np.float32)
    superior_points = np.array(superior_points, dtype=np.float32)
    
    # Compute the affine transformation matrix
    affine_transform = cv2.estimateAffinePartial2D(superior_points, inferior_points)[0]
    
    # Warp the superior face to align with the inferior face
    aligned_face = cv2.warpAffine(superior_image, affine_transform, (inferior_image.shape[1], inferior_image.shape[0]))
    return aligned_face

aligned_superior_face = align_faces(inferior_landmarks, superior_landmarks, superior_image)

# Create a mask for the face region
mask = np.zeros_like(inferior_image)
hull = cv2.convexHull(np.array(inferior_landmarks))
cv2.fillConvexPoly(mask, hull, (255, 255, 255))

# Extract face regions
inferior_face_region = cv2.bitwise_and(inferior_image, mask)
superior_face_region = cv2.bitwise_and(aligned_superior_face, mask)

# Compute mean color values
inferior_mean = cv2.mean(inferior_face_region, mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))
superior_mean = cv2.mean(superior_face_region, mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))

# Color correction
correction_factors = (inferior_mean[0] / superior_mean[0], inferior_mean[1] / superior_mean[1], inferior_mean[2] / superior_mean[2])
corrected_superior_face = np.zeros_like(aligned_superior_face)
for c in range(3):
    corrected_superior_face[:, :, c] = cv2.multiply(aligned_superior_face[:, :, c], correction_factors[c])


# Define the mask for the source region
source_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
cv2.imwrite('trg.jpg', inferior_image)

# Find the seam using graph cut optimization
seamless_image = cv2.seamlessClone(corrected_superior_face, inferior_image, source_mask, (inferior_landmarks[30][0], inferior_landmarks[30][1]), cv2.NORMAL_CLONE)

images[0][1][top:bottom, left:right] = seamless_image

# Save the final output
cv2.imwrite('src.jpg', images[0][1])