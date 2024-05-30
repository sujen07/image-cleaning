import logging
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import requests
from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def authenticate_google_photos():
    # Define the scopes
    scopes = ['https://www.googleapis.com/auth/photoslibrary.readonly']

    # Create the flow using the client secrets file
    flow = InstalledAppFlow.from_client_secrets_file(
        'client_secrets.json', scopes=scopes)

    # Run the flow using a local server to handle the authentication
    credentials = flow.run_local_server(host='localhost',
                                        port=0,
                                        authorization_prompt_message="Test",
                                        success_message='The auth flow is complete; you may close this window.',
                                        open_browser=True)
    
    # Build the service from the credentials
    service = build('photoslibrary', 'v1', credentials=credentials, static_discovery=False)
    logging.info("Successfully built the service.")
    return service

def list_albums(service):
    results = service.albums().list(pageSize=10).execute()
    albums = results.get('albums', [])
    if albums:
        logging.info("Found albums.")
    else:
        logging.warning("No albums found.")
    for album in albums:
        logging.info(f"Album id: {album['id']} - Album title: {album['title']}")
    return albums

def download_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content))
    return None

def save_image(image, filename, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    image.save(os.path.join(folder, filename))
    logging.info(f"Saved {filename} to {folder}")

def get_all_images(service, album_id, folder):
    next_page_token = None
    images = []
    tensors = []
    count = 0
    image_counter = 0  # Counter for filenames
    
    all_media_items = []
    while count <= 3:
        body = {'albumId': album_id, 'pageSize': 100}
        if next_page_token:
            body['pageToken'] = next_page_token
        results = service.mediaItems().search(body=body).execute()
        media_items = results.get('mediaItems', [])
        next_page_token = results.get('nextPageToken')
        all_media_items.append(media_items)
        
        # Create a list of URLs for downloading
        #urls = [f"{item['baseUrl']}=d" for item in media_items]
        urls = []
        # Use ThreadPoolExecutor to download images in parallel
        with ThreadPoolExecutor(max_workers=16) as executor:
            # Map download_image function to each URL
            for image in executor.map(download_image, urls):
                if image:
                    images.append(image)
                    tensor = transforms.ToTensor()(image)
                    tensors.append(tensor)
                    filename = f"{image_counter}.jpg"  # Use the counter for filename
                    save_image(image, filename, folder)
                    logging.info(f"Converted and saved image {image_counter}.jpg to tensor.")
                    image_counter += 1
        
        if not next_page_token:
            break
        count += 1

    return all_media_items, tensors

def main():
    service = authenticate_google_photos()
    albums = list_albums(service)
    album_id = albums[0]['id']
    folder = 'album_images'
    imgs, tensors = get_all_images(service, album_id, folder)
    return imgs

if __name__ == '__main__':
    imgs = main()
