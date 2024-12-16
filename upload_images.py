import os
import base64
from pymongo import MongoClient

# MongoDB connection string
connection_string = "mongodb+srv://ravitarun2103:kF1SLaoKVkwBnQzu@cluster0.brb5i.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Connect to MongoDB
def get_mongo_client():
    return MongoClient(connection_string)

# Convert image to base64 for MongoDB storage
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# Insert images into MongoDB
def insert_images_into_db(collection, image_folder):
    for image_name in os.listdir(image_folder):
        if image_name.endswith(('.png', '.jpg', '.jpeg')):  # Only include image files
            image_path = os.path.join(image_folder, image_name)
            encoded_image = image_to_base64(image_path)
            collection.insert_one({
                'name': image_name,
                'image': encoded_image
            })
            print(f"Uploaded {image_name}")

# Main script
if __name__ == "__main__":
    # MongoDB setup
    client = get_mongo_client()
    db = client['image_database']  # Replace with your desired database name
    collection = db['images']     # Replace with your desired collection name

    # Path to the folder containing images
    image_folder = "images/"  # Update this to the path of your image folder

    # Insert images into MongoDB
    insert_images_into_db(collection, image_folder)
    print("All images have been uploaded successfully!")
