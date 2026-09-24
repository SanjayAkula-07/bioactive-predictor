import os
import shutil

# Define the exact folder structure required by the application
folders_to_create = [
    "frontend",
    "backend",
    "backend/model"
]

# Map each file to its correct destination folder
file_destinations = {
    # Frontend web files
    "index.html": "frontend",
    "style.css": "frontend",
    "script.js": "frontend",
    
    # Backend server files
    "main.py": "backend",
    
    # Machine learning model artifacts
    "metadata.json": "backend/model",
    "best_model.pkl": "backend/model"
}

def organize_files():
    # 1. Create the directories
    for folder in folders_to_create:
        os.makedirs(folder, exist_ok=True)
        print(f"Created directory: {folder}/")

    # 2. Move the files
    for filename, destination_folder in file_destinations.items():
        if os.path.exists(filename):
            destination_path = os.path.join(destination_folder, filename)
            # Move and overwrite if it already exists in the destination
            shutil.move(filename, destination_path)
            print(f"Moved: {filename} -> {destination_folder}/")
        else:
            print(f"Skipped: {filename} (File not found in current directory)")

    print("\nProject organization complete! You can now start the FastAPI server.")

if __name__ == "__main__":
    organize_files()