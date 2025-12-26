from roboflow import Roboflow
import os

API_KEY = "qYCetXYYwF59Mr56eezU"  
WORKSPACE = "university-bswxt"   
PROJECT = "crack-bphdr"       
VERSION = 1                   

def download_dataset():
    print("Starting dataset download...")
    print(f"Workspace: {WORKSPACE}")
    print(f"Project: {PROJECT}")
    print(f"Version: {VERSION}")
    
    rf = Roboflow(api_key=API_KEY)
    
    project = rf.workspace(WORKSPACE).project(PROJECT)
    
    dataset = project.version(VERSION).download("coco-segmentation")
    
    print(f"\n✓ Dataset downloaded to: {dataset.location}")
    print("\nDataset structure:")
    
    for root, dirs, files in os.walk(dataset.location):
        level = root.replace(dataset.location, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  
            print(f'{subindent}{file}')
        if len(files) > 5:
            print(f'{subindent}... and {len(files) - 5} more files')

if __name__ == "__main__":
    download_dataset()