import os
import cv2
import shutil
import imagehash
from PIL import Image
import random
from tqdm import tqdm

class_mapping = {
    'Acne': ['acne'],
    'Rosacea': ['rosacea'],
    'Eczema': ['eczema', 'eczema photos', 'atopic dermatitis photos', 'poison ivy photos and other contact dermatitis'],
    'Normal': ['heathy', 'normal']
}
MAX_PER_CLASS = 250

folder_to_class = {}
for target_class, folder_names in class_mapping.items():
    for name in folder_names:
        folder_to_class[name.lower()] = target_class

raw_dir = 'data/raw'
processed_dir = 'data/processed'

if os.path.exists(processed_dir):
    shutil.rmtree(processed_dir)
os.makedirs(processed_dir, exist_ok=True)

for target_class in class_mapping.keys():
    os.makedirs(os.path.join(processed_dir, target_class), exist_ok=True)

class_paths = {k: [] for k in class_mapping.keys()}
for root, dirs, files in os.walk(raw_dir):
    folder_name = os.path.basename(root).lower()
    if folder_name in folder_to_class:
        target_class = folder_to_class[folder_name]
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                class_paths[target_class].append(os.path.join(root, file))

for k in class_paths.keys():
    random.seed(42) # Reproducibility
    random.shuffle(class_paths[k])

seen_hashes = set()
stats = {k: {'total_scanned': 0, 'deduped': 0} for k in class_mapping.keys()}

print("Beginning deduplication and subset extraction...")

for target_class, paths in class_paths.items():
    print(f"\nProcessing {target_class}...")
    pbar = tqdm(total=MAX_PER_CLASS)
    
    for img_path in paths:
        if stats[target_class]['deduped'] >= MAX_PER_CLASS:
            break
            
        stats[target_class]['total_scanned'] += 1
        
        img = cv2.imread(img_path)
        if img is None: 
            continue
            
        try:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img_hash = str(imagehash.phash(pil_img))
        except Exception:
            continue
            
        if img_hash in seen_hashes: 
            continue
            
        seen_hashes.add(img_hash)
        stats[target_class]['deduped'] += 1
        
        ext = os.path.splitext(img_path)[1]
        new_path = os.path.join(processed_dir, target_class, f"{img_hash}{ext}")
        shutil.copy2(img_path, new_path)
        
        pbar.update(1)
        
    pbar.close()

print("\n--- Deduplication Complete ---")
for k, v in stats.items():
    print(f"{k}: Scanned = {v['total_scanned']} -> Final Curated = {v['deduped']}")
