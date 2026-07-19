import os
import shutil
import random

# Paths
source_dir = "dataset/images/train"
val_dir = "dataset/images/val"

# Create val folder if not exists
os.makedirs(val_dir, exist_ok=True)

# Loop through each class
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    val_class_path = os.path.join(val_dir, class_name)

    os.makedirs(val_class_path, exist_ok=True)

    images = os.listdir(class_path)
    random.shuffle(images)

    split_size = int(0.2 * len(images))  # 20%

    val_images = images[:split_size]

    for img in val_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(val_class_path, img)
        shutil.move(src, dst)

print("✅ Validation split completed!")