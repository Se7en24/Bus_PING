# scripts/split_dataset.py
import os, random, shutil
from glob import glob

src_img = "data/images"
src_lbl = "data/labels"
os.makedirs("data/train/images", exist_ok=True)
os.makedirs("data/train/labels", exist_ok=True)
os.makedirs("data/valid/images", exist_ok=True)
os.makedirs("data/valid/labels", exist_ok=True)

files = [os.path.basename(f) for f in glob(src_img+"/*.jpg")]
random.shuffle(files)
split = int(len(files)*0.8)
train = files[:split]
valid = files[split:]

for f in train:
    shutil.copy(os.path.join(src_img,f), "data/train/images/"+f)
    lbl = f.rsplit(".",1)[0]+".txt"
    shutil.copy(os.path.join(src_lbl,lbl), "data/train/labels/"+lbl)

for f in valid:
    shutil.copy(os.path.join(src_img,f), "data/valid/images/"+f)
    lbl = f.rsplit(".",1)[0]+".txt"
    shutil.copy(os.path.join(src_lbl,lbl), "data/valid/labels/"+lbl)

print("Done: train", len(train), "valid", len(valid))
