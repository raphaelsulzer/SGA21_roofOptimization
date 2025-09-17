import os
import sys
import json
import random
import shutil
import subprocess

from pathlib import Path
import numpy as np
from PIL import Image


class Annotator:
    
    def __init__(self,config_file="config.json"):
        
        with open(config_file, "r") as f:
            cfg = json.load(f)
        
        self.inpath = Path(cfg.get("inpath", ""))
        self.outpath = Path(cfg.get("outpath", ""))

    def _load_vertices(self, vert_file):
        verts = []
        with open(vert_file, "r") as f:
            for line in f:
                if line.strip():
                    x, y = map(float, line.strip().split(","))
                    verts.append((x, y))
        return verts

    def _load_faces(self, face_file):
        faces = []
        with open(face_file, "r") as f:
            for line in f:
                if line.strip():
                    idxs = list(map(int, line.strip().split(",")))
                    # remove duplicate last index if same as first
                    if idxs[0] == idxs[-1]:
                        idxs = idxs[:-1]
                    faces.append(idxs)
        return faces

    def _polygon_area(self, poly):
        """Shoelace formula"""
        x = np.array(poly[0::2])
        y = np.array(poly[1::2])
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def convert_to_coco(self, split):
        """
        Converts a single image + vert + face into a COCO JSON file
        """
                
        image_path = self.outpath / "data" / "images" / split
        images = os.listdir(image_path)
        poly_id = 0
        
        coco = {
            'info': {'district': 'P3', 'description': 'building footprints', 'contributor': 'RSulzer'},
            'categories': [{'id': 100, 'name': 'building'}],
            'images': [],
            'annotations': [],
        }
        
        for img_id, img_file in enumerate(images):    
        
            im = Image.open(image_path / img_file)
            width, height = im.size
            
            file_id = img_file.split('.')[0]  # Use filename without extension as ID

            coco["images"].append({
                'id': img_id,
                'file_id': file_id, # not used by COCO, but can be useful for getting file_name from id
                'file_name': os.path.join("data", "images", split, img_file),
                'image_path': os.path.join("data", "images", split, img_file),
                'width': width,
                'height': height
            })

            vert_file = self.outpath / "data" / "verts" / split / img_file.replace(".jpg", ".verts")
            face_file = self.outpath / "data" / "faces" / split / img_file.replace(".jpg", ".faces")

            if not vert_file.exists() or not face_file.exists():
                print(f"Missing .vert or .faces file for {img_file}, skipping.")
                continue

            verts = self._load_vertices(vert_file)
            faces = self._load_faces(face_file)

            for face in faces:
                poly = []
                for idx in face:
                    x, y = verts[idx]
                    poly.extend([x, y])
                poly.extend([poly[0],poly[1]])
                
                
                xs = poly[0::2]
                ys = poly[1::2]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                area = self._polygon_area(poly)

                coco["annotations"].append({
                    'id': poly_id,
                    'image_id': img_id,
                    'segmentation': [poly],
                    'area': area,
                    'bbox': bbox,
                    'category_id': 100,
                    'iscrowd': 0,
                })
                poly_id += 1

        out_file = self.outpath / "annotations" / "polygons" / f"annotations_{split}.json"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(out_file, "w") as f:
            json.dump(coco, f, indent=2)

        print(f"Saved COCO annotations to {out_file}")



    def split_dataset(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        """
        Splits dataset into train/val/test and saves text files with image paths.
        Args:
            images (list[str] or list[Path]): list of image paths
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

        random.seed(seed)
        images = os.listdir(os.path.join(self.inpath,"resized","dt_roof_image"))
        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_set = images[:n_train]
        val_set = images[n_train:n_train + n_val]
        test_set = images[n_train + n_val:]

        
        # Save lists
        for split_name, split_set in zip(["train", "val", "test"], [train_set, val_set, test_set]):
            
            img_outpath  = self.outpath / "data" / "images" / split_name
            img_outpath.mkdir(parents=True, exist_ok=True)
            
            vert_outpath = self.outpath / "data" / "verts" / split_name
            vert_outpath.mkdir(parents=True, exist_ok=True)
            
            face_outpath = self.outpath / "data" / "faces" / split_name
            face_outpath.mkdir(parents=True, exist_ok=True)

            for img_file in split_set:
                img_path = self.inpath / "resized" / "dt_roof_image" / img_file
                vert_path = self.inpath / "resized" / "dt_roof_label" / (img_file.replace(".jpg", ".verts"))
                face_path = self.inpath / "resized" / "dt_roof_label" / (img_file.replace(".jpg", ".faces"))
                if not img_path.exists() or not vert_path.exists() or not face_path.exists():
                    print(f"Missing files for {img_file}, skipping.")
                    continue
                shutil.copy(img_path, img_outpath / img_file)
                shutil.copy(vert_path, vert_outpath / (img_file.replace(".jpg", ".verts")))
                shutil.copy(face_path, face_outpath / (img_file.replace(".jpg", ".faces")))

        return train_set, val_set, test_set


    def run_command(self,command):
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
            # Print progress in real time, staying on the same line
            for line in process.stdout:
                if "%" in line:  # Progress lines contain a percentage
                    print(f"\r{line.strip()}", end="")
                    sys.stdout.flush()
                else:
                    print(line, end="")  # Print other lines normally

            # Wait for the process to complete and get the exit code
            process.wait()

            # Optionally print any remaining stderr messages
            stderr = process.stderr.read()
            if stderr:
                print(stderr)

    
    def upload(self, to="gin", copy_annotations=True, copy_data=True):
        
                
        if to == "gin":
            remote_path = "rsgin:/data/rsulzer/RoofGraphDataset/"
        else:
            raise NotImplementedError(f"Copy to {to} not implemented.")
        
        if copy_annotations:
            print(f"\nCopy annotations to {to}:\n")
            
            command = ["rsync", "-ar", "--info=progress2", str(self.inpath / "annotations"), remote_path]
            print(" ".join(command))
            self.run_command(command)

        if copy_data:
            print(f"\nCopy data to {to}:\n")

            command = ["rsync", "-ar", "--info=progress2", str(self.inpath / "data"), remote_path]
            print(" ".join(command))
            self.run_command(command)
