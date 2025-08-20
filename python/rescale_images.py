import os
import shutil

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx

from tqdm import tqdm

def get_dimensions(folder):

    widths = []
    heights = []

    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            path = os.path.join(folder, filename)
            with Image.open(path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)

    # Convert lists to numpy arrays for easier handling
    widths = np.array(widths)
    heights = np.array(heights)

    # Define bins with step size 100 px
    max_width = widths.max()
    max_height = heights.max()

    bins_width = np.arange(0, max_width + 100, 100)
    bins_height = np.arange(0, max_height + 100, 100)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=bins_width, edgecolor='black')
    plt.title('Width distribution')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Number of images')

    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=bins_height, edgecolor='black')
    plt.title('Height distribution')
    plt.xlabel('Height (pixels)')
    plt.ylabel('Number of images')

    plt.tight_layout()
    plt.show()




def num_vert_histogram(vert_folder, bin_size=5):
    vert_counts = []
    for filename in os.listdir(vert_folder):
        if filename.lower().endswith('.verts'):
            path = os.path.join(vert_folder, filename)
            with open(path, 'r') as f:
                count = sum(1 for line in f if line.strip())
                vert_counts.append(count)
    if vert_counts:
        bins = range(0, max(vert_counts) + 3, bin_size)
        plt.figure(figsize=(8, 5))
        plt.hist(vert_counts, bins=bins, edgecolor='black')
        plt.title('Vertex Count Distribution')
        plt.xlabel('Number of Vertices')
        plt.ylabel('Number of Files')
        plt.tight_layout()
        plt.show()
    else:
        print("No .verts files found in the folder.")
    


def read_graph_from_verts_and_faces(verts_path, faces_path):
    G = nx.Graph()
    verts = []
    with open(verts_path, 'r') as vf:
        for i, line in enumerate(vf):
            line = line.strip()
            if not line:
                continue
            x_str, y_str = line.split(',')
            x, y = float(x_str), float(y_str)
            verts.append((x, y))
            G.add_node(i, pos=(x, y))
    with open(faces_path, 'r') as ff:
        for line in ff:
            line = line.strip()
            if not line:
                continue
            if ',' in line:
                indices = [int(idx) for idx in line.split(',')]
            else:
                indices = [int(idx) for idx in line.split()]
            for i in range(len(indices)):
                v1 = indices[i]
                v2 = indices[(i + 1) % len(indices)]
                G.add_edge(v1, v2)
    return G


def plot_graph_on_images(image_folder, ann_folder, output_folder, target_size=224):
    
    os.makedirs(output_folder, exist_ok=True)

    image_files = os.listdir(image_folder)
    
    for filename in tqdm(image_files):
        if not filename.lower().endswith(('.jpg', '.jpeg')):
            continue
        
        basename = os.path.splitext(filename)[0]

        img_path = os.path.join(image_folder, filename)
        verts_path = os.path.join(ann_folder, basename + '.verts')
        faces_path = os.path.join(ann_folder, basename + '.faces')

        if not (os.path.exists(verts_path) and os.path.exists(faces_path)):
            print(f"Skipping {filename}: missing verts or faces file")
            continue

        # Load image and graph
        img = Image.open(img_path)
        G = read_graph_from_verts_and_faces(verts_path, faces_path)
        
        planar = nx.is_planar(G)
        if not planar:
            print(f"Graph for {filename} is not planar, skipping.")
            continue

        # # Extract node positions as dict {node: (x, y)}
        pos = nx.get_node_attributes(G, 'pos')

        # # Plot
        dpi = 100
        fig_w = target_size / dpi
        fig_h = target_size / dpi
        plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        
        plt.imshow(img)
        ax = plt.gca()
        ax.invert_yaxis()  # Because image origin is top-left, matplotlib y-axis is bottom-left

        # Draw edges in turquoise
        for (u, v) in G.edges():
            x_vals = [pos[u][0], pos[v][0]]
            y_vals = [pos[u][1], pos[v][1]]
            plt.plot(x_vals, y_vals, color=(64/255, 224/255, 208/255), linewidth=3)  # turquoise

        # Draw vertices in magenta
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        plt.scatter(xs, ys, c='magenta', s=30, edgecolors=None, linewidths=0.5, zorder=2)

        plt.axis('off')
        save_path = os.path.join(output_folder, filename)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved graph overlay for {filename}")


def pad_images_with_keypoints(input_folder, output_folder, target_size=500):
    
    print(f"Padding images from {input_folder} to {output_folder} with target size {target_size}")
    
    os.makedirs(output_folder, exist_ok=True)

    image_input_folder = os.path.join(input_folder, "dt_roof_image")
    annotation_input_folder = os.path.join(input_folder, "dt_roof_label")
    
    image_output_folder = os.path.join(output_folder, "dt_roof_image")
    os.makedirs(image_output_folder, exist_ok=True)
    annotation_output_folder = os.path.join(output_folder, "dt_roof_label")
    os.makedirs(annotation_output_folder, exist_ok=True)
    
    for filename in tqdm(os.listdir(image_input_folder)):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            # Process image
            img_path = os.path.join(image_input_folder, filename)
            with Image.open(img_path) as img:
                w, h = img.size
                if w < target_size and h < target_size:
                    # Create black background
                    new_img = Image.new('RGB', (target_size, target_size), (0, 0, 0))
                    left = (target_size - w) // 2
                    top = (target_size - h) // 2
                    new_img.paste(img, (left, top))
                    
                    # Save padded image
                    save_img_path = os.path.join(image_output_folder, filename)
                    new_img.save(save_img_path)

                    # Now handle the .verts file
                    verts_filename = os.path.splitext(filename)[0] + '.verts'
                    verts_path = os.path.join(annotation_input_folder, verts_filename)
                    if os.path.exists(verts_path):
                        with open(verts_path, 'r') as vf:
                            lines = vf.readlines()
                        
                        adjusted_points = []
                        for line in lines:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                x_str, y_str = line.split(',')
                                x = float(x_str)
                                y = float(y_str)
                                # Adjust by padding offset
                                x_adj = x + left
                                y_adj = y + top
                                adjusted_points.append(f"{x_adj},{y_adj}\n")
                            except Exception as e:
                                print(f"Skipping invalid line in {verts_filename}: {line} ({e})")
                        
                        # Save adjusted verts file
                        save_verts_path = os.path.join(annotation_output_folder, verts_filename)
                        with open(save_verts_path, 'w') as outf:
                            outf.writelines(adjusted_points)
                            
                        # Now also copy the faces file
                        shutil.copy(os.path.join(annotation_input_folder, os.path.splitext(filename)[0] + '.faces'), annotation_output_folder)
                        
                        
                        
                else:
                    # Image discarded, do nothing
                    pass



def resize_images_with_keypoints(input_folder, output_folder, target_size=224, original_size=500):
    
    print(f"Resizing images from {input_folder} to {output_folder} with target size {target_size} and original size {original_size}")
    
    image_input_folder = os.path.join(input_folder, "dt_roof_image")
    annotation_input_folder = os.path.join(input_folder, "dt_roof_label")
    
    image_output_folder = os.path.join(output_folder, "dt_roof_image")
    os.makedirs(image_output_folder, exist_ok=True)
    annotation_output_folder = os.path.join(output_folder, "dt_roof_label")
    os.makedirs(annotation_output_folder, exist_ok=True)
    

    scale = target_size / original_size

    for filename in tqdm(os.listdir(image_input_folder)):
        if filename.lower().endswith(('.jpg', '.jpeg')):
            basename = os.path.splitext(filename)[0]

            img_path = os.path.join(image_input_folder, filename)
            verts_path = os.path.join(annotation_input_folder, basename + '.verts')

            # Load image and resize
            with Image.open(img_path) as img:
                img_resized = img.resize((target_size, target_size), Image.BILINEAR)
                img_resized.save(os.path.join(image_output_folder, filename))

            # Adjust verts if exists
            if os.path.exists(verts_path):
                with open(verts_path, 'r') as vf:
                    lines = vf.readlines()

                adjusted_points = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        x_str, y_str = line.split(',')
                        x = float(x_str) * scale
                        y = float(y_str) * scale
                        adjusted_points.append(f"{x},{y}\n")
                    except Exception as e:
                        print(f"Skipping invalid line in {verts_path}: {line} ({e})")

                save_verts_path = os.path.join(annotation_output_folder, basename + '.verts')
                with open(save_verts_path, 'w') as outf:
                    outf.writelines(adjusted_points)

                # Now also copy the faces file
                shutil.copy(os.path.join(annotation_input_folder, os.path.splitext(filename)[0] + '.faces'), annotation_output_folder)

if __name__ == "__main__":
    
    # TODO: 1
    # Check if pix2poly training and inference allow more than one connections per vertex
    # the paper says no, but where is the mech that disallows it?
    # see if the row or col sums are > 1 in the adjacency matrix for the building dataset
    
    # ANSWER:
    # adjacency matrix which is not a permutation matrix, i.e. not a binary matrix with exactly one 1 in each row and column, is not compatible with pix2poly training. The sinkhorn algorithm requires a permutation matrix during training.
    # Two possible solutions: Try to implement Re:PolyWorld, which addresses exactly this limitation. Or, try to predict an adjacency matrix, and use e.g. gumbel sigmoid to make it binary.. Also check how BSPNet does it, they also predict an adjacency matrix.
    
    # TODO: 2
    # make pix2poly training data out of this. should be enough to just store the graph adjacency matrix and vert list
    

    
    # TODO: 3
    # for a first test, convert wireframes to polygons and run pix2poly on that
    
    
    
    infolder_path = "/home/rsulzer/data/SGA21_roofOptimization/RoofGraphDataset"
    outfolder_path = "/home/rsulzer/data/SGA21_roofOptimization/RoofGraphDataset/padded"
    
    
    num_vert_histogram(os.path.join(infolder_path, "dt_roof_label"))
    
    target_size = 500  # Target size for padding
    
    # pad_images_with_keypoints(infolder_path, output_folder=outfolder_path, target_size=target_size)
    
    infolder_path = outfolder_path
    outfolder_path = "/home/rsulzer/data/SGA21_roofOptimization/RoofGraphDataset/resized"
    
    # resize_images_with_keypoints(infolder_path, outfolder_path, target_size=224, original_size=target_size)
    
    infolder_path = outfolder_path
    image_folder = os.path.join(infolder_path, "dt_roof_image")
    ann_folder = os.path.join(infolder_path, "dt_roof_label")
    output_folder = os.path.join(infolder_path, "dt_plots")
    
    plot_graph_on_images(image_folder, ann_folder, output_folder)