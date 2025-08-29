import os
from skimage import io
import numpy as np
from pycocotools.coco import COCO
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import matplotlib.patches as Patches
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

class CocoPoly:
    
    def __init__(self, geom, bbox, id):
        self.geom = geom
        self.bbox = bbox
        self.id = id

class CocoVisualizer:
    def __init__(self, img_in, gt_file, pred_files={}, plot_bbox=True, img_out=None):
        self.path = img_in
        self.img_out = img_out
        self.coco_anns = {}
        self.coco_anns["gt"] = COCO(gt_file)
        
        if pred_files:
            for k,v in pred_files.items():
                self.coco_anns[k] = self.coco_anns["gt"].loadRes(v)
                self.fig, self.ax = plt.subplots(1, 2, figsize=(20, 20))
        else:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 20))
            self.ax = [self.ax]
        
        
        self.fig.canvas.mpl_connect('key_press_event', self.update_visualization)
        self.img_id = 0
        
        self.colormap = list(plt.get_cmap('tab10').colors)  # Use a standard matplotlib colormap
        # remove gray
        del self.colormap[-3]
        
        self.plot_bbox = plot_bbox
        

    def plot_polygons(self, polygons, axis, pointcolor='yellow', linecolor='red', linestyle='-', linewidth=5, markersize=20, alpha=0.6, outfile=None):


        for poly in polygons:
            
            mpl_poly = Patches.Polygon(poly.geom.exterior.coords[:-1], fill=False, ec=linecolor, linewidth=linewidth, linestyle=linestyle, alpha=alpha)
            axis.add_patch(mpl_poly)
            juncs = np.array(poly.geom.exterior.coords[:-1])
            axis.plot(juncs[:, 0], juncs[:, 1], color=pointcolor, marker='.', markersize=markersize, linestyle='none')
            
            if self.plot_bbox:
                axis.add_patch(Patches.Polygon(poly.bbox.exterior.coords[:-1], fill=False, ec=linecolor, linewidth=linewidth, linestyle='--'))
            

        if outfile is not None:
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            plt.savefig(outfile, bbox_inches='tight', pad_inches=0.0)

        return mpl_poly

    def plot_image(self, image, axis):
        axis.imshow(image)
        # Set ticks on x and y axes at the image dimensions
        height, width = image.shape[:2]
        axis.set_xticks(np.linspace(0, width, num=11,dtype=np.int_))  # 11 points to include both 0 and width
        axis.set_yticks(np.linspace(0, height, num=11,dtype=np.int_))   # Adjust step as needed

    def get_polygons_from_coco(self, coco_annotations, img_id):
        annotations = coco_annotations.imgToAnns[img_id]
        polygons = []
        for ann in annotations:
            if not len(ann.get('segmentation')):
                print(f"Strange annotation without segmentation in image {img_id}")
                continue
            poly = np.array(ann.get('segmentation')[0])
            poly = poly.reshape(int(len(poly) / 2), 2)
            
            xmin, ymin, w, h = ann.get('bbox')
            bbox_poly = Polygon([(xmin, ymin), (xmin + w, ymin), (xmin + w, ymin + h), (xmin, ymin + h)])
            
            
            poly = CocoPoly(Polygon(poly), bbox_poly, ann.get('id'))
            polygons.append(poly)
        return polygons


    def update_visualization(self, event):
        if event.key == 'right':
            self.img_id += 1
        elif event.key == 'left' and self.img_id > 0:
            self.img_id -= 1
        elif event.key == 'p':
            self.save_screenshot()
            return
        else:
            return  # Do nothing if left key is pressed and img_id is 0

        self.plot(self.img_id)
        plt.draw()


    def plot(self, img_id):
        
        for ax in self.ax:
            ax.clear()
            ax.axis('off')

        img_id = self.coco_anns["gt"].getImgIds()[img_id]
        
        filename = self.coco_anns["gt"].imgs[img_id]['file_name']
        img_file = os.path.join(self.path, filename)
        if not os.path.isfile(img_file):
            raise FileNotFoundError(f"Image file {img_file} not found")
        
        img = io.imread(img_file)
        self.plot_image(img, self.ax[0])
        self.ax[0].set_title(f"Ground truth {filename}({img_id})", fontsize=20)
        if len(self.ax) > 1:
            self.plot_image(img, self.ax[1])
            self.ax[1].set_title(f"Prediction {filename}({img_id})", fontsize=20)

        
        colormap = plt.get_cmap('tab10')
        colors = [colormap(i % colormap.N) for i in range(len(self.coco_anns) - 1)]
        i = 0
        legend = {}
        for k,v in self.coco_anns.items():
            
            polygons = self.get_polygons_from_coco(v, img_id)

            if k == "gt":
                linecolor = 'green'
                poly = self.plot_polygons(polygons, self.ax[0], linecolor=linecolor)
                
                self.ax[0].legend([poly], k, loc='upper right', fontsize=16)

            else:
                poly = self.plot_polygons(polygons, self.ax[1], linecolor=colors[i], pointcolor=colors[i])
                legend[k] = poly

                i+=1
            
        self.ax[1].legend(legend.values(), legend.keys(), loc='upper right', fontsize=16)

    def save_screenshot(self):
        if self.img_out:
            filename = self.coco_anns["gt"].imgs[self.img_id]['file_name']
            outfile = os.path.join(self.img_out, os.path.splitext(filename)[0] + ".jpg")
            self.fig.savefig(outfile, bbox_inches='tight', pad_inches=0.0, dpi=250)
            self.logger.info(f"Screenshot saved to {outfile}")
        else:
            self.logger.error("No output directory specified")

