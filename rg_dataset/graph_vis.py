import os
import json

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as Patches
import matplotlib.pyplot as plt

from skimage import io

from .conversions import get_graph_from_file

class GraphVisualizer:
    def __init__(self, img_in, gt_file, pred_files={}, plot_bbox=True, img_out=None):
        self.path = img_in
        self.img_out = img_out
        self.graph_anns = {}
        self.graph_anns["gt"] = json.load(open(gt_file))
        
        if len(pred_files):
            self.fig, self.ax = plt.subplots(1, 2, figsize=(20, 20))
            self.graph_anns.update(pred_files)
        else:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 20))
            self.ax = [self.ax]
        
        
        self.fig.canvas.mpl_connect('key_press_event', self.update_visualization)
        self.img_id = 0
        self.prediction_id = 0
        
        self.colormap = list(plt.get_cmap('tab10').colors)  # Use a standard matplotlib colormap
        # remove gray
        del self.colormap[-3]
        
        self.plot_bbox = plot_bbox
        
        self.current_image = None
        

    def plot_polygons(self, polygons, axis, pointcolor='magenta', linecolor='turquoise', linestyle='-', linewidth=5, markersize=20, alpha=0.6, outfile=None):


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

    def plot_edges(self, edge, axis, color='cyan', linestyle='-', linewidth=5, alpha=0.6):

        for e in edge:
            line = plt.Line2D(e[:,0], e[:,1], color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
            axis.add_line(line)
            
        return line
        
    def plot_vertices(self, points, axis, color='magenta', markersize=20, alpha=0.6):

        axis.plot(points[:,0], points[:,1], color=color, marker='.', markersize=markersize, linestyle='none', alpha=alpha)
        
    def update_visualization(self, event):
        if event.key == 'right':
            self.img_id += 1
            self.plot(self.img_id)
            plt.draw()
        elif event.key == 'left' and self.img_id > 0:
            self.img_id -= 1
            self.plot(self.img_id)
            plt.draw()
        elif event.key == 'p':
            self.save_screenshot()
            return
        elif event.key == 'up':
            self.prediction_id += 1
            self.plot_pred(self.img_id)
            plt.draw()
            return
        elif event.key == 'down':
            self.prediction_id -= 1
            self.plot_pred(self.img_id)
            plt.draw()
            return
        else:
            return  # Do nothing if left key is pressed and img_id is 0


    def plot(self, img_id):
        
        self.plot_gt(img_id)
        if len(self.graph_anns) > 1:
            self.plot_pred(img_id)

    def plot_pred(self, img_id, axis=1):

        self.ax[axis].clear()
        self.ax[axis].axis('off')
        
        
        pred_keys = list(self.graph_anns.keys() - {"gt"})
        prediction_id = self.prediction_id % len(pred_keys)  # Wrap around
        key = pred_keys[prediction_id]  # First prediction key
        
        graph_file_name = os.path.join(self.graph_anns[key],f"{img_id}.npz")
        
        self.ax[axis].set_title(f"Prediction", fontsize=20)
        
        # plot image                
        self.plot_image(self.current_image, self.ax[axis])
        
        # plot graph
        graph = get_graph_from_file(os.path.join(self.path, graph_file_name))

        verts = np.array(list(nx.get_node_attributes(graph, 'pos').values()))
        edges = verts[list(graph.edges)]
        
        one_edge = self.plot_edges(edges, self.ax[axis], linewidth=2, alpha=0.8)
        self.plot_vertices(verts, self.ax[axis], markersize=10, alpha=0.8)
        

        self.ax[axis].legend([one_edge], [f"Prediction {key}"], loc='upper right', fontsize=15)

    def plot_gt(self, img_id, key='gt', axis=0):
                
        self.ax[axis].clear()
        self.ax[axis].axis('off')
        
        img_id = list(self.graph_anns[key].keys())[img_id]
        data = self.graph_anns[key][img_id]

        img_file = data['image']['image_path']
        filename = img_file
        img_file = os.path.join(self.path, img_file)
        if not os.path.isfile(img_file):
            raise FileNotFoundError(f"Image file {img_file} not found")

        self.ax[axis].set_title(f"Ground truth {filename}({img_id})", fontsize=20)
        
        # plot image                
        img = io.imread(img_file)
        self.current_image = img
        self.plot_image(img, self.ax[axis])
        
        # plot graph
        graph = get_graph_from_file(os.path.join(self.path, data['graph']['graph_path']))

        verts = np.array(list(nx.get_node_attributes(graph, 'pos').values()))
        edges = verts[list(graph.edges)]
        
        one_edge = self.plot_edges(edges, self.ax[axis], linewidth=2, alpha=0.8)
        self.plot_vertices(verts, self.ax[axis], markersize=10, alpha=0.8)
        
        self.ax[axis].legend([one_edge], ["Ground Truth"], loc='upper right', fontsize=15)
        
        return img


    def save_screenshot(self):
        if self.img_out:
            filename = self.graph_anns["gt"].imgs[self.img_id]['file_name']
            outfile = os.path.join(self.img_out, os.path.splitext(filename)[0] + ".jpg")
            self.fig.savefig(outfile, bbox_inches='tight', pad_inches=0.0, dpi=250)
            self.logger.info(f"Screenshot saved to {outfile}")
        else:
            self.logger.error("No output directory specified")

