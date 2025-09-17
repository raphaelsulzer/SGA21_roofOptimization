import matplotlib.pyplot as plt

from src.matplotlib_vis import CocoVisualizer


if __name__ == "__main__":
    
    split = "train"
    split = "val"


    img_in = "/home/rsulzer/data/SGA21_roofOptimization/RoofGraphDataset"
    gt_file = f"/home/rsulzer/data/SGA21_roofOptimization/RoofGraphDataset/annotations/polygons/annotations_{split}.json"
    
    
    preds = {}
    
    # exp = "gaussnoise"
    # preds[exp] = f"/home/rsulzer/data/RoofGraphDataset_output/pix2poly/224/{exp}/predictions_rgd_{split}/best_val_iou.json"
    
    exp = "sort_verts_by_coords"
    preds[exp] = f"/home/rsulzer/data/RoofGraphDataset_output/pix2poly/224/{exp}/predictions_rgd_{split}/best_val_iou.json"
    
    exp = "no_sort_verts"
    preds[exp] = f"/home/rsulzer/data/RoofGraphDataset_output/pix2poly/224/{exp}/predictions_rgd_{split}/best_val_iou.json"

    
    ccv = CocoVisualizer(img_in,gt_file,pred_files=preds,plot_bbox=False)
    
    ccv.plot(0)
    plt.show(block=True)
