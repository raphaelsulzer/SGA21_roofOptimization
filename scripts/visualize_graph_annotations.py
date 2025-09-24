import matplotlib.pyplot as plt

from rg_dataset.coco_vis import CocoVisualizer
from rg_dataset.graph_vis import GraphVisualizer


if __name__ == "__main__":
    
    split = "train"
    split = "val"
    split = "test"


    img_in = "/home/rsulzer/python/rg_dataset/RoofGraphDataset"
    gt_file = f"/home/rsulzer/python/rg_dataset/RoofGraphDataset/annotations/graphs/annotations_{split}.json"
    
    
    preds = {}
    experiments = ["predict_adj_plw10", "predict_adj_cvp_1_5_25"]
    for exp in experiments:
        preds[exp] = f"/home/rsulzer/data/RoofGraphDataset_output/pix2poly/224/{exp}/predictions_rgd_{split}/best_val_iou/graphs"

    ccv = GraphVisualizer(img_in,gt_file,pred_files=preds,plot_bbox=False)

    ccv.plot(0)
    plt.show(block=True)
