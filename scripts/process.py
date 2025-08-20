from python.annotator import Annotator


if __name__ == "__main__":
    
    aa = Annotator()
    
    # aa.split_dataset()
    
    splits = ["train", "val", "test"]
    
    for split in splits:

        aa.convert_to_coco(split=split)

    aa.upload(copy_data=False)