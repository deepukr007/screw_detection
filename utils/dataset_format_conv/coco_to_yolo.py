from ultralytics.data.converter import convert_coco

in_path = '/home/krishnar@iff.intern/thesis/datasets/WDSD/Annotations/Object Detection _ Instance Segmentation COCO-format'
out_path = '/home/krishnar@iff.intern/thesis/datasets/WDSD/labels'

convert_coco(in_path, save_dir  = out_path ,  use_segments=False, use_keypoints=False, cls91to80=True)