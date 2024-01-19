from super_gradients.training import models
from config import dataset_params , train_params
import os
import cv2
from pprint import pprint
import glob 
from time import time 

exp_path = "/home/krishnar@iff.intern/thesis/screw_detection/yolo-nas/checkpoints/yolo_nas_s/RUN_20231228_022744_497947"

model = models.get('yolo_nas_s', num_classes=len(dataset_params['classes']), checkpoint_path=os.path.join(exp_path, "ckpt_best.pth"))
input = "/home/krishnar@iff.intern/thesis/datasets/crs/validation/images"
output = "/home/krishnar@iff.intern/thesis/screw_detection/yolo-nas/predictions"
count = 0 
start = time()
input_images = glob.glob( os.path.join(input , '*.JPG'))

for image in input_images:
    count = count +1
    result = model.predict(image, fuse_model=False)
    file_name = image.split("/")[-1] 
    result_item = result._images_prediction_lst[0]
    #pprint(result_item.prediction)

    boxes = result_item.prediction.bboxes_xyxy
    for box in boxes:
        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[2]), int(box[3]))
        np_image = cv2.rectangle(result_item.image, start_point, end_point, (0,0,255), 10) 
    cv2.imwrite(os.path.join(output , file_name) , np_image)

time_taken = time() - start
fps = len(input_images) / time_taken
print(fps)
