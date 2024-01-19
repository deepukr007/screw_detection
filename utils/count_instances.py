from glob import glob
import os


label_path = "/home/krishnar@iff.intern/thesis/screw_detection/yolov8/labels"

count_images_with_objects =0 
count_total_objects = 0 

conf_threshold = 0 


for label in glob(os.path.join(label_path , '*.txt')):
    count_images_with_objects = count_images_with_objects +1 
    with open(label , 'r') as file:
        objects = file.readlines() 
        count_total_objects = count_total_objects + len(objects)
        

print("Number of labelled images : " , count_images_with_objects)
print("Number of total instances: "  , count_total_objects)


