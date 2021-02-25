import os
import cv2
import json
import pandas as pd
import numpy as np

video_dir='B20_human11_Camera_1.avi'
gt_dir="AlphaPose_B20_human11_Camera_1.csv"
gt = pd.read_csv(gt_dir, encoding='big5')
gt = gt.iloc[:,1:-2]
gt = gt.values
print(len(gt))
aa=json.load(open('person_keypoints_train2017.json', 'r'))

if not os.path.exists('added_data'):
    os.makedirs('added_data')

count=700000+len(os.listdir('added_data'))
count_gt=0
cap=cv2.VideoCapture(video_dir)
while cap.isOpened():

    ret, frame = cap.read()
    if not ret:
        print("finish")
        break
    cv2.imwrite("added_data/"+str(count)+'.jpg', frame)
    a={
	"license":6,
	"file_name":str(count)+'.jpg',
	"coco_url":"http:\/\/mscoco.org\/added_data/"+str(count)+'.jpg',
	"height":540,"width":720,"date_captured":"2013-11-14 11:18:45",
	"flickr_url":"http:\/\/farm9.staticflickr.com\/8186\/8119368305_4e622c8349_z.jpg",
	"id":count
    }
    aa['images'].append(a)
    if(count_gt==0):
        print(a)
    skeleton=gt[count_gt]

    seg=[]
    xs=[]
    ys=[]
    bbox=[]
    keypoints=[]
    area=0
    num_keypoints=0

    for j in range(17):
        xs.append(skeleton[j*2])
        ys.append(skeleton[j*2+1])
        keypoints.append(skeleton[j*2])
        keypoints.append(skeleton[j*2+1])
        keypoints.append(int(2))
        num_keypoints+=1


    bbox.append(min(xs))
    bbox.append(min(ys))
    bbox.append(max(xs)-min(xs))
    bbox.append(max(ys)-min(ys))

    seg.append(min(xs))
    seg.append(min(ys))
    seg.append(min(xs))
    seg.append(max(ys))
    seg.append(max(xs))
    seg.append(max(ys))
    seg.append(max(xs))
    seg.append(min(ys))

    area=bbox[2]*bbox[3]

    a={'segmentation': [seg],
    'num_keypoints': num_keypoints,
    'area': area,
    'iscrowd': 0,
    'keypoints': keypoints,
    'image_id': count,
    'bbox': bbox,
    'category_id': 1,
    'id': 100000000000+count}
    if (len(keypoints)==51):
        aa['annotations'].append(a)

    if(count_gt==0):
        print(a)
    count+=1
    count_gt+=1
json.dump(aa,open('result.json', 'w'))
