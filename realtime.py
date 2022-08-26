import numpy as np
import cv2
import torch
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")                   # Setup Device Reccomend to use GPU
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval().to(device)     # Load Model Fasterrcnn
transform = transforms.Compose([transforms.ToTensor()])                                 # transform to tensor

# select the second camera by passing 1 and so on
cap = cv2.VideoCapture(0)  

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Model
    threshold = 0.75
    pred = model([transform(Image.fromarray(frame)).to(device)]) # load | transform | predict incoming frame
    pred_score = list(pred[0]['scores'].cpu().detach().numpy())  # convert prediction to numpy 
    
    # filter scores according threshold
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    #  perlu di adjust  
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())][:pred_t+1]   
         # filter classes within the threshold

    pred_boxes = [[(int(i[0]),int( i[1])), (int(i[2]), int(i[3]))] for i in list(pred[0]['boxes'].cpu().detach().numpy())][:pred_t+1]   # filter boxes within the threshold                                    
    
    # Our operations on the frame 
    for i in range(len(pred_boxes)):
        col= (0, 255,0)
        print("range - ", i," = ", pred_boxes[i][0],"/", pred_boxes[i][1])
        

        # cv2.rectangle(frame,(4, 24) , (1000,100),col, thickness=2) 
        cv2.rectangle(frame, pred_boxes[i][0], pred_boxes[i][1],col, thickness=2)    
        # cv2.line(frame,(coord[0][0],coord[0][1]),(coord[1][0],coord[1][1]),(0,0,255),2)             # Draw Rectangle with the coordinates
        cv2.putText(frame,pred_class[i], pred_boxes[i][0],  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0),thickness=1) # Write the prediction class
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()