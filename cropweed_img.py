import torch
import numpy as np
import cv2
from time import time

model = torch.hub.load('ultralytics/yolov5', 'custom', path='C:/Users/User/Documents/EBTECH/EB_Torch/cropweed.pt', force_reload=False)
classes = model.names
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using Device: ", device)
def score_frame(frame):
    model.to(device)
    frame = [frame]
    results = model(frame)
    labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    return labels, cord

def class_to_label(x):
    return classes[int(x)]

def plot_boxes(results, frame):
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
      row = cord[i]
      score = row[4].cpu().numpy()
    #   name = "{} :{}%".format(class_to_label(labels[i]),str(int(score*100)))
      name = "{}".format(class_to_label(labels[i]))
      print(name)
      if row[4] >= 0.3:
          x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
          bgr = (255, 0, 00) if class_to_label(labels[i])=="weed" else (0, 0, 255) 
          cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 4)
          cv2.putText(frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, bgr, 2)

    return frame

img = cv2.imread("C:/Users/User/Documents/EBTECH/EB_Dataset/cropvsweed/test/images/agri_0_76.jpeg")
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
results = score_frame(img)
frame = plot_boxes(results, img)
frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
# cv2.imwrite("C:/Users/User/Desktop/agri_0_76.jpeg", frame)
cv2.imshow("window_name", frame)
cv2.waitKey(0) 
cv2.destroyAllWindows() 



