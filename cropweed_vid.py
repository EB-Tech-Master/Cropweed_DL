import torch
import cv2

VIDEO_PATH = "C:/Users/User/Desktop/VideoP1.mp4"
OUT_NAME = 'C:/Users/User/Desktop/DrNish/torch_cropweed.mp4'
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
          Mox = int((x1+x2)/2)
          Moy = int((y1+y2)/2)
          mid2 = (Mox  , Moy)
          cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 4)
          cv2.putText(frame, name, (Mox, Moy), cv2.FONT_HERSHEY_SIMPLEX, 1, bgr, 2)
    return frame

# Open video file
video = cv2.VideoCapture(VIDEO_PATH)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
# out = cv2.VideoWriter(OUT_NAME,cv2.VideoWriter_fourcc(*'MP4V'), 30, (int(imW),int(imH)))

while(video.isOpened()):
    ret, img = video.read()
    if not ret:
        print('Reached the end of the video!')
        break
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = score_frame(img)
    frame = plot_boxes(results, img)
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    # out.write(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cv2.destroyAllWindows()