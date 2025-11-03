# importing the required modules
import torch
import torchvision
from torchvision import transforms as tf
import cv2 as cv
from torchvision import models
# loading the pretrained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=1)

# evaluating the model
model.eval()

# loading the video
vid = cv.VideoCapture("/Users/tamaldas/Documents/Project/FasterRCNN/sample2.mp4")

# Get input video properties
fps = vid.get(cv.CAP_PROP_FPS)
width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))
print("video properties: ", fps, width, height)

# output video
# Define output video properties
output_video_path = 'output_video.mp4'
fourcc = cv.VideoWriter_fourcc(*'mp4v')
output_video = cv.VideoWriter(output_video_path, fourcc, fps, (width, height))

# looping through each frame
while True:
    # reading the frame
    ret, frame = vid.read()

    if not ret:
        break
    
    # transforming the frame
    transform = tf.ToTensor()
    img = transform(frame)

    # training the model
    with torch.no_grad():
        pred = model([img])

    pred[0].keys()
    bboxes, labels, scores = pred[0]["boxes"], pred[0]["labels"], pred[0]["scores"]

    # saving the number of confident detection [where score is more than 90%]
    num = torch.argwhere(scores > 0.85).shape[0]

    # name of classes used in form of list
    coco_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                  "traffic light",
                  "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
                  "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe",
                  "eye glasses", "handbag", "tie", "suitcase",
                  "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                  "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                  "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
                  "pizza", "donut", "cake", "chair", "couch", "rope", "potted plant", "bed",
                  "mirror", "dining table", "window", "desk", "toilet", "door", "tv",
                  "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
                  "oven", "toaster", "sink", "refrigerator", "blender", "book",
                  "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hair brush", ]

    font = cv.FONT_HERSHEY_SIMPLEX

    # testing the model with video
    for i in range(num):
        x1, y1, x2, y2 = bboxes[i].numpy().astype("int")
        # print(x1, y1, x2, y2)
        # calculating the centroid of  the bounding box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        class_name = coco_names[labels.numpy()[i] - 1]

        # create  a circle in the middle of the bounding box
        cv.circle(frame, (cx, cy), 10, (0, 0, 255), - 1)
        inp = cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 2))
        inp = cv.putText(frame, class_name, (x1, y1 - 10), font, 0.5, (255, 0, 0), 2, cv.LINE_AA)

    # cv.imshow("result", frame)
    output_video.write(frame)
    # Break the loop if 'q' is pressed
    # if cv.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release resources
vid.release()
output_video.release()
cv.destroyAllWindows()
