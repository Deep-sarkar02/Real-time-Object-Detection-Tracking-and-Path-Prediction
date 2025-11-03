import torch
import torchvision
from torchvision import transforms as tf

from PIL import Image
import cv2 as cv

"""##Importing the module"""

from torchvision import models


"""Loading the pretrained faster R-CNN model"""

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=1)

"""Evaluating the model"""

model.eval()

"""Loading the image in colab"""


"""Reading the image"""

ig = Image.open("/Users/tamaldas/Documents/Project/FasterRCNN/img.jpg")

"""Transforming the image"""

transform = tf.ToTensor()
img = transform(ig)


with torch.no_grad():
    pred = model([img])

pred[0].keys()

bboxes, labels, scores = pred[0]["boxes"], pred[0]["labels"], pred[0]["scores"]

"""saving the number of confident detection [where score is more than 90%]"""

num = torch.argwhere(scores > 0.89).shape[0]

"""name of classes used in form of list"""

coco_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
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

# drawing bbox on the frame
test = cv.imread("/Users/tamaldas/Documents/Project/FasterRCNN/img.jpg")
for i in range(num):
    x1, y1, x2, y2 = bboxes[i].numpy().astype("int")
    print(x1, y1, x2, y2)
    class_name = coco_names[labels.numpy()[i] - 1]
    inp = cv.rectangle(test, (x1, y1), (x2, y2), (0, 255, 1))
    inp = cv.putText(test, class_name, (x1, y1 - 10), font, 0.5, (255, 0, 0), 2, cv.LINE_AA)

cv.imshow("result", test)
cv.waitKey(0)
