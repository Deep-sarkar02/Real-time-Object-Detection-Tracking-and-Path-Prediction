# with this program i want to edit the CSV file
# in such a way that the CSV will
# contain only those imageIDs which are
# present in the training folder
# to do that..
# i will read the name of the images that i have in the training folder
# and check if it is in the main CSV
# if it is there, I will save all info in another CSV
import os
import pandas as pd

# bbox details link "/Users/tamaldas/Documents/Project/FasterRCNN/jhu_crowd_v2.0/train/gt"
# image names link "/Users/tamaldas/Documents/Project/FasterRCNN/jhu_crowd_v2.0/train/images"

# listing the images names
# respective url of the label folder
imgName = os.listdir('/Users/tamaldas/Documents/Project/FasterRCNN/jhu_crowd_v2.0/train/images')
print("Number of Images in folder: ", len(imgName))

# remove the .txt form list elements
idList = []
for ele in imgName:
    ele = ele[:-4]
    idList.append(ele)

print("List of ImageIDs: ", idList)

# reading the main CSV
# url of the CSV file
# mainCSVFile = pd.read_csv("train-annotations-bbox.csv")

# # discarding imagesID that are missing from idList
# # print(mainCSVFile['ImageID'])
# discardList = []
# for index, row in mainCSVFile.iterrows():
#     if row['ImageID'] not in idList:
#         discardList.append(index)
#         # print(row)
#         # dropping records which does not exist in idList

# print("Number of index to drop: ", len(discardList))
# # param:
# # uncomment to drop sorted index
# mainCSVFile.drop(discardList, axis=0, inplace=True)
