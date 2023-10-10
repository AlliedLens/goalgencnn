#!/usr/bin/env python

import rospy
import cv2
import random
import numpy as np
from nav_msgs.msg import OccupancyGrid
import json
import torch
from torchvision import models
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms

mobileNetPretrained = models.mobilenet_v2(pretrained=True)
num_classes = 2
mobileNetPretrained.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(mobileNetPretrained.last_channel, num_classes)
)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(mobileNetPretrained.parameters(), lr=0.0008)

model = mobileNetPretrained
model.load_state_dict(torch.load("/home/vk2494/igvc_ws/src/mobileNetV2GoalGeneration.pth"))

class mapDataset(Dataset):

  def __init__(self, images, labels):
    self.images = images
    self.labels = labels
  def __len__(self):
    return len(self.images)

  def __getitem__(self, index):
    imgSample = self.images[index]
    labelSample = self.labels[index]
    return [imgSample, labelSample]

def loadImages(directory):
  data = []
  for file in sorted(os.listdir(directory)):
      img_path = os.path.join(directory, file)
      if (file != 'imgCoordinates.json'):
        img = cv2.imread(img_path)
        img[np.all(img == (255,255,255), axis=-1)] = (0,0,255)
        img[np.all(img == (100,100,100), axis=-1)] = (255,0,0)
        img[np.all(img == (0,0,0), axis=-1)] = (0,255,0)
        img = cv2.resize(img, (224,224))
        data.append(img)

  #cv2_imshow(data[126])
  return data

def loadLabels(directory, jsonFile):
  data = []
  jsonDictionary = {}

  with open(os.path.join(directory, jsonFile), 'r') as js:
    json_data = js.read()

  jsonDictionary = json.loads(json_data)

  for file in sorted(os.listdir(directory)):
    if (file != 'imgCoordinates.json'):
      entry = jsonDictionary["database/"+file]
      values = entry.strip('()')
      values = values.split(',')
      values = [int(val) for val in values]
      values = np.array(values)
      data.append(values)

  return data

def modelTraining():    
    data = loadImages('/content/drive/MyDrive/newDatabase')
    data = np.array(data)
    
    trainImages = np.array(data)
    min_val = np.min(trainImages)
    max_val = np.max(trainImages)
    
    trainImages = (trainImages - min_val) / (max_val - min_val)
    
    data = loadLabels('/content/drive/MyDrive/newDatabase', 'imgCoordinates.json')
    
    trainLabels = np.array(data)
    trainLabels = (trainLabels)/400
    mobileNetPretrained = models.mobilenet_v2(pretrained=True)
    num_classes = 2
    mobileNetPretrained.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(mobileNetPretrained.last_channel, num_classes)
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(mobileNetPretrained.parameters(), lr=0.00075)
    EPOCHS = 7
    model = mobileNetPretrained
    losses=[]
    for epoch in range(EPOCHS):
      runningLoss = 0
      for i, (inputs, targets) in enumerate(trainDl):
        optimizer.zero_grad()
        inputs = inputs.to(torch.float32)
        inputs = inputs.view((inputs.shape[0], inputs.shape[3],inputs.shape[2],inputs.shape[1]))
        yhat = model(inputs)
        targets = targets.to(torch.float32)
        loss = criterion(yhat, targets)
        loss.backward()
        optimizer.step()
        runningLoss += loss.item()
        print(f"{((i+1) / len(trainDl))*100} / 100 of epoch")
      losses.append(runningLoss)
    
      print("----------")
      print(f"LOSS: {runningLoss/len(trainDl) }")
      print(f"{epoch+1} / {EPOCHS} epochs")
      print("-----------")
    print("completed training :)")

def occupancy_grid_callback(grid_msg):
    global img
    global model
    img = []

    cv2.namedWindow("ROI")

    width = grid_msg.info.width
    height = grid_msg.info.height
    data = grid_msg.data
    for cell in data:
        if(cell>=10):
            img.append([255,255,255])
            continue
        if (cell == -1):
            img.append([100,100,100])
            continue
        else:
            img.append([cell, cell, cell])
    img = np.array(img)
    img.resize((height,width,3)) 
    img = img.astype(np.uint8)
    img[np.all(img == (255,255,255), axis=-1)] = (255,0,0)
    img[np.all(img == (100,100,100), axis=-1)] = (0,0,255)
    img[np.all(img == (0,0,0), axis=-1)] = (0,255,0)
    
    img = cv2.resize(img, (224,224))
    img = (img - np.min(img) ) / (np.max(img)-np.min(img))

    preprocess = transforms.Compose([
        transforms.ToTensor(),  # Convert to PyTorch tensor
    ])
    
    #eval = torch.utils.data.DataLoader(img, batch_size=1)
    #print(eval[0])
    inputs = preprocess(img).unsqueeze(0)
    #inputs = inputs.view((inputs.shape[0], inputs.shape[3], inputs.shape[2], inputs.shape[1]))

    with torch.no_grad():
        inputs = inputs.to(torch.float32)
        output = model(inputs)
        output = output.numpy()[0]
        output = output*224
    cv2.circle(img, ( int(output[0]), int(output[1]) ), 5, [255,255,255], -1)

    print(( int(output[0]), int(output[1]) ))

    # with torch.no_grad():
    #     runningLoss = 0
    #     for i, (inputs) in enumerate(eval):
    #         inputs = inputs.to(torch.float32)
    #         print(inputs.shape)
    #         #inputs = inputs.view((inputs.shape[0], inputs.shape[3],inputs.shape[2],inputs.shape[1]))
    #         outputs = model(inputs)
    #         predictions.append(outputs.numpy())
    # predictions = np.array(predictions)
    # print(predictions)

    # prediction = outputs.numpy()
    # predictionX = prediction[:,:,0].flatten()*224
    # predictionY = prediction[:,:,1].flatten()*224
    # print((predictionX, predictionY))
    cv2.imshow("ROI", img)
    cv2.waitKey(1)

if __name__ == '__main__':
    rospy.init_node('roi_subscriber')
    # Create an occupancy grid subscriber
    rospy.Subscriber('roi_image', OccupancyGrid, occupancy_grid_callback)
    rospy.spin()
