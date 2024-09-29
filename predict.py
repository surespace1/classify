import cv2
import os
import time
import torch
import random
import matplotlib.pyplot as plt
from PIL import Image
from utils.model import ResNet34
from torchvision import transforms

classify = {'fire': 0, 'smoke': 1, 'nan': 2, }
classify = {value: key for key, value in classify.items()}
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()])

cap = cv2.VideoCapture(0)


net = ResNet34(3)
net.load_state_dict(torch.load('./model_weights/ResNet34.pth',map_location='cpu'))
net.eval()

fire_detected_time = 0  # 记录火灾检测的时间
detection_duration = 15  # 持续检测时间

while True:
    ret, frame = cap.read()

    if not ret:
        print("无法读取帧，请检查摄像头连接是否正常")
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_tensor = transform(img_pil).unsqueeze(0)

    with torch.no_grad():
        pred = net(img_tensor)
        predicted_class = classify[torch.argmax(pred, dim=1).item()]

    cv2.putText(frame, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if predicted_class == "fire"or predicted_class == "smoke":
        fire_detected_time += 1  # 每帧增加 1 秒
    else:
        fire_detected_time = 0  # 重置计时器

    if fire_detected_time >= detection_duration:
        # 发出警报
        print("警报：检测到火灾！")
        # 这里可以添加发出声音或其他警报逻辑

    cv2.imshow('Camera Stream Classification', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


