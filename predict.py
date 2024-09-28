import cv2
import os
import torch
import random
import matplotlib.pyplot as plt
from PIL import Image
from utils.model import ResNet34
from torchvision import transforms

classify = {'fire': 0, 'smoke': 1, 'nan': 2, }

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()])

cap = cv2.VideoCapture(0)


net = ResNet34(6)
net.load_state_dict(torch.load('model_weights/ResNet34.pth'))
net.eval()

while True:
    # 读取摄像头的一帧图像
    ret, frame = cap.read()

    if not ret:
        print("无法读取帧，请检查摄像头连接是否正常")
        break

    # 预处理图像
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_tensor = transform(img_pil).unsqueeze(0)  # 添加batch维度

    # 进行预测
    with torch.no_grad():  # 不需要计算梯度
        pred = net(img_tensor)
        predicted_class = classify[torch.argmax(pred, dim=1).item()]

    # 显示预测结果
    cv2.putText(frame, predicted_class, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示图像
    cv2.imshow('Camera Stream Classification', frame)




