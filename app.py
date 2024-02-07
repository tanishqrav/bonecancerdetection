import streamlit as st
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.avg = nn.AvgPool2d(8)
        self.fc = nn.Linear(512 * 1 * 1, 2)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avg(x)
        x = x.view(-1, 512 * 1 * 1)
        x = self.fc(x)
        return x

Transformations = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(64, padding_mode='reflect'),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

path_to_model = 'model/model.ckpt'

def preprocess(data):
    data = [data]
    dataset_inp = Dataset(data_df=data, transform=Transformations)
    load_inp = DataLoader(dataset=dataset_inp, batch_size=1)
    return load_inp

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_df, transform=None):
        super().__init__()
        self.df = data_df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img = self.df[index]
        image = np.array(Image.open(img))
        if self.transform is not None:
            image = self.transform(image)
        return image

st.title("Bone Cancer Detection")
st.markdown("For Bone Cancer from  images using a Convolutional Neural Network, implemented with PyTorch!")

wav = st.file_uploader("Upload your Image file (TIF)")
if wav is not None:
    st.image(Image.open(wav), width=300)
    wav = preprocess(wav)
    model = CNN()
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
    model.to(torch.device('cpu'))
    model.eval()

    # Predictions list to collect all predictions
    predictions = []

    with torch.no_grad():
        for img in wav:
            img = img.to(torch.device('cpu'))
            output = model(img)
            _, predicted = torch.max(output.data, 1)
            predictions.append(predicted.cpu().numpy()[0])  # Convert tensor to numpy array and extract item

    # Count the number of positive predictions
    num_positive_predictions = sum(pred == 1 for pred in predictions)

    # Decide the final prediction based on majority vote
    final_prediction = 1 if num_positive_predictions > len(predictions)//1.5 else 0

    st.write('The model predicts', 'that the sample is Bone Cancer positive' if final_prediction == 1 else 'the sample doesn\'t have Bone Cancer')
