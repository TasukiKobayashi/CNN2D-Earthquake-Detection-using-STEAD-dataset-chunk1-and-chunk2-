import pandas as pd
import numpy as np
import os
import h5py
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split


# PyTorch CNN Models
class SeismicClassificationCNN(nn.Module):
    def __init__(self, input_height, input_width):
        super(SeismicClassificationCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * (input_height // 2) * (input_width // 2), 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout1(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

#File PATHS
hd5_1 = "/Users/whynot.son/Downloads/Dataset/chunk1.hdf5"
csv_file_1 = "/Users/whynot.son/Downloads/Dataset/chunk1.csv"
hdf5_2 =  "/Users/whynot.son/Downloads/Dataset/chunk2.hdf5"
csv_file_2 = "/Users/whynot.son/Downloads/Dataset/chunk2.csv"

chunk_1 = pd.read_csv(csv_file_1, low_memory=False)
chunk_2 = pd.read_csv(csv_file_2, low_memory=False)
full_csv = pd.concat([chunk_1,chunk_2])


class SeismicCNN():
    def __init__(self,model_type,target,choose_dataset_size,full_csv,dir):
        self.model_type = model_type
        self.target = target
        self.choose_dataset_size = choose_dataset_size
        self.full_csv = full_csv
        self.dir = dir
        self.traces_array = []
        self.img_dataset = []
        self.labels = []
        self.imgs = []
        self.train_images = []
        self.train_labels = []
        self.test_images = []
        self.test_labels = []
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_loss = []
        self.test_acc = []
        self.predicted_classes = []
        self.predicted_probs = []
        self.cm = []
        self.epochs = []
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.img_height = None
        self.img_width = None

        if self.model_type == 'classification':
            # create list of traces in the image datset
            print('Creating seismic trace list')
            for filename in os.listdir(dir): # loop through image directory and get filenames
                if filename.endswith('.png'):
                    self.traces_array.append(filename[0:-4]) # remove the .png from filename

            if choose_dataset_size == 'full':
                # select only the rows in the metadata dataframe which correspond to images
                print('Selecting traces matching images in directory')
                self.img_dataset = self.full_csv.loc[self.full_csv['trace_name'].isin(self.traces_array)] # select rows from the csv that have matching image files
                self.labels = self.img_dataset['trace_category'] # target variable, 'earthquake' or 'noise'
                self.labels = np.array(self.labels.map(lambda x: 1 if x == 'earthquake_local' else 0)) # transform target variable to numerical categories
                print(f'The number of traces in the directory is {len(self.img_dataset)}')
                count = 0
                for i in range(0,len(self.img_dataset['trace_name'])): # loop through images and read them into the imgs array
                    count += 0
                    #print(f'Working on trace # {count}')
                    img= cv2.imread(self.dir+'/'+self.img_dataset['trace_name'].iloc[i]+'.png',0) # read in image as grayscale image
                    self.imgs.append(img)
                self.imgs = np.array(self.imgs)
                # Set image dimensions for CNN model
                if len(self.imgs) > 0:
                    self.img_height, self.img_width = self.imgs[0].shape
                else:
                    self.img_height, self.img_width = 224, 224  # default dimensions
                self.imgs.shape
                
            elif isinstance(choose_dataset_size, int):
                seismic_dataset = self.full_csv.loc[self.full_csv['trace_name'].isin(self.traces_array)] # get rows of csv dataset that have corresponding images in directory
                choose_seismic_dataset = np.random.choice(np.array(seismic_dataset['trace_name']),choose_dataset_size,replace=False)
                self.img_dataset = seismic_dataset.loc[seismic_dataset['trace_name'].isin(choose_seismic_dataset)] # random choice of images from the directory
                self.labels = self.img_dataset['trace_category'] # target variable, 'earthquake' or 'noise'
                self.labels = np.array(self.labels.map(lambda x: 1 if x == 'earthquake_local' else 0)) # transform target variable to numerical categories
                print(f'The number of traces in the directory is {len(self.img_dataset)}')
                count = 0
                for i in range(0,len(self.img_dataset['trace_name'])): # loop through trace names in filtered dataframe and append images to imgs array
                    count += 1
                    #print(f'Working on trace # {count}')
                    img= cv2.imread(self.dir+'/'+self.img_dataset['trace_name'].iloc[i]+'.png',0) # read in image as grayscale image
                    self.imgs.append(img)
                self.imgs = np.array(self.imgs)
                # Set image dimensions for CNN model
                if len(self.imgs) > 0:
                    self.img_height, self.img_width = self.imgs[0].shape
                else:
                    self.img_height, self.img_width = 224, 224  # default dimensions
                self.imgs.shape
                
            else:
                print('Error: please choose either "full" for variable choose_dataset_size to use the full dataset, or provide an integer number of random samples to take from the dataset')

    def train_test_split(self, test_size=0.25, random_state=42):
        self.train_images, self.test_images, self.train_labels, self.test_labels = train_test_split(
            self.imgs, self.labels, test_size=test_size, random_state=random_state
        )

        # Convert numpy array (if labels are still Series)
        if hasattr(self.train_labels, "to_numpy"):
            self.train_labels = self.train_labels.to_numpy()
        if hasattr(self.test_labels, "to_numpy"):
            self.test_labels = self.test_labels.to_numpy()

        # Convert labels to tensors (classification only)
        self.train_labels = torch.LongTensor(self.train_labels)
        self.test_labels = torch.LongTensor(self.test_labels)

        # Convert images sang Tensor (luôn float32) và thêm channel dimension
        self.train_images = torch.FloatTensor(self.train_images).unsqueeze(1)  # Add channel dimension
        self.test_images = torch.FloatTensor(self.test_images).unsqueeze(1)    # Add channel dimension

    def train(self, epochs=10, batch_size=64):
        self.epochs = epochs
        train_dataset = TensorDataset(self.train_images, self.train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = TensorDataset(self.test_images, self.test_labels)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        self.model.train()
        for epoch in range(epochs):
            train_loss = 0.0
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)

                # classification: targets are integer class indices

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output.squeeze(), target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            val_loss = 0.0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    # classification: targets are integer class indices
                    output = self.model(data)
                    loss = self.criterion(output.squeeze(), target)
                    val_loss += loss.item()

            self.history['train_loss'].append(train_loss/len(train_loader))
            self.history['val_loss'].append(val_loss/len(val_loader))
            print(f"Epoch {epoch+1}/{epochs} - Train loss: {train_loss/len(train_loader):.4f} - Val loss: {val_loss/len(val_loader):.4f}")



    def classification_cnn(self,epochs):
        self.epochs = epochs
        
        # Create model
        print('Building CNN model')
        self.model = SeismicClassificationCNN(self.img_height, self.img_width).to(self.device)
        
        # Define loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-6)
        
        # Create data loaders
        train_dataset = TensorDataset(self.train_images, self.train_labels)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # Validation split
        val_size = int(0.2 * len(self.train_images))
        train_size = len(self.train_images) - val_size
        train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
        
        # Training loop
        print('Starting training...')
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            # Store history
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            self.history['train_loss'].append(train_loss / len(train_loader))
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss / len(val_loader))
            self.history['val_acc'].append(val_acc)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save model
        save_dir = os.path.join(os.getcwd(), 'models', 'classification')
        os.makedirs(save_dir, exist_ok=True)
        saved_model_path = os.path.join(save_dir, 'classification.pth')
        torch.save(self.model.state_dict(), saved_model_path)
        print(f'Model saved to {saved_model_path}')

    def evaluate_classification_model(self):
        print('Evaluating model on test dataset')
        
        # Create test data loader
        test_dataset = TensorDataset(self.test_images, self.test_labels)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Evaluate model
        self.model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                test_total += target.size(0)
                test_correct += (predicted == target).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(output.cpu().numpy())
        
        self.test_acc = 100. * test_correct / test_total
        self.test_loss = test_loss / len(test_loader)
        
        print(f"\nTest data, accuracy: {self.test_acc:.2f}%")
        print(f"Test data, loss: {self.test_loss:.4f}")

        print('Finding predicted classes and probabilities to build confusion matrix')
        self.predicted_classes = np.array(all_predictions)
        self.predicted_probs = np.array(all_probabilities)

        # create confusion matrix
        print('Building confusion matrix')
        self.cm = confusion_matrix(self.test_labels.cpu().numpy(), self.predicted_classes)
        print(self.cm)
        accuracy = accuracy_score(self.test_labels.cpu().numpy(), self.predicted_classes)
        precision = precision_score(self.test_labels.cpu().numpy(), self.predicted_classes)
        recall = recall_score(self.test_labels.cpu().numpy(), self.predicted_classes)
        print(f'The accuracy of the model is {accuracy}, the precision is {precision}, and the recall is {recall}.')

        # plot confusion matrix
        plt.style.use('default')
        disp = ConfusionMatrixDisplay(confusion_matrix=self.cm, display_labels=['not earthquake','earthquake'])
        disp.plot(cmap='Blues', values_format='')
        plt.title(f'Classification CNN Results ({self.epochs} epochs)')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()
        
        # plot accuracy history
        plt.style.use('ggplot')
        fig, ax = plt.subplots(figsize=(7,7))
        ax.plot(self.history['train_acc'])
        ax.plot(self.history['val_acc'])
        ax.set_title('Model Accuracy')
        ax.set_ylabel('Accuracy')
        ax.set_xlabel('Epoch')
        ax.legend(['train','test'])
        plt.savefig('model_accuracy.png')
        plt.show()
        
        fig, ax = plt.subplots(figsize=(7,7))
        ax.plot(self.history['train_loss'])
        ax.plot(self.history['val_loss'])
        ax.set_title('Model Loss')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.legend(['train','test'])
        plt.savefig('model_loss.png')
        plt.show()
        


if __name__ == '__main__':
    dir = '/Users/whynot.son/Downloads/Test/image'  #image path
    
    model_cnn_c1 = SeismicCNN('classification','trace_category',12000,full_csv,dir) # initialize the class
    model_cnn_c1.train_test_split(test_size=0.25,random_state=42) # train_test_split
    model_cnn_c1.classification_cnn(80) # use the classification cnn method with 80 epochs
    model_cnn_c1.evaluate_classification_model() # evaluate the model
