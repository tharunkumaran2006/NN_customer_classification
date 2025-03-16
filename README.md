# EX 02 : Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

![image](https://github.com/user-attachments/assets/efbece2f-9841-424e-a7be-644448ec1714)


## DESIGN STEPS

### STEP 1:
Data Preprocessing: Clean, normalize, and split data into training, validation, and test sets.

### STEP 2:
Model Design:
 * Input Layer: Number of neurons = features.
 * Hidden Layers: 2 layers with ReLU activation.
 * Output Layer: 4 neurons (segments A, B, C, D) with softmax activation.

### STEP 3:
Model Compilation: Use categorical crossentropy loss, Adam optimizer, and track accuracy.

## STEP 4:
Training: Train with early stopping, batch size (e.g., 32), and suitable epochs.

## STEP 5:
Evaluation: Assess using accuracy, confusion matrix, precision, and recall.

## STEP 6:
Optimization: Tune hyperparameters (layers, neurons, learning rate, batch size).


## PROGRAM

### Name: THARUN V K 
### Register Number: 212223230231

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size,32)
        self.fc2 = nn.Linear(32,16)
        self.fc3 = nn.Linear(16,8)
        self.fc4 = nn.Linear(8,4)

    def forward(self,x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = F.relu(self.fc3(x))
      x = self.fc4(x)
      return x
        

```
```python
# Initialize the Model, Loss Function, and Optimizer
model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.01)

```
```python
def train_model(model, train_loader,criterion,optimizer,epochs):
  for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in train_loader:
      optimizer.zero_grad()
      output = model(X_batch)
      loss = criterion(output,y_batch)
      loss.backward()
      optimizer.step()

    if (epoch + 1) % 10 == 0:
      print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
```



## Dataset Information

![Screenshot 2025-03-10 111631](https://github.com/user-attachments/assets/17ce6ccf-d49a-4916-b88c-744ee2aaf6bf)


## OUTPUT

### Confusion Matrix

![Screenshot 2025-03-10 210908](https://github.com/user-attachments/assets/612289e7-37c7-4bc6-9af4-e3ce85f2e0b0)


### Classification Report

![Screenshot 2025-03-10 210922](https://github.com/user-attachments/assets/c3875263-f1b0-4e0d-b25a-f67eb4a64e40)


### New Sample Data Prediction

![Screenshot 2025-03-10 210933](https://github.com/user-attachments/assets/0e14c2c3-05da-4a00-9277-9363ecf8b56e)


## RESULT
So, To develop a neural network classification model for the given dataset is executed successfully.
