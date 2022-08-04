import torch
import numpy as np
import matplotlib.pyplot as plt

def show_examples(train_loader):
    for img1, img2, label in train_loader:
        if label[0] == 1.0:
          plt.subplot(1,2,1)
          plt.imshow(img1[0][0])
          plt.subplot(1,2,2)
          plt.imshow(img2[0][0])
          break
    plt.show()

# plot trainining and validation loss for each epoch
def plot_results(train_losses, valid_losses, num_epochs):
    epochs = range(1, num_epochs+1)
    
    plt.plot(epochs, train_losses, 'bo', label="Training loss")
    plt.plot(epochs, valid_losses, 'b', label="Validation loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Training and validation loss')
    plt.legend(loc='upper right')
    plt.show()
