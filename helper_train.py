import torch 
import numpy as np

def train(model, train_loader, valid_loader, num_epochs, criterion, optimizer, device):
    min_valid_loss = np.inf
    train_losses = []
    valid_losses = []
    for epoch in range(num_epochs):
        running_train_loss = 0.0
        model.train()
        for batch_idx, (img1, img2, targets) in enumerate(train_loader):
            img1, img2, targets = img1.to(device), img2.to(device), targets.to(device)
            outputs = model(img1, img2)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
        running_train_loss = running_train_loss / len(train_loader)
        train_losses.append(running_train_loss)

        running_valid_loss = 0.0
        with torch.no_grad():
            model.eval()
            for batch_idx, (img1, img2, targets) in enumerate(valid_loader):
                img1, img2, targets = img1.to(device), img2.to(device), targets.to(device)
                outputs = model(img1, img2)
                loss = criterion(outputs, targets)
                running_valid_loss += loss.item()
        running_valid_loss = running_valid_loss / len(valid_loader)
        valid_losses.append(running_valid_loss)

        print("Epochs: {} \tTraining loss: {:.6f} \tValidation loss: {:.6f}".format(epoch+1, running_train_loss, running_valid_loss))

        if running_valid_loss < min_valid_loss:
          print("Validation loss decreased ({:.6f} --> {:.6f}). Saving model ... ".format(
              min_valid_loss, running_valid_loss
          ))
          torch.save(model.state_dict(), 'model.pt')
          min_valid_loss = running_valid_loss

    print("Finished training!")
    return train_losses, valid_losses