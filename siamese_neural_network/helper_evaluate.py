import torch 

def test(model, test_loader, device):
  correct = 0
  count = 0
  model.eval()
  with torch.no_grad():
    for batch_idx, (main_img, test_set, target) in enumerate(test_loader):
      main_img = main_img.to(device)
      target = target.to(device)
      pred_val = 0
      pred = -1
      for i, test_img in enumerate(test_set):
        test_img = test_img.to(device)
        output = model(main_img, test_img)
        if output > pred_val:
          pred_val, pred = output, i
      if pred == target:
        correct += 1
      count += 1
      if count % 20 == 0:
        print("Accuracy on n way: {}".format(correct/count))