The implementation of baseline code. I have some trouble in training the model. You could add the dataset under the folder 'baselinemodified'. 
After downloading the 'dataset', you could copy the folder 'train' into the folder /baselinemodified/dataset. Then it will fit the path in the config.yaml. 
But once you launch the code, it will restart again and again. I didn't find why. But I am pretty sure the problem lis on 'train_dataloader' in line 31 train.py.
 Do you know how to fix it?