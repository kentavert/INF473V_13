### VM
The size NC6v3 of VM has Nvidia V100 which is quite enough for our training. Takes about 6min per epoch. You can access the VM that I've created via SSH with password. Or you can create your own VM under your account that will be more flexible. You can shut down the VM when you don't use it without telling me to stop it and without asking me everytime the ip of the VM.(Because after each time you start the VM, ip changes)

### Pseudo-labelling
implemented under guidance of https://paperswithcode.com/task/semi-supervised-image-classification. Didn't try that many different hyperparameters. Not sure it is coded correctly (didn't have a better result within 20 epochs). Maybe it should take more epochs? So I create VM in Azure which takes much shorter time than my computer to take more epochs (besides the fans of computer in casert are annoying).

The train accuracy and train_loss_epoch on WandB are fully dependent on the train_dataset. The loss is calculated on the combined dataset.

### Data augmentation
I have successfully applied the augmentation, though the accuracy may not be improved significantly, while the loss and the training accuracy do not increase that high than before. Therefore, I think it prevents the overfitting here, which is kind of an improvement. 

Also i have add a folder copied directly from https://paperswithcode.com/paper/autoaugment-learning-augmentation-strategies. But i didn't make it work for the moment.