### VM
I recommend you to use the pc in info salle. But they are probably limited by storage, you can put half of the unlabelled dataset in it.

### FixMatch (most recent work)
I have newly implement https://github.com/google-research/fixmatch/tree/master or you can see https://paperswithcode.com/paper/fixmatch-simplifying-semi-supervised-learning. 
Also very useful: https://amitness.com/2020/03/fixmatch-semi-supervised/ https://amitness.com/2020/07/semi-supervised-learning/.

CutOut augmentation: https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py 
CTaugment: https://github.com/google-research/remixmatch/blob/master/libml/ctaugment.py
randaugment: https://github.com/ildoonet/pytorch-randaugment

### Pseudo-labelling
implemented under guidance of https://paperswithcode.com/task/semi-supervised-image-classification and https://github.com/iBelieveCJM/pseudo_label-pytorch. Didn't try that many different hyperparameters. Not sure it is coded correctly (didn't have a better result within 20 epochs). Maybe it should take more epochs? So I create VM in Azure which takes much shorter time than my computer to take more epochs (besides the fans of computer in casert are annoying).

The train accuracy and train_loss_epoch on WandB are fully dependent on the train_dataset. The loss is calculated on the combined dataset.

### Data augmentation
I have successfully applied the augmentation, though the accuracy may not be improved significantly, while the loss and the training accuracy do not increase that high than before. Therefore, I think it prevents the overfitting here, which is kind of an improvement. 

Also i have add a folder copied directly from https://paperswithcode.com/paper/autoaugment-learning-augmentation-strategies. But i didn't make it work for the moment.