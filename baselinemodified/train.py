import torch
import wandb
import hydra
from tqdm import tqdm
import data.datamodule
from torch.utils.data import DataLoader


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg):

    logger = wandb.init(project="challenge", name="trypseudolabel")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = hydra.utils.instantiate(cfg.model).to(device)
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    unlabel_loader = datamodule.unlabelled_dataloader()

    #newly created with traindata inside
    combinedataset = data.datamodule.combinedDataset(train_loader)
    combined_loader = DataLoader(combinedataset, batch_size=128, num_workers=8)

    for epoch in tqdm(range(cfg.epochs)):
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0
        for i, batch in enumerate(combined_loader):
            images, labels = batch
            #print(images.shape,labels.shape)
            #images = datamodule.data_augment(images)
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            loss = loss_fn(preds, labels)
            logger.log({"loss": loss.detach().cpu().numpy()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().cpu().numpy() * len(images)
            epoch_num_correct += (
                (preds.argmax(1) == labels).sum().detach().cpu().numpy()
            )
            num_samples += len(images)

        epoch_loss /= num_samples
        epoch_acc = epoch_num_correct / num_samples
        logger.log(
            {
                "epoch": epoch,
                "train_loss_epoch": epoch_loss,
                "train_acc": epoch_acc,
            }
        )
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0

        #reseting pseudo labels
        for i,batch in enumerate(combined_loader):
            images, labels = batch
            if i < 720//len(images) + 1 :
                continue
            images = images.to(device)
            labels = labels.to(device)
            preds = torch.nn.functional.softmax(model(images),dim=-1)
            for j in range(len(images)):
                pred = preds[j]
                if pred[pred.argmax()]>0.9 and pred.argmax()!=labels[j]:
                    combinedataset.resetlabel(pred.argmax(), i*len(images)+j)


        #setting pseudo labels
        for i, batch in enumerate(unlabel_loader):

            images, labels, idxs = batch
            images = images.to(device)
            labels = labels.to(device)
            preds = torch.nn.functional.softmax(model(images),dim=-1)
            for j in range(len(images)):
                if labels[j]==48: #attention
                    pred = preds[j]
                    if pred[pred.argmax()]>0.8:
                        combinedataset.adddata(images[j], pred.argmax())
                        labels[j] = pred.argmax()

        '''
        #train on the pseudo labels
        for i, batch in enumerate(unlabel_loader):
            if epoch<10 :
                break
            images, labels, idxs = batch
            #images = datamodule.data_augment(images)
            extracted_images = []
            extracted_labels = []

            for j in range(len(images)):
                if labels[j]!=48:
                    extracted_images.append(images[j].tolist())
                    extracted_labels.append(labels[j])
            extracted_images = torch.tensor(extracted_images).to(device)
            extracted_labels = torch.tensor(extracted_labels).type(torch.LongTensor).to(device)
            if(len(extracted_labels)==0):
                continue
            preds = model(extracted_images)
            loss = loss_fn(preds, extracted_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            '''

        for i, batch in enumerate(val_loader):
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            loss = loss_fn(preds, labels)
            epoch_loss += loss.detach().cpu().numpy() * len(images)
            epoch_num_correct += (
                (preds.argmax(1) == labels).sum().detach().cpu().numpy()
            )
            num_samples += len(images)
        epoch_loss /= num_samples
        epoch_acc = epoch_num_correct / num_samples
        logger.log(
            {
                "epoch": epoch,
                "val_loss_epoch": epoch_loss,
                "val_acc": epoch_acc,
            }
        )
    torch.save(model.state_dict(), cfg.checkpoint_path)
    wandb.finish()


if __name__ == "__main__":
    train()
