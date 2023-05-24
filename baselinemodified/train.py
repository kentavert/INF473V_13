import torch
import wandb
import hydra
from tqdm import tqdm
import data.datamodule
from torch.utils.data import DataLoader


@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg):

    logger = wandb.init(project="challenge", name=f"{cfg.model._target_}{cfg.model.frozen} {cfg.epochs}epochs af{cfg.af} unlabel{cfg.unlabelled_total} {cfg.optim._target_}")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = hydra.utils.instantiate(cfg.model).to(device)
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    combinedataset = data.datamodule.combinedDataset(datamodule.train_dataset, datamodule.unlabelled_dataset, unlabelled_total=cfg.unlabelled_total)
    combined_loader = DataLoader(combinedataset, batch_size=cfg.dataset .batch_size, num_workers=cfg.dataset.num_workers, shuffle=True)

    #threshold function
    def unlabelweight(epoch):
        alpha = 0.0
        if epoch > cfg.T1:
            alpha = (epoch - cfg.T1)/(cfg.T2 - cfg.T1)*cfg.af
            if epoch >cfg.T2 :
                alpha = cfg.af
        return alpha

    for epoch in tqdm(range(cfg.epochs)):
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0
        for i, batch in enumerate(combined_loader):
            
            if epoch<cfg.T1 :
                break
            
            images, labels = batch
            #print(images.shape,labels.shape)
            #images = datamodule.data_augment(images)
            images_sup = datamodule.data_augment(images.to(device))
            images = datamodule.data_augment(images.to(device))
            
            labels = labels.to(device)
            preds_sup = model(images_sup)
            preds = model(images)
            nolabelsize = (labels == torch.tensor([-1]*len(labels),device=device)).sum().detach()
            consistencyloss = torch.nn.functional.mse_loss(preds, preds_sup)
            labelledloss = torch.sum(loss_fn(preds, labels))/ (len(labels)-nolabelsize+1e-10)
            
            with torch.no_grad():
                pseudolabels = preds.max(1)[1]
            unlabelledloss = torch.sum(labels.eq(-1).float() * loss_fn(preds, pseudolabels)) / (nolabelsize+1e-10)
            loss = labelledloss + unlabelweight(epoch)*unlabelledloss + consistencyloss
            #print(nolabelsize, labelledloss, loss)
            logger.log({"loss": loss.detach().cpu().numpy()})
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        for i, batch in enumerate(train_loader):
            images, labels = batch
            images = datamodule.data_augment(images.to(device))
            labels = labels.to(device)
            preds = model(images)
            
            loss = torch.nn.functional.cross_entropy(preds, labels, reduction='mean', label_smoothing=0.05)
            
            if epoch<cfg.T1 :
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.detach().cpu().numpy() #* len(images)
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

        #print(len(combinedataset.indexs))
        '''
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
                    combinedataset.resetlabel(pred.argmax().type(torch.LongTensor), i*len(images)+j)
        
        #setting pseudo labels
        threshold = 0.90
        for i, batch in enumerate(unlabel_loader):
            if epoch<3 :#determine
                break
            images, labels, idxs = batch
            images = images.to(device)
            labels = labels.to(device)
            preds = torch.nn.functional.softmax(model(images),dim=-1)
            for j in range(len(images)):
                if labels[j]==48: #attention
                    pred = preds[j]
                    if pred[pred.argmax()]>threshold:#determine
                        combinedataset.adddata(idxs[j], pred.argmax().to('cpu').type(torch.LongTensor))
                        datamodule.unlabelled_dataset.set_label(pred.argmax().type(torch.LongTensor),idxs[j])
        if threshold>0.6:
            threshold-=0.01
        '''
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
            with torch.no_grad():
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                preds = model(images)
                loss = torch.nn.functional.cross_entropy(preds,labels,label_smoothing=0.05)
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
