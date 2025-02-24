import torch
import wandb
import hydra
from tqdm import tqdm



@hydra.main(config_path="configs", config_name="config", version_base=None)
def train(cfg):

    logger = wandb.init(project="challenge", name="new run 48 with DA 40 epochs")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = hydra.utils.instantiate(cfg.model).to(device)
    optimizer = hydra.utils.instantiate(cfg.optim, params=model.parameters())
    loss_fn = hydra.utils.instantiate(cfg.loss_fn)
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    for epoch in tqdm(range(cfg.epochs)):
        epoch_loss = 0
        epoch_num_correct = 0
        num_samples = 0
        for i, batch in enumerate(train_loader):
            images, labels = batch
            images = datamodule.data_augment(images)
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
