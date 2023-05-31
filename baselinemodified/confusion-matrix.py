from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import torch
import hydra
import matplotlib.pyplot as plt

@hydra.main(config_path="configs", config_name="config", version_base=None)
def confusion_matrix(cfg):

    y_pred = []
    y_label = []

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    data = Datamodule.val_dataloader()
    model = hydra.utils.instantiate(cfg.model).to(device)

    images = []
    labels = []
    for batch in data:
        data_images, data_labels = batch
        data_images = data_images.to(device)
        data_labels = data_labels.to(device)
        images.append(data_images)
        labels.append(data_labels)

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)

    predictions = model(images)

    cm = confusion_matrix(labels, predictions)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)


    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()
    plt.savefig('confusion_matrix.png')


if __name__ == "__main__":
    confusion_matrix()