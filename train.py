from Model import VisionTransformer
from Dataloader import ImageDataset, collate_fn
import os
import argparse
import wandb
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def get_metrics(y_true, y_pred):
    # breakpoint()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted') # Or 'macro', 'micro'
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", report)
    wandb.log({"accuracy": accuracy, "precision": precision, "recall": recall, "f1-score": f1})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model for training a very simple VIT")
    parser.add_argument("--project_name",type=str, required=True,  help="Name of project(this will also be the saving directory")
    parser.add_argument("--data_path", type = str, required=True, help= "Path where data is present(for structure to use, check image)")
    parser.add_argument("--optimizer",type=str, required=True,  help="Which optimizer to use")
    parser.add_argument("--learning_rate",type=float, required=True,  help="The learning rate to use")
    parser.add_argument("--epochs", type=int, required=True, help="Total number of epochs")
    parser.add_argument("--image_shape", nargs= '+', type=int, required=True, help="Image shape")
    parser.add_argument("--patch_shape", nargs='+', type=int, required=True, help="Patch shape")
    parser.add_argument("--channels", type=int, required=True, help="Num of channels")
    parser.add_argument("--save_freq", type=int, default=10, help="Save frequency of model")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--eval_freq", type=int, default = 10, help="Evaluation frequency")
    parser.add_argument("--device", type=str, default='cuda', help="Get which device is being used")
    args = parser.parse_args()
    wandb.login()
    run = wandb.init(
            project=args.project_name,
            config={
                'data_path': args.data_path,
                'optimizer': args.optimizer,
                "lr": args.learning_rate,
                "epochs": args.epochs,
                "image_shape": args.image_shape,
                "patch_shape": args.patch_shape,
                "channels": args.channels,
                "save_freq": args.save_freq,
                "batch_size": args.batch_size,
                "eval_freq": args.eval_freq
                })
    device = args.device
    epochs = args.epochs
    batch_size = args.batch_size
    train_dataset = ImageDataset(data_path = args.data_path, shape = args.image_shape, split='train')
    test_dataset = ImageDataset(data_path = args.data_path, shape = args.image_shape, split="test")
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True, collate_fn = collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, drop_last = True, collate_fn = collate_fn )
    criterion = nn.CrossEntropyLoss()
    model = VisionTransformer(num_classes = len(train_dataset.classes), channels = args.channels, patch_shape = args.patch_shape, image_shape = args.image_shape).to(device)

    
    if(args.optimizer.lower() == "adamw"):
        optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,            
        )
    else:
        raise ValueError(f"Not implemented {args.optimizer} yet")
    Min_loss = 1e9
    loss_list = []
    BASE_PATH = 'runs/' + args.project_name
    os.makedirs(BASE_PATH, exist_ok=True)

    for epoch in tqdm(range(epochs)):
        losses = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            logits = outputs.permute(0,2,1)
            # breakpoint()
            loss = criterion(logits, targets)
            losses += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_list += [losses / batch_size]
        wandb.log({"loss":losses})
        print(sum(loss_list)/len(loss_list))
        if(epoch%args.save_freq == 0):
            torch.save(model.state_dict(), f'{BASE_PATH}/model_epoch_{epoch}.pt')
        if(losses<Min_loss):
            torch.save(model.state_dict(), f'{BASE_PATH}/best_model.pt')
            Min_loss = losses
        if(epoch%args.eval_freq==0):
            model.eval()
            y_true = []
            y_pred = []
            with torch.no_grad():
                for inputs, targets in test_dataloader:
                    # breakpoint()
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    # breakpoint()
                    _, predicted_labels = torch.max(outputs, 2)
                    y_true.extend(targets.cpu().numpy())
                    y_pred.extend(predicted_labels.cpu().numpy())
                    get_metrics(y_true, y_pred)
            model.train()


    run.finish()
