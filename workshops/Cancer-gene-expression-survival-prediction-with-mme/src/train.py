from _model import SurvivalModel

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

import logging
import argparse
import pandas as pd
import os
import genome_groups as gg
import json

def read_data_from_file(args):
    
    
    print("Training data location [{}]".format(args.train_data))
    train_data = pd.read_csv(args.train_data + "/train_data.csv")

    print("Validation data location [{}]".format(args.val_data))   
    validation_data = pd.read_csv(args.val_data + "/validation_data.csv")
    
    return train_data, validation_data


def read_data(args):
        
    train_data, validation_data = read_data_from_file(args)
    
    group = gg.GENOME_GROUPS[args.genome_group].copy()        
    group.append("SurvivalStatus")
    # Check group is invalid
    train_data = train_data[group]
    validation_data = validation_data[group]
            
    xtrain_tensor = torch.from_numpy(train_data.drop(columns=["SurvivalStatus"]).to_numpy()).float()
    ytrain_tensor = torch.from_numpy(train_data["SurvivalStatus"].to_numpy()).float()
    ytrain_tensor = ytrain_tensor.unsqueeze(1)
    
    xtest_tensor =  torch.from_numpy(validation_data.drop(columns=["SurvivalStatus"]).to_numpy()).float()
    ytest_tensor =  torch.from_numpy(validation_data["SurvivalStatus"].to_numpy()).float()
    ytest_tensor = ytest_tensor.unsqueeze(1)

    train_ds = TensorDataset(xtrain_tensor, ytrain_tensor)
    test_ds = TensorDataset(xtest_tensor, ytest_tensor)
    
    train_ds, test_ds
     
    return train_ds, test_ds

def save_model(args, model):
    
    model_dir = args.model_dir
    path = os.path.join(model_dir, "model.pth")
    print("Saving the model on location [{}]".format(path))
    torch.save(model.cpu().state_dict(), path)
    
    with open(os.path.join(model_dir, "meta.json"), "w") as f:
        f.write(json.dumps(model.serialize_params()))
        
    print("Model Artifacts Saved.")

def run_training_epoch(args, epoch, model, train_loader, loss_func, optimizer, device):
    model.train()
    
    for batch_id, batch_data in enumerate(train_loader):
        xb, yb = batch_data
        xb, yb = xb.to(device), yb.to(device)
        
        y_pred = model(xb) # Forward Propagation
        loss = loss_func(y_pred, yb) # Loss Computation
        optimizer.zero_grad() # Clearing all previous gradients, setting to zero 
        loss.backward() # Back Propagation
        optimizer.step() # Updating the parameters 
        
        print('Train epoch [{:d}] of [{:d}], batch {:d}/{:d}, loss [{:.4f}]'.format(
                epoch + 1, args.epochs, batch_id + 1, len(train_loader), loss.item()))
        

def run_an_eval_epoch(args, epoch, model, val_loader, loss_function, device):
      model.eval()
      
      with torch.no_grad():
        for batch_id, batch_data in enumerate(val_loader):
            xb, yb = batch_data
            xb, yb = xb.to(device), yb.to(device)
            
            y_pred = model(xb)
            loss = loss_function(y_pred, yb)
            
            print('Validation epoch [{:d}] of [{:d}], batch {:d}/{:d}, loss [{:.4f}]'.format(
                epoch + 1, args.epochs, batch_id + 1, len(val_loader), loss.item()))
        

def train(args):
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    print("Device : [{}]".format(device))
    
    print("Loading the model\n")
    
    batch_size = args.batch_size
    train_ds, test_ds = read_data(args)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    n_input_dim = len(gg.GENOME_GROUPS[args.genome_group])
    
    print(type(train_ds.tensors[0]))
    
    model = SurvivalModel(n_input_dim=n_input_dim).to(device)
    print(model)
    
    loss_func = nn.BCELoss()
    
    learning_rate = args.learning_rate
    
    #Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    train_loss = []
    
    for epoch in range(args.epochs):
        
        # Within each epoch run the subsets of data = batch sizes.
        run_training_epoch(args, epoch, model, train_loader, loss_func, optimizer, device)
        run_an_eval_epoch(args, epoch, model, test_loader, loss_func, device)
                 
    save_model(args, model)
                 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # hyper parameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--genome-group", type=str, default="ALL")
        
    # Data arguments
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--train-data", type=str, default=os.environ["SM_CHANNEL_TRAIN_DATA"])
    parser.add_argument("--val-data", type=str, default=os.environ["SM_CHANNEL_VAL_DATA"])

    args = parser.parse_args()
    print(args)
    
    train(args)
    
    
                 
  
    
    