import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models

from utils import load_datasets, get_dataloaders, get_input_args

def train_data(model, dataloaders, criterion, optimizer,device, epochs = 5):
    '''train data from input folder'''
    # training model
    steps = 0
    running_loss = 0
    print_every = 15
    for epoch in range(epochs):
        for inputs, labels in dataloaders['train']:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in dataloaders['validation']:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"validation loss: {valid_loss/len(dataloaders['validation']):.3f}.. "
                      f"validation accuracy: {accuracy/len(dataloaders['validation']):.3f}")
                running_loss = 0
                model.train()
    return model
def test_data(data_dir, model,testloader, criterion, device):
    '''test accuracy of model'''
    image_datasets = load_datasets(data_dir)
    dataloaders = get_dataloaders(image_datasets)
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            test_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
    print(f"test loss: {test_loss/len(testloader):.3f}.. "
        f"test accuracy: {accuracy/len(testloader):.0%}")
def main():
    '''main function when run this module'''
    # get input parameters from command line
    in_arg = get_input_args()
    data_dir = in_arg.dir
    
    #load datasets from data folder
    image_datasets = load_datasets(data_dir)
    
    # get dataloaders from datasets
    dataloaders = get_dataloaders(image_datasets)
    # Build and train your network
    # Use GPU if it's available
    if in_arg.train_on_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print('device usage:', device)
    architecture=in_arg.arch
#     model = models.vgg19_bn()
    model = eval("models.{}(pretrained=True)".format(architecture.lower()))
    model.name = architecture
    print('model', model)
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    print('hidden units', int(in_arg.hidden_units))
    classifier = nn.Sequential(nn.Linear(model.classifier[0].in_features, int(in_arg.hidden_units)),
                               nn.ReLU(),
                               nn.Dropout(0.5),
                               nn.Linear(int(in_arg.hidden_units), 102),
                               nn.LogSoftmax(dim=1))
    # Create the network, define the criterion and optimizer
    model.classifier = classifier
    print(model.classifier)
    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(in_arg.lr))
    model.to(device)
    model = train_data(model, dataloaders, criterion, optimizer,device, int(in_arg.epochs))
    
    # test the model
    test_data(data_dir, model,dataloaders['test'], criterion, device)
    
    # Save the checkpoint 
    model.class_to_idx = image_datasets['train'].class_to_idx
    checkpoint = {
        'epoch': int(in_arg.epochs),
        'output_size': 102,
        'model_state_dic': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': model.class_to_idx,
        'classifier': model.classifier
    }
    torch.save(checkpoint, in_arg.checkpoint)
    print("finished training and checkpoint saved to file %s" % in_arg.checkpoint)

# Call to main function to run the program
if __name__ == "__main__":
    main()