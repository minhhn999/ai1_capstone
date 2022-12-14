import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import json
import argparse

def load_datasets(data_dir_path):
    '''load datasets from a directory'''
    data_dir = data_dir_path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    validation_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    image_datasets = {
        'train': train_data,
        'validation': validation_data,
        'test': test_data
    }
    return image_datasets

def get_dataloaders(image_datasets):
    ''' get dataloaders from datasets'''
    trainloader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=128, shuffle=True)
    validationloader = torch.utils.data.DataLoader(image_datasets['validation'], batch_size=64)
    testloader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=64)
    dataloaders = {
        'train': trainloader,
        'validation': validationloader,
        'test': testloader
    }
    return dataloaders

def cat_to_name_mapping(json_file_path):
    '''load data mapping categories to names from json file path'''
    with open(json_file_path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def get_input_args():
    """
    Retrieves and parses the 3 command line arguments provided by the user when
    they run the program from a terminal window. This function uses Python's 
    argparse module to created and defined these command line arguments. If 
    the user fails to provide some or all of the arguments, then the default 
    values are used for the missing arguments. 
    Command Line Arguments:
      1. Image Folder as --dir with default value 'flowers'
      2. checkpoint path as --checkpoint with default value 'checkpoint.pth'
      3. image file path as --imagefile with default value 'flowers/test/1/image_06743.jpg'
      4. architecture name as --arch with default value 'vgg19_bn'
      5. learning rate name as --lr with default value 0.003
      6. number of hidden unit name as --hidden_units with default value 4096
      7. training epochs name as --epochs with default value 5
      8. training on GPU name as --train_on_gpu with default value False
      9. top K classes along with associated probabilities name as --topk with default value 5
      10. JSON file that maps the class values to other category names name as --cat_to_name_json_file with default value 'cat_to_name.json'
      11. use the GPU to calculate the predictions name as --predict_on_gpu with default value False
      
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object  
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument("-d", "--dir", default = 'flowers')
    parser.add_argument("--checkpoint", default = 'checkpoint.pth')
    parser.add_argument("-f", "--imagefile", default = 'flowers/test/1/image_06743.jpg')
    parser.add_argument("--arch", default = 'vgg19_bn')
    parser.add_argument("--lr", default = 0.003)
    parser.add_argument("--hidden_units", default = 4096)
    parser.add_argument("--epochs", default = 5)
    parser.add_argument("--train_on_gpu", action='store_true', help='default value is False')
    parser.add_argument("--topk", default=5)
    parser.add_argument("--cat_to_name_json_file", default='cat_to_name.json')
    parser.add_argument("--predict_on_gpu", action='store_true', help='default value is False')
    
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    args = parser.parse_args()
    print('args',args)
    return args

def load_checkpoint(file, architecture):
    '''
    load trained model from checkpoint file.
    '''
    checkpoint = torch.load(file)
    model = eval("models.{}(pretrained=True)".format(architecture.lower()))
    model.name = architecture
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dic'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    width, height = im.size
    ratio=width/height
    if ratio >1:
        im.thumbnail((ratio*256,256))
    else:
        im.thumbnail((256,(1/ratio)*256))
    resized_width, resized_height = im.size
    left = (resized_width - 224)/2
    top = (resized_height - 224)/2
    right = (resized_width + 224)/2
    bottom = (resized_height+ 224)/2
    im=im.crop((left, top, right, bottom))
    np_image=np.array(im)
    np_image = np_image / 255
    means=np.array([0.485, 0.456, 0.406])
    std= np.array([0.229, 0.224, 0.225])
    np_image=(np_image-means)/std
    np_image = np_image.transpose(2,0,1)

    return torch.from_numpy(np_image)

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model,device, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    model.eval()
    model.to(device)
    
    image = process_image(image_path)
    image = torch.unsqueeze(image, 0)
    image = image.to(device, dtype=torch.float)
    
    logps = model.forward(image)
    ps = torch.exp(logps)
    
    top_p, top_labels = ps.topk(topk, dim=1)
    
    top_p = top_p.tolist()[0]
    top_labels = top_labels.tolist()[0]
    
    class_to_idx_revert = {v: k for k, v in model.class_to_idx.items()}
    top_labels = [class_to_idx_revert[label] for label in top_labels]
    
    return top_p, top_labels

