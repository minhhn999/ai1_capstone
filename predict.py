import torch
from utils import predict, cat_to_name_mapping, get_input_args, load_checkpoint

def main():
    '''main function when run this module'''
    # get input parameters from command line
    in_arg = get_input_args()
    # Use GPU if it's available
    if in_arg.predict_on_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print('device usage:', device)
    
    cat_to_name = cat_to_name_mapping(in_arg.cat_to_name_json_file)
    model = load_checkpoint(in_arg.checkpoint, in_arg.arch)
    model.to(device)
    print(model.classifier)
    
    # predict top 5 probabilities for image
    probs, classes = predict(in_arg.imagefile, model, device, int(in_arg.topk))
    flowers = [cat_to_name[cat] for cat in classes]
    result = zip(flowers, probs)
    print("{:30}: {:20}".format('Flowers', 'probabilities'))
    for item in result:
        print("{:30}: {:.0%}".format(*item))

# Call to main function to run the program
if __name__ == "__main__":
    main()