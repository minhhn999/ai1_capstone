# Project summary:
- Project name: AI Application with python
- Description: project is to training flowers data from folder using Pytorch and predict probatilities top 5 of flowers
- project source code link: https://github.com/minhhn999/ai1_capstone

# How to run project:
- open terminal
- run training command:
python train.py --dir flowers --train_on_gpu --arch vgg19_bn --epochs 3
- run predict command:
python predict.py --imagefile flowers/test/1/image_06743.jpg --checkpoint checkpoint.pth --predict_on_gpu --arch vgg19_bn

# Command Line Arguments:
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