
import pathlib
import torch
from typing import List
import torchvision
from torchvision import transforms


def save_model_to_dir(model_name: str,
                      model_dir : str,
                      model_state_dict : torch.nn.Module.state_dict):
    model_path = pathlib.Path(model_dir)

    if not model_path.is_dir():
        print(f"Creating Directory with Name : {model_path}")
        model_path.mkdir(parents= True, exist_ok=True)
    
    MODEL_SAVE_PATH = model_path / model_name
    
    print(f"saving model to :{model_path}/{model_name}")

    torch.save(model_state_dict, MODEL_SAVE_PATH)
    
    print("Saving Finished")


# Pipeline for infernce Mode "Predicting on custome images"

# setting device agnostic code
device = 'cuda' if torch.cuda.is_available() else "cpu"

def Predict_Custom_Images(model:torch.nn.Module,
                          image_path : str,
                          
                          class_names : List[str],
                          device : torch.device = None):
    
    """Function taking image path as a string and a model and returns model_prediction and confedence level"""
    transform = transforms.Compose([
        transforms.Resize(size = (64, 64), antialias=True),
        # transforms.ToTensor()
    ])
    
    # reading an image using its path
    img = torchvision.io.read_image(image_path).type(torch.float32) / 255
    # Transforming our image into the right requirnments for our model
    img = transform(img)
 
    # try :
    #     model.to(device)
    # except :
    #     print(f"you are not specifing and device and you are now working on the cpu")
    #     model.to('cpu')
 
 
    # Making our model into the inference_mode so our grades does not change
    model.eval()
    with torch.inference_mode():
        # Calculating the raw logits 
        raw_logits = model(img.unsqueeze(0))

    # Converting the raw logits into prediction probabilities and then into prediction labels
    prediction_probability = torch.softmax(raw_logits, dim = 1).max().item()

    prediction_label = torch.softmax(raw_logits, dim = 1).argmax(dim = 1).item()


    print(f"prediction label : {class_names[prediction_label]}\nConfidence : {(prediction_probability * 100):.2f}%")