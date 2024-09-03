
from utils import Predict_Custom_Images
import sys
from model_builder import BaseLineModel
import torch


if __name__ == "__main__":

   # Instintiate tinyVGG model
   model = BaseLineModel(3, 10, 3)

   # Getting the Model path and image path from commaned line arguments
   MODEL_SAVED_PATH = sys.argv[1]
   IMAGE_PATH = sys.argv[2]
    
   # Loading model weights from the model path
   model.load_state_dict(torch.load(MODEL_SAVED_PATH))

    # Defining Model output classes
   class_names = ['pizza', 'steak', 'sushi']
    
   Predict_Custom_Images(model,
                        IMAGE_PATH,
                        class_names= class_names)
