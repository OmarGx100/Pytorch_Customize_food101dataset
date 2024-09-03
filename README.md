---
# Pizza, Steak, Sushi Classification with TinyVGG

This project is a deep learning-based image classification task using a TinyVGG model to classify images into three categories: Pizza, Steak, and Sushi. The model is trained using PyTorch, and the dataset follows the `ImageFolder` format.

## Project Structure

```plaintext
├── Data/
│   └── pizza_steak_sushi/
│       ├── train/
│       │   ├── pizza/
│       │   │   ├── image1.jpg
│       │   │   └── ...
│       │   ├── steak/
│       │   │   ├── image1.jpg
│       │   │   └── ...
│       │   └── sushi/
│       │       ├── image1.jpg
│       │       └── ...
│       └── test/
│           ├── pizza/
│           │   ├── image1.jpg
│           │   └── ...
│           ├── steak/
│           │   ├── image1.jpg
│           │   └── ...
│           └── sushi/
│               ├── image1.jpg
│               └── ...
├── Models/
│   └── best_model.pth
├── engin.py
├── utils.py
├── model_builder.py
├── data_setup.py
├── train.py
└── predict.py
```

## Dataset

The dataset is structured into training and testing sets, with three classes: `pizza`, `steak`, and `sushi`. The data is organized in folders according to the `torchvision.datasets.ImageFolder` format.

## Getting Started

### 1. Prepare the Data

Ensure your dataset is structured as described above and placed in the `Data/pizza_steak_sushi/` directory.

### 2. Train the Model

To train the TinyVGG model, run the following command:

```bash
python train.py
```

This will:
- Load the dataset using `torchvision.datasets.ImageFolder`.
- Create data loaders for training and testing.
- Build the TinyVGG model using `model_builder.py`.
- Train the model and save the best-performing model to the `Models/` directory.

### 3. Predict with a Pretrained Model

To predict the class of a new image using a pretrained model, run:

```bash
python predict.py --model_path Models/best_model.pth --image_path path_to_image.jpg
```

This script will load the pretrained model and predict the class of the provided image.

## Utilities

### Saving the Model
The `utils.py` file contains functions for saving the trained model. The model is automatically saved during training if it achieves the best performance.

### Training and Evaluation
The `engin.py` file contains functions like `train_step`, `test_step`, and `train` to handle the training and evaluation of the model.

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---
