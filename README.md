# Dinov2 PCam Classification

This project demonstrates image classification on the PatchCamelyon (PCam) dataset using a self-supervised Dinov2 model. The goal is to classify histopathological images of lymph node sections as either containing metastatic cancer or not.

The project is implemented as a Jupyter Notebook (`pcam-dinov2-classification.ipynb`) and covers the following key steps:

## 1. Project Setup

*   **Clone Dinov2 Repository**: Clones the official Dinov2 repository from Facebook Research.
*   **Install Dependencies**: Installs necessary Python libraries including PyTorch, torchvision, timm, scikit-learn, matplotlib, opendatasets, and pandas.

## 2. Dataset Handling

*   **Download PCam Dataset**: Downloads the PatchCamelyon dataset, typically from Kaggle.
*   **Prepare Dataset**: Organizes the dataset into training and validation sets. The notebook includes logic for splitting the data and can be adapted for using a subset of the data for faster experimentation.

## 3. Exploratory Data Analysis (EDA)

*   Visualizes sample images from each class (cancer/no cancer).
*   Checks the class distribution in the dataset.
*   Calculates basic statistics about the image dimensions and modes.

## 4. Data Preprocessing

*   **Image Transformations**: Resizes images to be compatible with Dinov2 (e.g., 224x224 pixels).
*   **Normalization**: Normalizes pixel values according to ImageNet standards, which Dinov2 expects.
*   **Data Augmentation**: Applies augmentations like random horizontal flips and rotations to the training set to improve model generalization.
*   **PyTorch DataLoaders**: Creates `DataLoader` instances for efficient batching and loading of data during training and evaluation. A custom `PCamDataset` class is defined for this purpose.

## 5. Model Loading and Adaptation

*   **Load Pre-trained Dinov2**: Loads a pre-trained Dinov2 model (e.g., `dinov2_vits14`). Dinov2 models are powerful vision transformers trained using self-supervised learning.
*   **Adapt Classification Head**: Modifies the model by replacing its original classification head with a new linear layer suitable for the binary classification task (2 classes: cancer/no cancer). The backbone of the Dinov2 model is initially frozen.

## 6. Model Training

*   **Loss Function**: Uses CrossEntropyLoss, standard for classification tasks.
*   **Optimizer**: Employs the AdamW optimizer to update the weights of the new classification head.
*   **Learning Rate Scheduler**: Optionally uses a learning rate scheduler (e.g., `StepLR`) to adjust the learning rate during training.
*   **Training Loop**: Iterates through the training data for a specified number of epochs, performing forward and backward passes, and updating model weights.
*   **Validation**: Monitors validation accuracy after each epoch and saves the model weights that achieve the best validation performance.
*   **Metrics Tracking**: Records and plots training and validation loss and accuracy over epochs.

## 7. Model Evaluation

*   **Performance Metrics**: Evaluates the trained model on the validation set.
*   Calculates metrics such as accuracy, precision, recall, and F1-score.
*   **Confusion Matrix**: Visualizes a confusion matrix to understand the types of errors the model makes.
*   **Visualize Predictions**: Shows some sample images from the validation set along with their true and predicted labels.

## 8. Saving the Model

*   Saves the state dictionary of the best-performing fine-tuned model to a file (e.g., `dinov2_pcam_best_model.pth`). This allows the model to be loaded later for inference or further training.

## How to Run

1.  **Prerequisites**:
    *   Ensure you have Python installed.
    *   It is recommended to use a virtual environment.
    *   Access to a machine with a GPU (NVIDIA CUDA enabled) is highly recommended for faster training.

2.  **Clone this repository (if you've put this project on GitHub):**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

3.  **Clone Dinov2 and Install Dependencies**:
    The notebook contains commands to clone the Dinov2 repository and install dependencies. Execute the cells in section "1. Clone Dinov2 Repository and Install Dependencies".
    ```python
    # Example commands from the notebook:
    # !git clone https://github.com/facebookresearch/dinov2.git
    # !pip install torch torchvision torchaudio timm scikit-learn matplotlib opendatasets pandas
    # import sys
    # sys.path.append('./dinov2')
    ```

4.  **Download Dataset**:
    The notebook uses `opendatasets` to download the PCam dataset from Kaggle. You might need to provide your Kaggle API credentials. Alternatively, manually download the dataset and adjust the paths in the notebook accordingly (see `base_dir` variable in section "2. Download and Prepare PCam Dataset").
    The dataset is expected to be found at `/kaggle/input/histopathologic-cancer-detection` if running in a Kaggle environment, or a local path if downloaded manually.

5.  **Run the Jupyter Notebook**:
    Open and run the cells in `pcam-dinov2-classification.ipynb` sequentially.

    ```bash
    jupyter notebook pcam-dinov2-classification.ipynb
    ```
    or
    ```bash
    jupyter lab pcam-dinov2-classification.ipynb
    ```

## Expected Results

Using a pre-trained Dinov2 model, even by only fine-tuning the classification head, is expected to yield strong performance on the PCam dataset. Accuracies can vary based on the subset of data used, the number of training epochs, and specific hyperparameters, but should be significantly above random chance. The notebook aims for high accuracy in distinguishing cancerous patches.

## Potential Improvements

*   **Full Fine-tuning**: Unfreeze more layers of the Dinov2 backbone for potentially better performance, though this requires more resources and careful tuning.
*   **Larger Dinov2 Models**: Experiment with `dinov2_vitb14` or `dinov2_vitl14` for higher capacity.
*   **Hyperparameter Optimization**: Tune learning rate, batch size, optimizer settings, etc.
*   **Cross-Validation**: Implement k-fold cross-validation for more robust performance estimation.

## Acknowledgements

*   [Dinov2: Learning Robust Visual Features without Supervision](https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning-foundation-model-announcement/)
*   [PatchCamelyon (PCam) Dataset](https://www.kaggle.com/competitions/histopathologic-cancer-detection)
