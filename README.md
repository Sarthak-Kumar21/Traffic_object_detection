# 🚗 Traffic Object Detection with YOLOv8
![Traffic Density Estimation](https://github.com/Sarthak-Kumar21/Traffic_object_detection/blob/main/Vehicle_Detection_Image_Dataset/cover_picture.jpg?raw=true)


## 🔍 Overview

This project focuses on <strong>Traffic Object Detection</strong>, a vital component in traffic estimation and urban planning. The goal is to identify vehicles within a specific area in each frame to assess traffic density. This shall aid in identifying peak traffic periods, congested zones, and assists in urban planning. Usine this project, we can plan for traffic management, highway tolls and urban planning.

## 🎯 Objectives

* **YOLOv8 Model Selection and Assessment:** Commencing with the selection of a pre-trained YOLOv8 model and evaluating its baseline performance on the COCO dataset for vehicle detection purposes.
* **Specialized Vehicle Dataset Curation:** Assembling and annotating a targeted dataset dedicated to vehicles to enhance the model's detection accuracy for a range of vehicle types.
* **Model Refinement for Superior Detection:** Applying transfer learning techniques to fine-tune the YOLOv8 model, with a special focus on detecting vehicles from aerial views, thus significantly improving precision and recall rates.
* **Thorough Evaluation of Model Performance:** Conducting a detailed analysis of learning curves, confusion matrices, and performance metrics to ensure the model's reliability and its capability to generalize.


## 📚 Dataset Description

### 🌐 Overview
The **Top-View Vehicle Detection Image Dataset for YOLOv8** is essential for tasks like traffic monitoring and urban planning. It provides a unique perspective on vehicle behavior and traffic patterns from aerial views, facilitating the creation of AI models that can understand and analyze traffic flow comprehensively.

### 🔍 Specifications 
- 🚗 **Class**: 'Vehicle' including cars, trucks, and buses.
- 🖼️ **Total Images**: 626
- 📏 **Image Dimensions**: 640x640 pixels
- 📂 **Format**: YOLOv8 annotation format

### 🔄 Pre-processing
Each image is carefully pre-processed and standardized to ensure consistency and high-quality training data for our model.

### 🔢 Dataset Split
The dataset is meticulously split into:
- **Training Set**: 536 images for model training with diverse scenarios.
- **Validation Set**: 90 images for unbiased model performance evaluation.

### 🚀 Significance
This dataset is pivotal in developing sophisticated vehicle detection models and shaping intelligent transportation systems for smarter city infrastructures.

### 🗃️ Sources
- The dataset is curated from [Pexels](https://www.pexels.com/search/videos/), offering diverse top-view videos for a rich vehicle detection dataset.

## 📁 File Descriptions

- **`Vehicle_Detection_Image_Dataset/`**: This directory houses the image dataset for the project and the sample image utilized within the notebook.
- **`Screenshots/`**: Contains the output and graph screenshots of relevant results utilized in the README.md file.
- **`LICENSE`**: The legal framework defining the terms under which this project's code and dataset can be used.
- **`README.md`**: The document you are reading that offers an insightful overview and essential information about the project.
- **`traffic_object_detection.ipynb`**: The Jupyter notebook that documents the model development pipeline, from data preparation to model evaluation and inference.


## Model Development Pipeline

1. **Setup and Initialization**: Loaded necessary packages and libraries from requirements.txt file.
Configured visual appearance of Seaborn plots for initial visualization.
    ```python
    !pip install ultralytics
    ```
    ```python 
    import warnings
    warnings.filterwarnings('ignore')

    import os
    import shutil
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import cv2
    import yaml
    from PIL import Image
    from ultralytics import YOLO
    from IPython.display import Video
    ```

2. **Loading YOLOv8 Pre-trained Model**: The pre-trained YOLOv8 object detection models, has been trained on the COCO dataset. The Common Objects in Context (COCO) dataset is designed for object detection, segmentation, and captioning, and encompasses 80 diverse object categories:
    ```python
    model = YOLO('yolov8n.pt')
    ```

3. **Model Selection**: The YOLOv8 suite presents five distinct models: nano, small, medium, large, and xlarge. A clear trend emerges from the data: as model size increases, there's a notable improvement in mAP, indicating enhanced accuracy. Conversely, this augmentation comes at the cost of speed, with larger models being slower. All models adhere to a standard input size of 640x640 pixels, optimizing performance across diverse applications. I have used nano model.

    ![Performance Tradeoffs](https://github.com/Sarthak-Kumar21/Traffic_object_detection/blob/main/Screenshots/YOLO%20Model%20Statistics.jpg?raw=true)


4. **Dataset Exploration**: From uploaded dataset onto github, I have used requests library and Github API URL to list out the folder contents. By initializing an empty set and iterating over the train dataset folder, I have calculated the total number of images. Additional check is put to check if all images are of the same size.
    ```python
    # GitHub repository details
    repo_owner = "Sarthak-Kumar21"
    repo_name = "Traffic_object_detection"
    folder_path = "Vehicle_Detection_Image_Dataset/valid/images"

    # GitHub API URL to list the folder contents
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{folder_path}"

    # Fetch the list of files in the folder
    response = requests.get(api_url)
    response.raise_for_status()  # Ensure the request was successful
    files = response.json()

    # Initialize counters and sets for image sizes
    num_valid_images = 0
    valid_image_sizes = set()

    # Download and check each .jpg image
    for file in files:
        if file['name'].lower().endswith('.jpg'):
            num_valid_images += 1
            image_url = file['download_url']
            image_response = requests.get(image_url)
            image_response.raise_for_status()
            with Image.open(io.BytesIO(image_response.content)) as img:
                valid_image_sizes.add(img.size)

    # Print the results
    print(f"Number of images: {num_valid_images}")
    if len(valid_image_sizes) == 1:
        print(f"All images have the same size: {valid_image_sizes.pop()}")
    else:
        print("Images have varying sizes.")
    ```
    Output: 
    ```python 
    Number of images: 90  
    All images have the same size: (640, 640)
    ```
5. **Repeat the same step for validation dataset** 

    Output:   
    ```python
    Number of images: 90    
    All images have the same size: (640, 640)
    ```

5. **Sample random images from train dataset to see how the data looks**:  
    Wrote a function **fetch_image** to retrieve images from github URL.  
    Wrote a function **select_images** to pick equal spaced 8 images as a sample from the image URLs in train dataset.  

    ```python
        # Function to fetch image from URL
    def fetch_image(url):
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))

    # Select 8 images from the list of file URLs at equal intervals
    def select_images(image_urls, num=8):
        total_images = len(image_urls)
        return [image_urls[i] for i in range(0, total_images, max(total_images // num, 1))][:num]

    # URLs of images to be displayed
    selected_image_urls = select_images([file['download_url'] for file in files if file['name'].endswith('.jpg')])

    # Display the images in a 2x4 grid
    fig, axes = plt.subplots(2, 4, figsize=(20, 11))

    for ax, img_url in zip(axes.ravel(), selected_image_urls):
        image = fetch_image(img_url)
        ax.imshow(image)
        ax.axis('off')

    plt.suptitle('Sample Images from Dataset', fontsize=20)
    plt.tight_layout()
    plt.show()
    ```
    Output:

    ![Output](https://github.com/Sarthak-Kumar21/Traffic_object_detection/blob/main/Screenshots/View%20sample%20images%20in%20dataset.jpg?raw=true)

6. **Update .yaml file with current directories of train and validation data**  
    ```python
    # Path to data.yaml file
    yaml_file = '/content/data.yaml'

    # Updated configuration with local paths
    updated_content = """train: /content/Traffic_object_detection/Vehicle_Detection_Image_Dataset/train/images
    val: /content/Traffic_object_detection/Vehicle_Detection_Image_Dataset/valid/images

    nc: 1
    names: ['Vehicle']
    """

    # Write the updated content to the .yaml file
    with open(yaml_file, 'w') as file:
        file.write(updated_content)

    print("Updated .yaml file with local paths.")
    ```

7. **Train YOLOv8 Model on custom dataset**  
    Now, we can fine-tune our YOLOv8 pre-trained object detection model using transfer learning, specifically tailoring it to our 'Top-View Image Dataset'.  
    By leveraging the YOLOv8 model's existing weights from its training on the comprehensive COCO dataset, we start from a robust foundation rather than from scratch.   
    This approach saves significant time and resources and also capitalizes on our focused dataset to enhance the model's ability to accurately recognize and detect vehicles in top-view images.   

    ```python
    # Train the model on our custom dataset
    results = model.train(
    data='data.yaml',     # Path to the dataset configuration file
    epochs=5,              # Number of epochs to train for
    imgsz=640,               # Size of input images as integer
    device='cpu',                # Device to run on, here cpu.
    patience=50,             # Epochs to wait for no observable improvement for early stopping of training
    batch=32,                # Number of images per batch
    optimizer='auto',        # Optimizer to use, choices=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto]
    lr0=0.0001,              # Initial learning rate
    lrf=0.1,                 # Final learning rate (lr0 * lrf)
    dropout=0.1,             # Use dropout regularization
    seed=0                   # Random seed for reproducibility
    )
    ```
8. **Train Output files**

    ![Output](https://github.com/Sarthak-Kumar21/Traffic_object_detection/blob/main/Screenshots/Post_training_metrics.jpg?raw=true)
    ```python
    args.yaml					                        P_curve.png	        train_batch2.jpg
    confusion_matrix_normalized.png	                    PR_curve.png        val_batch0_labels.jpg
    confusion_matrix.png				                R_curve.png	        val_batch0_pred.jpg
    events.out.tfevents.1730032281.abdd5cf7c7ed.310.0   results.csv	        val_batch1_labels.jpg
    F1_curve.png					                    results.png	        val_batch1_pred.jpg
    labels_correlogram.jpg				                train_batch0.jpg    weights
    labels.jpg					                        train_batch1.jpg
    ```

    Here's a rundown of each item:

    1. **Weights Folder**: Contains the 'best.pt' and 'last.pt' files, which are the best and most recent weights of our trained model respectively.
    2. **Args**: A file that stores the arguments or parameters that were used during the training process.
    3. **Confusion Matrix**: Visual representations of the model performance. One is normalized, which helps in understanding the true positive rate across classes.
    4. **Events File**: Contains logs of events that occurred during training, useful for debugging and analysis.
    5. **F1 Curve**: Illustrates the F1 score of the model over time, balancing precision and recall.
    6. **Labels**: Shows the distribution of different classes within the dataset and their correlation.

9. **Model Evaluation Analysis** I will use the following factors:  
    1. Learning Curve Analysis
    2. Confusion Matrix Evaluation
    3. Performance Metrics Assessment

10. **Learning Curves Analysis**

    Create a function to plot learning curves for loss values:  

    ```python
    def plot_learning_curve(df, train_loss_col, val_loss_col, title):
    plt.figure(figsize=(12, 5))
    sns.lineplot(data=df, x='epoch', y=train_loss_col, label='Train Loss', color='#141140', linestyle='-', linewidth=2)
    sns.lineplot(data=df, x='epoch', y=val_loss_col, label='Validation Loss', color='orangered', linestyle='--', linewidth=2)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    ```
    
    By creating a **results_csv_path**, I call the above function to plot learning curves for Box Loss, Classisfication loss and Distribution Focal Loss.

    ```python
    results_csv_path = os.path.join(post_training_files_path, 'results.csv')

    # Load the CSV file from the constructed path into a pandas DataFrame
    df = pd.read_csv(results_csv_path)

    # Remove any leading whitespace 
    df.columns = df.columns.str.strip()

    # Plot the learning curves for each loss
    plot_learning_curve(df, 'train/box_loss', 'val/box_loss', 'Box Loss Learning Curve')
    plot_learning_curve(df, 'train/cls_loss', 'val/cls_loss', 'Classification Loss Learning Curve')
    plot_learning_curve(df, 'train/dfl_loss', 'val/dfl_loss', 'Distribution Focal Loss Learning Curve')
    ```
    ![Output](https://github.com/Sarthak-Kumar21/Traffic_object_detection/blob/main/Screenshots/Box_loss_learning_curve.jpg?raw=true)

    ![Output](https://github.com/Sarthak-Kumar21/Traffic_object_detection/blob/main/Screenshots/Classification%20Loss%20Learning%20Curve.jpg?raw=true)

    ![Output](https://github.com/Sarthak-Kumar21/Traffic_object_detection/blob/main/Screenshots/Focal_loss.jpg?raw=true)

    **Insights**:  
    1. Clearly, validation loss reflects a downward trend across all three Learning Curves.    
    2. After 4 epochs delta between training and validation loss is stagnating.
    3. The learning curves are not fully smooth, which implies that a state of equilibrium may be achieved after training with more epochs. However, rapid drop in loss values after initial epoch reflects that the model is learning effectively without overfitting.

11. **Confusion Matrix Evaluation**:  

    ![Output](https://github.com/Sarthak-Kumar21/Traffic_object_detection/blob/main/Screenshots/Confusion%20Matrix.jpg?raw=true)

    Insights:  
    The confusion matrix for our YOLOv8 vehicle detection model illustrates decent accuracy as mentioned earlier as well. In 77% of instances, the model successfully identifies the presence of a vehicle when there is one, indicating fair detection capability. Conversely, in 23% of cases, the model fails to detect a vehicle that is actually present, suggesting room for improvement in reducing false negatives.

12. **Performance Metrics Assessment**:  

    Delving into various metrics to understand model's predictive accuracy and areas of potential improvement.  
    
    ```python
    metrics/precision(B):   	0.844  
    metrics/recall(B): 	        0.859  
    metrics/mAP50(B):  	        0.916  
    metrics/mAP50-95(B):  	    0.624  
    fitness:	                0.653  
    ```
13. **Model Evaluation Insights**: 

    The YOLOv8 model shows decent results on the validation set. With a precision of **84.4%**, it indicates that the majority of the predictions made by the model are correct.  
    The recall score of **85.9%** demonstrates the model's ability to find most of the relevant cases in the dataset.   
    The model's mean Average Precision (mAP) at 50% Intersection over Union (IoU) is **91.6%**, reflecting fair accuracy in detecting objects with a considerable overlap with the ground truth. When the IoU threshold range is expanded from 50% to 95%, the model displays a mAP of **62.4%**. 
    Finally, the fitness score of **65.3%** indicates a okayish balance between precision, recall, and the IoU of the predictions.  
    Given limited computation and training for only 5 epochs, these results are impressive. However future improvements are currently ongoing.

***Best Regards***

