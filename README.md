# Kaggle-CSIRO-Image2Biomass-Prediction
**2026/01/29 - Silver Medal**  
[Competition Link](https://www.kaggle.com/competitions/csiro-biomass/overview)
![image](https://github.com/RichardLiu083/Kaggle-HuBMAP-HPA-Hacking-the-Human-Body/blob/main/img/Rank.png)

## Insight
- Using stratified k-fold to split the dataset by Sampling_Date、State、Species, to make sure every samples under each environment and time interval are exist in each fold.
- Since the dataset is limited and consists of natural scene imagery, using a pretrained model yields substantially better results than training from scratch.
- Given the linear relationship between the five target labels, it is theoretically possible to predict only three and derive the others. However, empirical results show that letting the model learn and output all five values directly results in better overall accuracy.
- It is vital to keep images at their original scale. Manually scaling or distorting the images during training and prediction will only result in poorer model performance.
- Freezing model backbone while training, only train linear layer to avoid overfitting on small dataset.
- To ensure stable convergence, the EMA method was applied to the model weights throughout the training process.
  
## Model
- DinoV3
- We extract the intermediate layer outputs of DinoV3, flatten the tensors, and then feed them into a linear layer.  

## Augmentation
- RandomBrightnessContrast
- CLAHE
- HueSaturationValue
- Blur
- HorizontalFlip
- VerticalFlip
- ShiftScaleRotate
- Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Training
- 100 epochs
- lr 3e-4 for training linear layer
- MAE loss

## Validation
- 5 fold models
- only choose best model from validation

## Inference
- 5 fold model ensemble
- If the model predicts a negative value, we manually set it to zero as a post-processing step.
