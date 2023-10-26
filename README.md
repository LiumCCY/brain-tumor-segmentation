## brain-tumor-segmentation

### Switch to new branch

### Resut    
 * UNet3+ Training result     
    - Training Dice Score: 0.916    
    - Training Pearson Correlation Coefficient: 0.926   
    - Validation Dice Score: 0.822   
    - Validation Pearson Correlation Coefficient: 0.839   
 * ResUNet Training result    
    - Training Dice Score: 0.883   
    - Training Pearson Correlation Coefficient: 0.913     
    - Validation Dice Score: 0.737  
    - Validation Pearson Correlation Coefficient: 0.780   

### Environment
`pip install -r requirements.txt` 

### Directory
 * Modify saving directory like checkpoint, record and model via `config.py`

### Training
 * Model selection and loss function selection in `train.py`
 `python train.py`

### Plot
 * Plot training process through `plot/plot_result.py`
 * Predict through `plot/predict.py`

### Result
 * Few results saved in `result`
