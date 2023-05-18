# Handwritten digits recognition

## 1. Install the requirements
```python
pip install -r requirements.txt
```

## 2. Run the programm
```python
python main.py
```

The programm will train the model the first time you run it, then it will save it in a file called "model.h5" and load it the next time you run the programm.

## 3. Options
```python
python main.py -t
```
This option will force the programm to train the model even if the "model.h5" file already exist.

## 4. Results
The programm will print the accuracy and the loss of the model on the test dataset.
Then it will try to predict the class of the images in the "test" folder and print the result in the console, it will also display the image to see if the prediction is correct.

## 5. Dataset
The dataset used is the "mnist 0-9" dataset from Kaggle: https://www.kaggle.com/c/digit-recognizer/data. It contains 42000 images of handwritten digits from 0 to 9. The images are 28x28 pixels in grayscale.

