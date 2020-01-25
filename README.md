## Purpose

A project made to determine the building numbers and directions in a University Campus. This version does not support live video
camera feed.

## Dependencies

- openCV-Python
- Python 3.5+
- numpy
- pandas
- sklearn

## Explanation

- I tried to determine the places where a digit or a directional arrow can be found on a image by applying morphological
filters using openCV, with some hyper-parameter optimization I could get most of the areas of interest correctly.

- Then, I created a dataset by cropping those areas of interest and labeling them with hand as what digit they are.

- Applied data-augmentation(rotating, adding noise, shifting) methods to the labeled dataset, made the number of samples on each number equal.

- Feed those dataset to SVM and KNN by getting the histogram of gradients(HOG) features. 

- Model had %96.2 accuracy with SVM.

Model Running Video: https://vimeo.com/370869897
