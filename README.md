# Model Comparison Dashboard
This is a simple Dash based framework to compare the performance of two trained models on test sets, done as part of a freelancing project.

## Overview
The Dash app compares two saved models on a selected test set and generates 3 graphs for each model. The models and the test sets are selected by the user through a drop-down from the files present in the root directory. 

[![Screenshot-2022-11-22-at-3-10-58-AM.png](https://i.postimg.cc/02PM6PXT/Screenshot-2022-11-22-at-3-10-58-AM.png)](https://postimg.cc/nMScgbJ2)

The following 3 graphs are generated for both models on the test set: 
1. Precision Recall Curve
2. ROC Curve
3. Feature Importance 

[![Screenshot-2022-11-22-at-3-13-13-AM.png](https://i.postimg.cc/k4yCL3NS/Screenshot-2022-11-22-at-3-13-13-AM.png)](https://postimg.cc/ftk10PrW)
[![Screenshot-2022-11-22-at-3-13-49-AM.png](https://i.postimg.cc/6q1kN988/Screenshot-2022-11-22-at-3-13-49-AM.png)](https://postimg.cc/D4Ljr3dh)

## Usage
There are sample model and features/labels files in the sample folder above (which need to be in the same directory as the code). 
This framework can be easily modified to generate other graphs or use files with different formats and is intended to serve as a starting point for other projects.
