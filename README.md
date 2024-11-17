# Asteroid Classification and Visualization
- This project provides a comprehensive approach to classify asteroids based on their orbital and physical characteristics. The code leverages various machine learning algorithms to predict whether an asteroid is potentially hazardous (PHA) and visualizes the data for better understanding.

## Features
- Data Loading and Preprocessing: Reads asteroid data from a CSV file and checks for required columns and missing values.
- Feature Engineering: Calculates Earth Minimum Orbit Intersection Distance (MOID) and determines if an asteroid is potentially hazardous.
- Machine Learning Models: Implements several machine learning models, including:
Support Vector Machine (SVM)
Logistic Regression
Decision Tree
k-Nearest Neighbors (k-NN)
Naive Bayes
- Model Training and Evaluation: Trains models using grid search for hyperparameter tuning, evaluates them using metrics like accuracy, confusion matrix, and classification report, and visualizes the performance.
- Data Visualization: Utilizes 3D scatter plots to visualize asteroid attributes and orbits.
- Interactive User Interface: Allows users to search for asteroids by name and predict their hazardous status using trained models.
Usage
- Data Loading: The script reads asteroid data from a specified CSV file.
- Data Processing: Preprocesses and classifies data to prepare it for training.
- Model Training: Trains multiple models and stores them for future predictions.
- Visualization: Visualizes asteroid data and orbits in 3D plots.
- User Interaction: Provides a console-based interface for users to interact with the data and models.
## Dependencies
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- plotly
- scikit-learn
- imbalanced-learn

# Clone the repository:
```git clone https://github.com/Aryan-0001/asteroid-classification.git```
