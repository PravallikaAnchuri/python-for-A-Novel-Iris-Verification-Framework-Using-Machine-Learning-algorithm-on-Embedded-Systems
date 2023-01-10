import numpy as np
import cv2
from sklearn.svm import SVC

# Load iris data and labels
iris_data = np.load("iris_data.npy")
iris_labels = np.load("iris_labels.npy")

# Split data into training and test sets
train_data = iris_data[:120]
train_labels = iris_labels[:120]
test_data = iris_data[120:]
test_labels = iris_labels[120:]

# Train support vector machine (SVM) classifier
clf = SVC(kernel="linear")
clf.fit(train_data, train_labels)

# Test classifier on test data
accuracy = clf.score(test_data, test_labels)
print("Test accuracy:", accuracy)

# Function to extract iris features from an image
def extract_iris_features(image):
  # Convert to grayscale
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
  # Detect circles in the image
  circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
  
  # Extract features from the circles
  features = []
  for circle in circles:
    x, y, r = circle
    features.append(gray[y-r:y+r, x-r:x+r])
    
  return features

# Function to verify an iris
def verify_iris(image, user_id):
  # Extract features from the iris image
  features = extract_iris_features(image)
  
  # Predict class for each feature
  labels = []
  for feature in features:
    label = clf.predict(feature.reshape(1, -1))
    labels.append(label)
    
  # Return True if the majority of the predictions match the user's ID
  return sum(labels) > len(labels) // 2

# Test iris verification on an example image
image = cv2.imread("example_iris.jpg")
user_id = 0
if verify_iris(image, user_id):
  print("Iris verified for user ID", user_id)
else:
  print("Iris not verified for user ID", user_id)
