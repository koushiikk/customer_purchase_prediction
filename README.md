# 🚗 Customer Purchase Prediction using Decision Tree

Customer Purchase Prediction is a key application of supervised machine learning. Using the **Decision Tree Classification** algorithm, companies can predict whether a customer will purchase a product based on demographic features, enabling better marketing strategies and targeted advertisements.

---

## 📌 Table of Contents

- [Overview](#overview)
- [What is Customer Purchase Prediction?](#what-is-customer-purchase-prediction)
- [What is the Decision Tree Algorithm?](#what-is-the-decision-tree-algorithm)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Steps for Implementation](#steps-for-implementation)
- [Model Training & Tuning](#model-training--tuning)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)

---

## Overview

In this machine learning project, we use the **Decision Tree Classification algorithm** to predict whether a customer will purchase a product based on features like **Age** and **Estimated Salary**.

The project includes:
- Data preprocessing  
- Feature scaling  
- Model training  
- Overfitting analysis  
- Hyperparameter tuning  

---

## What is Customer Purchase Prediction?

Customer Purchase Prediction is the process of analyzing customer data to determine whether a person is likely to buy a product.

Businesses use this to:
- Target potential customers  
- Improve marketing strategies  
- Increase conversion rates  
- Reduce unnecessary advertising costs  

---

## What is the Decision Tree Algorithm?

A **Decision Tree** is a supervised machine learning algorithm used for classification and regression tasks.

### How Decision Tree Works:
1. Select the best feature using metrics like **Entropy** or **Gini Index**
2. Split the dataset into subsets based on feature values
3. Repeat the process recursively to create branches
4. Stop when a leaf node (final decision) is reached

### Key Points:
- Easy to understand and interpret  
- Works for both linear and non-linear data  
- Prone to **overfitting** if not controlled  

---

## Dataset

The dataset (`Customer_Data.csv`) contains customer information.

- **Features Used:**
  - `Age`
  - `EstimatedSalary`
- **Target Variable:**
  - `Purchased` (0 = No, 1 = Yes)

✔ No missing values  
✔ No categorical encoding required  

---

## Project Structure

```text
Customer_Purchase_Prediction/
│
├── Decision_Tree.ipynb
├── Customer_Data.csv
├── README.md
└── requirements.txt
