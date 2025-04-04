# Mushroom Classification with PySpark and Scikit-learn

## Project Overview

This project compares **PySpark** and **Scikit-learn** in classifying a **Mushroom dataset** using **Random Forest** models and **Gradient Boosting Classifier**. The Mushroom dataset contains various attributes that determine whether a mushroom is **edible** or **poisonous**.

The models employed for classification are:
- **Random Forest Classifier** by PySpark
- **Gradient Boosting Classifier** by Scikit-learn

Both models are renowned for their **accuracy** and ability to provide **feature importance insights**. This project demonstrates how each tool can be used to handle data preprocessing, model training, and performance evaluation.

## Dataset

The **Mushroom Dataset** consists of various features related to mushrooms, such as:
- **Cap shape**
- **Cap color**
- **Odor**
- **Gill size**
- **Spore print color**
- And several other features that help determine whether the mushroom is edible or poisonous.

The target variable is a categorical label (`edible` or `poisonous`), which is what the models aim to predict.

## Objectives

- **Data Preprocessing**: Prepare the dataset for model training by handling missing values, encoding categorical variables, and scaling features.
- **Model Training**: Train the classification models using the Random Forest algorithm in PySpark and the Gradient Boosting Classifier in Scikit-learn.
- **Performance Evaluation**: Compare the performance of both models, using metrics like **accuracy**, and **feature importance** to evaluate which tool performs better for this task.

## Tools & Libraries Used

- **PySpark**: 
  - Used for handling large datasets and distributed computation.
  - Random Forest Classifier was utilized for classification tasks.
- **Scikit-learn**: 
  - Efficient for smaller datasets.
  - Gradient Boosting Classifier was used for comparison.
- **Matplotlib & Seaborn**: 
  - Used for visualizations, including feature importance and model performance.
- **Pandas**: 
  - Used for data manipulation and conversion between PySpark DataFrame and Pandas DataFrame.

## Approach

### Data Preprocessing
Both models required similar steps for preprocessing:
1. **Data cleaning**: Removing missing or irrelevant values.
2. **Feature Encoding**: Transforming categorical features into numerical values using techniques such as **StringIndexer** (PySpark) or **LabelEncoder** (Scikit-learn).
3. **Feature Selection**: Selecting the most relevant features for model training.
4. **Splitting Data**: Dividing the data into training and testing sets.

### Model Training

- **PySpark**:
  - Used the **Random Forest Classifier** for model training.
  - PySpark was preferred for handling larger datasets due to its distributed processing capabilities.
  - Model training was conducted using a **RandomForestClassifier** with 50 trees.
  
- **Scikit-learn**:
  - Used the **Gradient Boosting Classifier** for model training.
  - This model is generally more efficient for smaller datasets and provides excellent predictive performance.
  - Model training was conducted using a **GradientBoostingClassifier** from the `sklearn` library.

### Performance Evaluation
Both models were evaluated using **accuracy** as the primary metric. Additionally, the **feature importance** from each model was extracted to provide insights into which features were most influential in predicting the target variable (edible or poisonous).

### Comparison

- **PySpark**:
  - Excelled in handling **large datasets** due to its distributed computing capabilities.
  - The **Random Forest** model in PySpark is robust, scalable, and can handle massive datasets efficiently.
  
- **Scikit-learn**:
  - Showed high **efficiency** and **accuracy** with **smaller datasets**.
  - The **Gradient Boosting Classifier** outperformed the Random Forest model in terms of accuracy on smaller datasets.
  - Feature importance from the Gradient Boosting model provided valuable insights.

## Results

Both models demonstrated comparable **accuracy** on the Mushroom dataset, indicating that both tools can perform classification tasks effectively. However, PySpark showed an advantage when dealing with larger datasets, while Scikit-learn performed exceptionally well with smaller datasets.

## Conclusion

This project highlights the strengths and differences between **PySpark** and **Scikit-learn** when applied to machine learning tasks. While **PySpark** is ideal for working with **large datasets** and distributed computation, **Scikit-learn** offers **efficiency** and **high accuracy** with **smaller datasets**. The choice of tool should depend on the size and complexity of the dataset being used.

## Key Takeaways

- **PySpark** is suitable for **large-scale** data processing and distributed computation.
- **Scikit-learn** is efficient and performs well for **smaller datasets**, providing tools for rapid model training.
- Both tools offer **feature importance** insights that are crucial for understanding model behavior and making informed decisions.
  
## How to Run the Project

To run this project on your local machine:

1. Clone the repository:
   ```bash
   git clone https://github.com/Kam-Willy/Pyspark-Mushroom-Classification.git
   cd Pyspark-Mushroom-Classification

