# AutoML Suite & Pipeline Checker 🚀🧪

## 📜 Description

The AutoML Suite & Pipeline Checker is a comprehensive Streamlit web application designed to streamline and automate machine learning workflows. It empowers users to:
1.  **AutoML Workflow**: Upload datasets, automatically preprocess data, train a variety of classification and regression models, perform hyperparameter tuning, compare model performance, visualize results, and download the best-performing model pipelines.
2.  **Model Pipeline Checker**: An integrated section to upload and test previously saved `.pkl` model pipelines with new data, view predictions, infer task types, and visualize prediction outputs.

This application aims to make the initial stages of model building and evaluation faster and more accessible.

## ✨ Features

### Common to Both Sections
* **Interactive User Interface**: Built with Streamlit, featuring a custom dark theme for a pleasant user experience.
* **Welcome Guide**: An initial step-by-step tutorial to guide users through the application's functionalities.
* **Clear Error Handling and User Prompts**: To guide the user through the process smoothly.

### ⚙️ AutoML Workflow
* **Data Input**:
    * Upload custom CSV datasets.
    * Option to use built-in sample datasets (Wine, Breast Cancer, Diabetes, California Housing).
* **Preprocessing**:
    * Automatic scaling of numerical features and one-hot encoding of categorical features.
* **Configuration**:
    * Easy selection of the target variable from the dataset.
    * Automatic detection and manual selection of task type (Classification or Regression).
    * Optional SMOTE (Synthetic Minority Over-sampling Technique) for handling imbalanced classification data (requires `imbalanced-learn`).
* **Automated Model Training & Tuning**:
    * Trains and tunes a suite of models using `GridSearchCV`:
        * **Classification Models**: Logistic Regression, K-Nearest Neighbors, Decision Tree, Random Forest, Gradient Boosting, Gaussian Naive Bayes, Support Vector Classifier (SVC).
        * **Regression Models**: Linear Regression, Ridge, Lasso, K-Nearest Neighbors Regressor, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, Support Vector Regressor (SVR).
* **Performance Evaluation & Visualization**:
    * Displays a comparison table of models based on relevant metrics (Accuracy/F1 Score for classification, R²/RMSE for regression).
    * Detailed look at the best performing model, including its parameters.
    * Visualizations:
        * 3D PCA plot of the dataset.
        * Confusion Matrix (for classification).
        * Predicted vs. Actual values plot (for regression).
        * Feature Importances or Model Coefficients plot.
* **Model Export**:
    * Download trained model pipelines (including preprocessor and model) as `.pkl` files using `joblib`.
    * Download the fitted `LabelEncoder` for the target variable in classification tasks.
* **Sample Data Utilities**:
    * **Load & Prepare Sample CSV**: Loads the selected sample dataset for use in the AutoML workflow.
    * **Download Sample CSV**: Allows downloading the raw CSV of the selected sample dataset.
    * **Download Sample Test Set CSV**: Generates the selected sample dataset, performs a 75/25 split (with stratification attempted), and offers the 25% test portion for download, ideal for testing the generated pipelines.

### 🔎 Model Pipeline Checker
* **Integrated Testing Environment**: Appears as a section on the main page after the AutoML workflow.
* **Pipeline Upload**: Upload your saved `.pkl` model pipeline.
* **Data Upload**: Upload a new CSV data file for making predictions.
* **Intelligent Analysis**:
    * Attempts to extract and display expected input features from the pipeline's preprocessor.
    * Infers task type (Classification/Regression) based on model attributes and prediction output.
* **Prediction & Output**:
    * Generates and displays predictions from the loaded model on the uploaded data.
    * For classification, displays class probabilities if available.
    * Clearly states what the model is predicting (numerical values or class labels).
* **Visuals for Predictions**:
    * **Regression**: Histogram showing the distribution of predicted values.
    * **Classification**: Bar chart displaying the counts of predicted classes.
* **Download Predictions**: Option to download the generated predictions as a CSV file.
* **"How to Use" Guide**: A user-friendly and engaging manual within this section ("Your Model's Next Adventure") detailing how to load and use the `.pkl` pipeline in external Python projects, complete with copy-pasteable English code examples.

## 🚀 How to Run

1.  **Prerequisites**:
    * Ensure Python 3.7+ is installed on your system.

2.  **Setup**:
    * Download the `main.py` file.
    * It's highly recommended to create and activate a virtual environment:
        ```bash
        python -m venv venv
        # On Windows:
        # venv\Scripts\activate
        # On macOS/Linux:
        # source venv/bin/activate
        ```
    * Install the required dependencies:
        ```bash
        pip install -r requirements.txt
        ```

3.  **Launch the Application**:
    * Navigate to the directory containing `main.py` in your terminal.
    * Run the Streamlit application:
        ```bash
        streamlit run main.py
        ```
    * The application will open in your default web browser.

## 🗺️ Using the Application

Upon launching the app, you'll be greeted with a **Welcome Guide**. This guide provides a step-by-step tutorial on how to use both the AutoML Workflow and the Model Pipeline Checker sections. Follow the on-screen instructions to get started!

Briefly:
1.  Use the **AutoML Workflow** section (top part of the page, controls in the sidebar) to upload your data (or use a sample), configure your target and task, and train models. Download the `.pkl` file of the best model.
2.  Then, scroll down to the **Model Pipeline Checker** section to upload the downloaded `.pkl` file and a new CSV data file (perhaps the sample test set you downloaded) to see it in action and get predictions.

## 🛠️ Technology Stack

* **Python 3**
* **Streamlit**: For creating the interactive web application.
* **Pandas**: For data manipulation and analysis.
* **NumPy**: For numerical operations.
* **Scikit-learn**: For machine learning tasks (preprocessing, modeling, metrics, pipelines).
* **Plotly & Plotly Express**: For interactive data visualizations.
* **Joblib**: For saving and loading trained model pipelines.
* **Imbalanced-learn**: (Optional) For SMOTE functionality to handle imbalanced datasets in classification.

## 🤖 Acknowledgment of AI Assistance

This project represents my effort in conceptualizing, designing, and building a functional AutoML tool. Throughout its development, I engaged with AI-based coding assistants. This collaboration was instrumental in various stages, including brainstorming features, refining code logic, debugging complex issues, generating boilerplate code, and exploring different implementation strategies. While the core direction and feature set were driven by my vision, the AI's assistance played a valuable role in accelerating the development process and helping to bring this application to its current state. Furthermore, after the main development was complete, AI tools were also utilized to review and format the codebase, ensuring it adheres to high-quality standards and best practices. (This ReadMe file is written using AI 🙂)