import streamlit as st
import pandas as pd
import numpy as np
import joblib # Model save/load ke liye
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV # train_test_split yahaan bhi zaroori hai test set banane ke liye
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, f1_score, r2_score, mean_squared_error, confusion_matrix)
from sklearn.base import BaseEstimator, TransformerMixin, clone
import plotly.express as px
import plotly.figure_factory as ff
from io import BytesIO
from collections import Counter
import os # For path operations in checker's download filename

# Sklearn ke sample datasets
from sklearn.datasets import load_wine, load_breast_cancer, load_diabetes, fetch_california_housing

# imblearn se SMOTE (agar install kiya hai toh)
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImblearnPipeline # SMOTE ke saath pipeline ke liye
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False

# Machine Learning Models
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR # Support Vector Machines
from sklearn.naive_bayes import GaussianNB # Seedha saadha Naive Bayes

# --- Custom Dark Theme CSS (Ek hi baar define karna hai) ---
DARK_THEME_CSS = """
<style>
    body { color: #E0E0E0; }
    .stApp { background_color: #0d1117; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    /* Sidebar Buttons: Default grey */
    [data-testid="stSidebar"] .stButton>button, 
    [data-testid="stSidebar"] .stDownloadButton>button { 
        background-color: #30363d; color: #c9d1d9; border: 1px solid #444c56; 
        border-radius: 6px; padding: 8px 15px; font-weight: 500; width: 100%; margin-bottom: 8px;
    }
    [data-testid="stSidebar"] .stButton>button:hover, 
    [data-testid="stSidebar"] .stDownloadButton>button:hover { 
        opacity: 0.85; border-color: #58a6ff; background-color: #444c56;
    }
    /* Specific Sidebar Button for Load Sample in AutoML - Green */
    .sidebar-load-sample-button button { 
        background-color: #204423 !important; color: white !important;
    }
    .sidebar-load-sample-button button:hover {
        background-color: #27522a !important;
    }


    h1, h2, h3, h4, h5, h6 { color: #58a6ff; }
    .stMarkdown, p, label, .stTextInput>label, .stSelectbox>label, .stRadio>label { color: #c9d1d9; }
    /* Main Area Buttons */
    .main .stButton>button, .stButton>button { /* Default green for main action buttons */
        background-color: #238636; color: white !important; border: 1px solid #30363d !important; 
        border-radius: 6px !important; padding: 10px 18px !important; font-weight: 500 !important; 
        transition: background-color 0.2s ease-in-out, border-color 0.2s ease-in-out, box-shadow 0.2s ease-in-out !important; 
        outline: none !important; box-shadow: none !important; 
    }
    .main .stButton>button:hover, .stButton>button:hover { 
        background-color: #2ea043 !important; border-color: #444c56 !important; color: white !important; 
    }
    .main .stButton>button:focus, .stButton>button:focus,
    .main .stButton>button:active, .stButton>button:active { 
        background-color: #238636 !important; color: white !important; border-color: #58a6ff !important; 
        box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.3) !important; outline: none !important; 
    }
    /* Main area download button */
    .main .stDownloadButton>button { 
        background-color: #30363d; color: #c9d1d9; border: 1px solid #444c56; 
        border-radius: 6px; padding: 10px 18px; font-weight: 500; 
    }
    .main .stDownloadButton>button:hover { 
        background-color: #444c56; border-color: #58a6ff; 
    }


    .stDataFrame { font-size: 1.0em; background-color: #161b22; }
    .stExpander { border: 1px solid #30363d; border-radius: 6px; background-color: #161b22; }
    .stExpander header { font-size: 1.1em; color: #58a6ff; }
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div, .stRadio div[data-baseweb="radio"] { background-color: #0d1117; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; }
    .stProgress > div > div { background-color: #2ea043; }
    .stAlert, .stWarning, .stInfo, .stError { border-radius: 6px; padding: 1em; }
</style>
"""

# --- Utility Functions ---

# Function: Welcome screen aur tutorial guide dikhane ke liye
def render_welcome_guide():
    st.balloons() # Thoda Swaagat toh banta hai!
    st.title("🎉 Welcome to the AutoML Suite! 🎉")
    st.markdown("Your one-stop solution for automated machine learning and model testing. Let's get you started!")
    st.markdown("---")

    st.header("🚀 Quick Start Guide")
    st.markdown("""
    This application is divided into two main parts:
    1.  **AutoML Workflow**: To upload data, train multiple ML models, and download the best one.
    2.  **Model Pipeline Checker**: To test your downloaded `.pkl` model pipeline with new data.
    """)
    st.markdown("---")

    cols_guide = st.columns(2)
    with cols_guide[0]:
        st.subheader("Part 1: AutoML Workflow ⚙️")
        st.markdown("""
        *Step-by-step to train and get your model:*

        1.  **Upload Data (Sidebar)**:
            * Use the "Upload CSV" button in the sidebar to upload your dataset.
            * Alternatively, select a "Sample Dataset" from the dropdown and click "Load & Prepare Sample CSV".

        2.  **Configure (Sidebar)**:
            * Once data is loaded, "Choose the target column" you want to predict.
            * "Select task type" (Classification or Regression). The app tries to guess, but you can change it.
            * For Classification, you can opt to "Use SMOTE" for imbalanced data (if `imbalanced-learn` is installed).

        3.  **Train Models (Main Panel)**:
            * Scroll to the "🤖 Model Training & Comparison" section.
            * Click the "Train & Tune Models" button. This will train several models and show their performance.

        4.  **Review & Download (Main Panel)**:
            * Examine the "🏆 AutoML Model Performance" table.
            * Detailed plots for the best model will be shown below it.
            * In the "💾 Save Trained AutoML Pipeline" section, choose your preferred model from the dropdown.
            * Click "⬇️ Download ... Pipeline (.pkl)" to save it.
            * If it's a classification task, also download the "Target Encoder (.pkl)" if available.
        """)

    with cols_guide[1]:
        st.subheader("Part 2: Model Pipeline Checker 🔎")
        st.markdown("""
        *After downloading a `.pkl` pipeline from the AutoML Workflow, test it here! This section appears below the AutoML Workflow on the main page.*

        1.  **Upload Model (Checker Section)**:
            * In the "Model Pipeline Checker" section, use the "1. Upload Model Pipeline (.pkl)" button to upload your saved `.pkl` model.

        2.  **Upload Data (Checker Section)**:
            * Use the "2. Upload Data for Prediction (.csv)" button to upload a new CSV file with data for prediction.
            * *Tip: You can use the "Download Sample Test Set CSV" button in the AutoML sidebar to get a test dataset corresponding to the sample data you might have trained on.*

        3.  **Generate Predictions (Checker Section)**:
            * Once both files are loaded, click the "🚀 Generate Predictions (Checker)" button.

        4.  **Review (Checker Section)**:
            * See the "🔮 Prediction Results", including the inferred task type.
            * Check out the "📈 Distribution of Predicted Values" (for regression) or "📊 Count of Predicted Classes" (for classification).
            * You can also "⬇️ Download Predictions as CSV".

        5.  **Learn More (Checker Section)**:
            * Refer to the "🚀 Your Model's Next Adventure: A 'How-To' Guide!" at the end of the Checker section for detailed Python code examples on using your pipeline in your own projects.
        """)
    
    st.markdown("---")
    # Centering the button using columns, a common workaround
    col1, col2, col3 = st.columns([1,2,1]) 
    with col2:
        if st.button("✅ Got it! Let's Go to the App", key="go_to_app_main_btn", help="Click here to start using the application"):
            st.session_state.show_welcome_guide = False
            st.rerun()


# Function: Alag alag tasks ke liye sample data generate karega
def generate_sample_data(task_key):
    df = None
    target_col_name = 'target'
    if task_key == "classification_wine":
        data = load_wine(); df = pd.DataFrame(data.data, columns=data.feature_names); df[target_col_name] = data.target_names[data.target]
    elif task_key == "classification_breast_cancer":
        data = load_breast_cancer(); df = pd.DataFrame(data.data, columns=data.feature_names); df[target_col_name] = data.target_names[data.target]
    elif task_key == "regression_diabetes":
        data = load_diabetes(); df = pd.DataFrame(data.data, columns=data.feature_names); df[target_col_name] = data.target
    elif task_key == "regression_california_housing":
        data = fetch_california_housing(); df = pd.DataFrame(data.data, columns=data.feature_names); df[target_col_name] = data.target
    return df, target_col_name

# Function: App ki shuruaat mein session state variables set karega
def init_session_state():
    if 'show_welcome_guide' not in st.session_state:
        st.session_state.show_welcome_guide = True
    # AutoML Workflow States
    automl_defaults = {
        'df': None, 'uploaded_file_name': None, 'target': None, 'previous_target': None,
        'task_type': "classification", 'results': [],
        'trained_models_for_saving': [],
        'X_train_processed_for_fitting_smote_model': None, 'X_test_processed_for_predict': None,
        'y_train_encoded_for_fitting_smote_model': None, 'y_test_encoded_for_predict': None,
        'le_target': None, 'feature_names_processed': None,
        'original_X_for_pipeline_fitting': None, 'original_y_for_pipeline_fitting': None,
        'main_preprocessor_fitted_on_full_X': None,
        'sample_data_intended_target': None,
        'train_button_clicked_once': False, 'task_type_radio_idx': 0,
        'use_smote_for_classification': True,
        'processed_uploaded_file_id_automl': None, 'df_source_is_upload_automl': False,
        'sample_data_for_download': None, 
        'sample_test_set_for_download': None, 
    }
    for key, value in automl_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Model Pipeline Checker States
    checker_defaults = {
        'model_pipeline_checker': None, 'data_df_checker': None,
        'predictions_df_checker': None, 'task_type_guess_checker': "Unknown",
        'expected_features_checker': None, 'uploaded_model_name_checker': None,
        'uploaded_data_name_checker': None,
    }
    for key, value in checker_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# --- Model Pipeline Checker related Functions ---

# Function: Checker ke liye model aur data process karke predictions dega
def process_model_and_data_checker():
    # Yeh function model_pipeline_checker, data_df_checker se data lega aur relevant states update karega
    if st.session_state.model_pipeline_checker and st.session_state.data_df_checker is not None:
        model = st.session_state.model_pipeline_checker
        data_for_prediction = st.session_state.data_df_checker.copy()

        try: # Expected features nikalne ki koshish
            if 'preprocessor' in model.named_steps:
                preprocessor_step = model.named_steps['preprocessor']
                if hasattr(preprocessor_step, 'feature_names_in_'):
                    st.session_state.expected_features_checker = list(preprocessor_step.feature_names_in_)
                    missing_cols = [col for col in st.session_state.expected_features_checker if col not in data_for_prediction.columns]
                    if missing_cols:
                        st.error(f"The uploaded data for checker is missing expected columns: {missing_cols}.")
                        st.session_state.predictions_df_checker = None; return
        except Exception as e:
            st.warning(f"Could not extract expected features from pipeline: {e}."); st.session_state.expected_features_checker = None

        try: # Predictions karne ki koshish
            predictions = model.predict(data_for_prediction)
            predictions_df = pd.DataFrame(predictions, columns=['Prediction_Value'])
            task_guess = "Unknown"; final_estimator = model.steps[-1][1]

            if hasattr(model, 'predict_proba') or hasattr(final_estimator, 'predict_proba'):
                task_guess = "Classification"
                try:
                    probabilities = model.predict_proba(data_for_prediction)
                    prob_cols = [f"Prob_Class_{i}" for i in range(probabilities.shape[1])]
                    if hasattr(final_estimator, 'classes_') and len(final_estimator.classes_) == probabilities.shape[1]:
                           prob_cols = [f"Prob_{str(cls)}" for cls in final_estimator.classes_]
                    prob_df = pd.DataFrame(probabilities, columns=prob_cols)
                    predictions_df = pd.concat([predictions_df.rename(columns={'Prediction_Value': 'Predicted_Label'}), prob_df], axis=1)
                except Exception as e_proba: st.warning(f"Could not get prediction probabilities: {e_proba}")
            elif hasattr(final_estimator, 'classes_'):
                 task_guess = "Classification"; predictions_df = predictions_df.rename(columns={'Prediction_Value': 'Predicted_Label'})
            else:
                if predictions.dtype == 'float': task_guess = "Regression"
                elif pd.api.types.is_integer_dtype(predictions.dtype) and len(np.unique(predictions)) > 20 : task_guess = "Regression (Discrete)"
            st.session_state.task_type_guess_checker = task_guess; st.session_state.predictions_df_checker = predictions_df
            st.success("Checker: Predictions generated successfully!")
        except Exception as e:
            st.error(f"Checker: Error during prediction: {e}."); st.session_state.predictions_df_checker = None
            st.session_state.task_type_guess_checker = "Error during prediction"

# Function: Model Pipeline Checker UI ko render karega
def render_model_checker():
    st.markdown("---") 
    st.header("🔎 Model Pipeline Checker")
    st.markdown("Upload your trained `.pkl` model pipeline and a CSV data file to test its predictions and see how to use it.")

    col1_checker, col2_checker = st.columns(2)
    with col1_checker:
        st.subheader("📤 Upload Files for Checker")
        uploaded_model_file_checker = st.file_uploader("1. Upload Model Pipeline (.pkl)", type="pkl", key="checker_model_upload")
        if uploaded_model_file_checker:
            if st.session_state.uploaded_model_name_checker != uploaded_model_file_checker.name:
                try:
                    model_bytes = BytesIO(uploaded_model_file_checker.getvalue())
                    st.session_state.model_pipeline_checker = joblib.load(model_bytes)
                    st.session_state.uploaded_model_name_checker = uploaded_model_file_checker.name
                    st.session_state.expected_features_checker = None; st.session_state.predictions_df_checker = None
                    st.success(f"Checker: Model '{uploaded_model_file_checker.name}' loaded.")
                    if 'preprocessor' in st.session_state.model_pipeline_checker.named_steps:
                        preprocessor_step = st.session_state.model_pipeline_checker.named_steps['preprocessor']
                        if hasattr(preprocessor_step, 'feature_names_in_'):
                            st.session_state.expected_features_checker = list(preprocessor_step.feature_names_in_)
                except Exception as e:
                    st.error(f"Checker: Error loading model: {e}")
                    st.session_state.model_pipeline_checker = None; st.session_state.uploaded_model_name_checker = None
    with col2_checker:
        st.subheader(" ") 
        uploaded_data_file_checker = st.file_uploader("2. Upload Data for Prediction (.csv)", type="csv", key="checker_data_upload")
        if uploaded_data_file_checker:
            if st.session_state.uploaded_data_name_checker != uploaded_data_file_checker.name:
                try:
                    st.session_state.data_df_checker = pd.read_csv(uploaded_data_file_checker)
                    st.session_state.uploaded_data_name_checker = uploaded_data_file_checker.name
                    st.session_state.predictions_df_checker = None
                    st.success(f"Checker: Data '{uploaded_data_file_checker.name}' loaded.")
                except Exception as e:
                    st.error(f"Checker: Error loading data CSV: {e}")
                    st.session_state.data_df_checker = None; st.session_state.uploaded_data_name_checker = None
    
    if st.session_state.model_pipeline_checker and st.session_state.data_df_checker is not None:
        if st.button("🚀 Generate Predictions (Checker)", key="checker_generate_preds_button_main_panel"):
            process_model_and_data_checker()

    if st.session_state.model_pipeline_checker:
        st.markdown(f"**Current Model for Checking:** `{st.session_state.uploaded_model_name_checker}`")
        if st.session_state.expected_features_checker:
            st.markdown(f"**Pipeline's Preprocessor Expects Input Features:**"); st.json(st.session_state.expected_features_checker)

    if st.session_state.data_df_checker is not None:
        st.markdown(f"**Current Data for Checking:** `{st.session_state.uploaded_data_name_checker}`")
        st.dataframe(st.session_state.data_df_checker.head(), height=150)

    if st.session_state.predictions_df_checker is not None:
        st.subheader("🔮 Prediction Results")
        task_type = st.session_state.task_type_guess_checker; st.info(f"**Inferred Task Type:** {task_type}")
        if task_type == "Regression" or task_type == "Regression (Discrete)": st.markdown("The model is predicting **numerical values**.")
        elif task_type == "Classification": st.markdown("The model is predicting **class labels**.")
        st.dataframe(st.session_state.predictions_df_checker)

        try: 
            if task_type == "Regression" or task_type == "Regression (Discrete)":
                if 'Prediction_Value' in st.session_state.predictions_df_checker.columns:
                    st.subheader("📈 Distribution of Predicted Values"); fig_hist = px.histogram(st.session_state.predictions_df_checker, x='Prediction_Value', title="Prediction Distribution", nbins=30, opacity=0.7); fig_hist.update_layout(bargap=0.1); st.plotly_chart(fig_hist, use_container_width=True)
            elif task_type == "Classification":
                if 'Predicted_Label' in st.session_state.predictions_df_checker.columns:
                    st.subheader("📊 Count of Predicted Classes"); counts_df = st.session_state.predictions_df_checker['Predicted_Label'].value_counts().reset_index(); counts_df.columns = ['Predicted_Label', 'Count']; fig_bar = px.bar(counts_df, x='Predicted_Label', y='Count', title="Predicted Class Counts", color='Predicted_Label'); st.plotly_chart(fig_bar, use_container_width=True)
        except Exception as e_viz: st.warning(f"Could not generate prediction visualization: {e_viz}")

        @st.cache_data 
        def convert_df_to_csv_for_download(df): return df.to_csv(index=False).encode('utf-8')
        csv_predictions = convert_df_to_csv_for_download(st.session_state.predictions_df_checker)
        download_file_name = f"predictions_on_{os.path.splitext(st.session_state.uploaded_data_name_checker)[0] if st.session_state.uploaded_data_name_checker else 'output'}.csv"
        st.download_button(label="⬇️ Download Predictions as CSV", data=csv_predictions, file_name=download_file_name, mime='text/csv', key='checker_download_preds_csv')
    
    elif st.session_state.model_pipeline_checker and st.session_state.data_df_checker is not None and not st.session_state.predictions_df_checker:
        st.markdown("Click 'Generate Predictions (Checker)' above after uploading both files.")

    st.markdown("---")
    st.header("🚀 Your Model's Next Adventure: A 'How-To' Guide!")
    st.markdown("""
    So, you've downloaded your shiny `.pkl` pipeline file from the AutoML Workflow! 🎉 
    Wondering how to unleash its predictive power in your own Python scripts or projects? 
    Let's embark on this coding quest together!
    """)

    st.markdown("#### Step 1: Awaken the Genie 🧞‍♂️ (Loading Your Pipeline)")
    st.markdown("First, you need to load your saved pipeline. It's like summoning a powerful ally!")
    st.code("""
import joblib
import pandas as pd

# Path to your saved .pkl pipeline file
pipeline_path = 'your_model_pipeline.pkl' # <-- IMPORTANT: Change this to your actual file path!

# Load the pipeline
try:
    loaded_pipeline = joblib.load(pipeline_path)
    print(f"Pipeline '{pipeline_path}' loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file '{pipeline_path}' was not found. Please check the path.")
except Exception as e:
    print(f"An error occurred while loading the pipeline: {e}")
    """, language='python')

    st.markdown("#### Step 2: Prepare Your Grand Feast 🍽️ (The Input Data)")
    st.markdown("""
    Your model is hungry for data! This data needs to be in a Pandas DataFrame and, crucially,
    it **must have the same features (columns) and data types** as the data used to train the model.
    The good news? The pipeline you saved includes the preprocessor, so it will handle things like
    scaling and encoding automatically, as long as the raw input features are correct!

    **Pro-Tip to know what features your pipeline expects (if it has a 'preprocessor' step):**
    """)
    st.code("""
# After loading the pipeline (from Step 1)
# try:
#     # Assumes your preprocessor step is named 'preprocessor'
#     preprocessor_step = loaded_pipeline.named_steps.get('preprocessor')
#     if preprocessor_step and hasattr(preprocessor_step, 'feature_names_in_'):
#         expected_features = list(preprocessor_step.feature_names_in_)
#         print(f"The pipeline's preprocessor expects these input features: {expected_features}")
#     else:
#         print("Could not automatically determine expected features from the 'preprocessor' step. Ensure your data matches the training data structure.")
# except Exception as e:
#     print(f"Error inspecting pipeline for feature names: {e}")

# Example: Creating new data for prediction
# Replace these with your actual feature names and data!
# feature_names = ['age', 'income', 'city_category'] # Example feature names
# new_data_list = [
#     [30, 50000, 'A'],
#     [45, 120000, 'B'],
#     [22, 35000, 'A']
# ]
# new_data_df = pd.DataFrame(new_data_list, columns=feature_names)
# print("\\nNew data for prediction:")
# print(new_data_df)
    """, language='python')

    st.markdown("#### Step 3: Unleash the Magic! ✨ (Making Predictions)")
    st.markdown("With your pipeline loaded and data ready, it's showtime!")
    st.code("""
# Assuming 'loaded_pipeline' and 'new_data_df' are from previous steps

# try:
#     predictions = loaded_pipeline.predict(new_data_df)
#     print("\\nPredictions:")
#     print(predictions)

    # If it's a classification model, you might also want probabilities
#     if hasattr(loaded_pipeline, 'predict_proba'):
#         probabilities = loaded_pipeline.predict_proba(new_data_df)
#         print("\\nPrediction Probabilities:")
#         print(probabilities)
# except Exception as e:
#     print(f"An error occurred during prediction: {e}")
#     print("Ensure your 'new_data_df' has the correct columns and data types expected by the pipeline.")
    """, language='python')

    st.markdown("#### Step 4 (Optional for Classification): The Grand Reveal 🎭 (Decoding Labels)")
    st.markdown("""
    If your task was classification and the target labels were numbers (because of Label Encoding),
    your predictions will also be numbers. To get back the original text labels (e.g., 'Yes', 'No', 'Spam'),
    you'll need the `target_label_encoder.pkl` file that the AutoML app might have offered for download.
    """)
    st.code("""
# Assuming 'predictions' are the numeric labels from Step 3
# And you have 'target_label_encoder.pkl' saved from the AutoML app

# try:
#     target_encoder = joblib.load('target_label_encoder.pkl') # <-- Change path if needed!
#     decoded_predictions = target_encoder.inverse_transform(predictions)
#     print("\\nDecoded Predictions (Original Labels):")
#     print(decoded_predictions)
# except FileNotFoundError:
#     print("\\n'target_label_encoder.pkl' not found. Cannot decode labels.")
# except Exception as e:
#     print(f"An error occurred while decoding labels: {e}")
    """, language='python')
    st.markdown("---")
    st.markdown("And that's it! You're now ready to use your powerful, custom-trained model pipeline anywhere you need it. Happy coding! 👍")

# Function: AutoML Workflow UI ko render karega
def render_automl_workflow():
    st.header("✨ AutoML Workflow")

    with st.expander("❗ Important Note on Model Accuracy & SMOTE", expanded=True):
        st.markdown("""
        Model accuracy depends on Data Quality, Feature Relevance, Dataset Size, Problem Complexity, Hyperparameters & CV Folds.
        Low scores (e.g., negative R² or low Accuracy/F1) suggest issues with these factors.
        For Classification: This app uses SMOTE by default if classes are imbalanced (requires `imbalanced-learn`).
        SMOTE is applied only to training data within CV folds. Toggle SMOTE in sidebar settings.
        **Crucially, for classification, ensure your target variable has at least 2 samples for *each* class to allow for data splitting and reliable training.**
        """)

    st.sidebar.header("AutoML: Data & Settings")
    uploaded_file_automl = st.sidebar.file_uploader("Upload CSV", type="csv", key="automl_file_uploader")

    if uploaded_file_automl is not None:
        if id(uploaded_file_automl) != st.session_state.get('processed_uploaded_file_id_automl'):
            try:
                st.session_state.df = pd.read_csv(uploaded_file_automl)
                st.session_state.uploaded_file_name = uploaded_file_automl.name
                st.session_state.processed_uploaded_file_id_automl = id(uploaded_file_automl)
                st.session_state.df_source_is_upload_automl = True
                st.session_state.target = None; st.session_state.previous_target = None
                st.session_state.task_type = "classification"; st.session_state.task_type_radio_idx = 0
                st.session_state.results = []; st.session_state.trained_models_for_saving = []
                st.session_state.X_train_processed_for_fitting_smote_model = None; st.session_state.X_test_processed_for_predict = None
                st.session_state.y_train_encoded_for_fitting_smote_model = None; st.session_state.y_test_encoded_for_predict = None
                st.session_state.le_target = None; st.session_state.feature_names_processed = None
                st.session_state.original_X_for_pipeline_fitting = None; st.session_state.original_y_for_pipeline_fitting = None
                st.session_state.main_preprocessor_fitted_on_full_X = None; st.session_state.sample_data_intended_target = None
                st.session_state.train_button_clicked_once = False
                st.sidebar.success(f"File '{uploaded_file_automl.name}' loaded.")
                st.rerun()
            except Exception as e: st.sidebar.error(f"Error loading CSV: {e}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("##### Sample Datasets")
    sample_options_automl = {
        "Wine (Classification)": "classification_wine", "Breast Cancer (Classification)": "classification_breast_cancer",
        "Diabetes (Regression)": "regression_diabetes", "California Housing (Regression)": "regression_california_housing"
    }
    selected_sample_key_name = st.sidebar.selectbox("Select Sample Dataset:", list(sample_options_automl.keys()), key="automl_sample_select")
    chosen_sample_key_automl = sample_options_automl[selected_sample_key_name]

    st.sidebar.markdown('<div class="sidebar-load-sample-button">', unsafe_allow_html=True) # For specific styling
    if st.sidebar.button("Load & Prepare Sample CSV", key="automl_load_sample_btn_new"): # Only one functional button
        sample_df, intended_target = generate_sample_data(chosen_sample_key_automl)
        st.session_state.df = sample_df
        st.session_state.uploaded_file_name = f"sample_{chosen_sample_key_automl}.csv"
        st.session_state.sample_data_intended_target = intended_target
        st.session_state.processed_uploaded_file_id_automl = None; st.session_state.df_source_is_upload_automl = False
        st.session_state.target = intended_target; st.session_state.previous_target = intended_target
        if "classification" in chosen_sample_key_automl: st.session_state.task_type = "classification"; st.session_state.task_type_radio_idx = 0
        else: st.session_state.task_type = "regression"; st.session_state.task_type_radio_idx = 1
        st.session_state.results = []; st.session_state.trained_models_for_saving = []; st.session_state.train_button_clicked_once = False
        st.sidebar.success(f"Sample '{selected_sample_key_name}' loaded.")
        st.rerun()
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    if st.sidebar.button("Download Sample CSV", key="automl_download_full_sample_btn"):
        full_sample_df, _ = generate_sample_data(chosen_sample_key_automl)
        if full_sample_df is not None: st.session_state.sample_data_for_download = full_sample_df
            
    if st.session_state.get('sample_data_for_download') is not None:
        csv_to_download = st.session_state.sample_data_for_download.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(label=f"⬇️ Download Full '{selected_sample_key_name}' CSV", data=csv_to_download,
                           file_name=f"sample_full_{chosen_sample_key_automl}.csv", mime="text/csv", key="automl_do_download_full_sample")
        st.session_state.sample_data_for_download = None 

    if st.sidebar.button("Download Sample Test Set CSV", key="automl_download_test_set_btn"):
        full_sample_df_for_split, target_col_name = generate_sample_data(chosen_sample_key_automl)
        if full_sample_df_for_split is not None and len(full_sample_df_for_split) > 1 :
            try:
                stratify_col = None
                if target_col_name and target_col_name in full_sample_df_for_split.columns and \
                   full_sample_df_for_split[target_col_name].nunique() > 1 and \
                   all(full_sample_df_for_split[target_col_name].value_counts(dropna=False) >=2) :
                     stratify_col = full_sample_df_for_split[target_col_name]
                _, df_test = train_test_split(full_sample_df_for_split, test_size=0.25, random_state=42, stratify=stratify_col)
                st.session_state.sample_test_set_for_download = df_test
                st.sidebar.info(f"Test set (25%) for '{selected_sample_key_name}' ready for download.")
            except ValueError as e_split: # Catch common splitting errors (e.g. too few samples in a class for stratification)
                 st.sidebar.warning(f"Could not stratify test split for '{selected_sample_key_name}', splitting without. Error: {e_split}")
                 _, df_test = train_test_split(full_sample_df_for_split, test_size=0.25, random_state=42) # Fallback
                 st.session_state.sample_test_set_for_download = df_test
            except Exception as e_general_split: st.sidebar.error(f"Error splitting test set: {e_general_split}")
        else: st.sidebar.warning(f"Not enough data in '{selected_sample_key_name}' to create a test split.")

    if st.session_state.get('sample_test_set_for_download') is not None:
        csv_test_set_to_download = st.session_state.sample_test_set_for_download.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(label=f"⬇️ Download Test Set '{selected_sample_key_name}'", data=csv_test_set_to_download,
                           file_name=f"sample_test_set_{chosen_sample_key_automl}.csv", mime="text/csv", key="automl_do_download_test_set")
        st.session_state.sample_test_set_for_download = None
    st.sidebar.markdown("---")

    if st.session_state.df is not None: 
        df_sidebar = st.session_state.df
        st.sidebar.subheader("AutoML: Target & Task")
        df_columns = df_sidebar.columns.tolist()
        if 'previous_target' not in st.session_state or st.session_state.previous_target is None or st.session_state.previous_target not in df_columns:
             st.session_state.previous_target = st.session_state.target if (st.session_state.target and st.session_state.target in df_columns) else (df_columns[-1] if df_columns else None)
        current_target_index = 0
        if st.session_state.target and st.session_state.target in df_columns: current_target_index = df_columns.index(st.session_state.target)
        elif df_columns:
            st.session_state.target = df_columns[-1]; current_target_index = len(df_columns) -1
            if st.session_state.previous_target not in df_columns or st.session_state.previous_target is None: st.session_state.previous_target = st.session_state.target
        if not st.session_state.df_source_is_upload_automl and st.session_state.get('sample_data_intended_target') and st.session_state.sample_data_intended_target in df_columns:
            if st.session_state.target != st.session_state.sample_data_intended_target:
                st.session_state.target = st.session_state.sample_data_intended_target; current_target_index = df_columns.index(st.session_state.sample_data_intended_target)

        new_target_selection = st.sidebar.selectbox("Choose the target column:", df_columns, index=current_target_index, key="automl_target_select")
        if st.session_state.previous_target != new_target_selection:
            st.session_state.target = new_target_selection; st.session_state.previous_target = new_target_selection
            st.session_state.results = []; st.session_state.trained_models_for_saving = []; st.session_state.train_button_clicked_once = False; st.session_state.sample_data_intended_target = None
            st.warning("Target variable changed. Previous results cleared. Please 'Train & Tune Models' again."); st.rerun()
        else: st.session_state.target = new_target_selection

        default_task_idx = 0
        if st.session_state.target and st.session_state.target in df_sidebar.columns:
            target_series = df_sidebar[st.session_state.target]
            if target_series.dtype == 'object' or target_series.nunique() < 0.05 * len(target_series) or target_series.nunique() <= 20: default_task_idx = 0
            else:
                try: pd.to_numeric(target_series); default_task_idx = 1
                except ValueError: default_task_idx = 0
        current_task_idx = st.session_state.get('task_type_radio_idx', default_task_idx)
        if st.session_state.df_source_is_upload_automl or (st.session_state.previous_target != st.session_state.target) : current_task_idx = default_task_idx
        
        selected_task = st.sidebar.radio("Select task type:", ["classification", "regression"], index=current_task_idx, key="automl_task_type_radio")
        if st.session_state.task_type != selected_task or st.session_state.task_type_radio_idx != current_task_idx :
            st.session_state.task_type = selected_task; st.session_state.task_type_radio_idx = 0 if selected_task == "classification" else 1
            st.session_state.results = []; st.session_state.trained_models_for_saving = []; st.session_state.train_button_clicked_once = False
            st.warning("Task type changed. Previous results cleared. Please 'Train & Tune Models' again."); st.rerun()

        if st.session_state.task_type == "classification" and IMBLEARN_AVAILABLE:
            st.sidebar.markdown("---"); st.sidebar.subheader("Data Balancing")
            st.session_state.use_smote_for_classification = st.sidebar.checkbox("Use SMOTE for Classification", value=st.session_state.get('use_smote_for_classification', True), key="automl_smote_checkbox", help="Requires `imbalanced-learn` library.")
        elif st.session_state.task_type == "classification" and not IMBLEARN_AVAILABLE:
            st.sidebar.markdown("---"); st.sidebar.warning("SMOTE not available. Install `imbalanced-learn`.")
            st.session_state.use_smote_for_classification = False
    else: st.sidebar.markdown("Upload a CSV or load sample data to configure AutoML.")

    # --- Main Panel Content for AutoML Workflow ---
    if st.session_state.df is not None:
        st.subheader("📊 Data Preview")
        st.dataframe(st.session_state.df.head())
        st.write(f"Shape of data: {st.session_state.df.shape}")

        if not st.session_state.target or st.session_state.target not in st.session_state.df.columns:
            st.error("Target variable not set or not found. Please select a valid target from the sidebar."); st.stop()

        X_original_full = st.session_state.df.drop(columns=[st.session_state.target])
        st.session_state.original_X_for_pipeline_fitting = X_original_full.copy()
        y_original_series = st.session_state.df[st.session_state.target]

        can_proceed_with_training = True # Flag to control training button
        if st.session_state.task_type == "classification":
            class_counts = Counter(y_original_series.astype(str)); min_samples_per_class_needed = 3
            problematic_classes = {cls: count for cls, count in class_counts.items() if count < min_samples_per_class_needed}
            if problematic_classes:
                error_msg = f"🚫 Critical Data Issue for Classification!\nTarget ('{st.session_state.target}') has classes with < {min_samples_per_class_needed} samples.\nProblematic: {problematic_classes}"
                st.error(error_msg); can_proceed_with_training = False
        
        y_original_full = None # Initialize
        if can_proceed_with_training:
            if st.session_state.task_type == "classification":
                le_target = LabelEncoder(); y_original_full = le_target.fit_transform(y_original_series.astype(str))
                st.session_state.le_target = le_target; st.session_state.original_y_for_pipeline_fitting = y_original_full.copy()
            else: # Regression
                try: y_original_full = pd.to_numeric(y_original_series); st.session_state.original_y_for_pipeline_fitting = y_original_full.copy(); st.session_state.le_target = None
                except ValueError: st.error(f"❌ Regression Target ('{st.session_state.target}') must be numeric."); can_proceed_with_training = False
        
        X_processed_full = None # Initialize
        if can_proceed_with_training and y_original_full is not None :
            numerical_features = X_original_full.select_dtypes(include=np.number).columns.tolist()
            categorical_features = X_original_full.select_dtypes(include=['object', 'category']).columns.tolist()
            numerical_pipeline = SklearnPipeline([('scaler', StandardScaler())])
            categorical_pipeline = SklearnPipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]) # sparse_output=False for dense array
            main_preprocessor = ColumnTransformer(transformers=[('num', numerical_pipeline, numerical_features),('cat', categorical_pipeline, categorical_features)], remainder='passthrough')
            try:
                X_processed_full = main_preprocessor.fit_transform(X_original_full)
                if not isinstance(X_processed_full, np.ndarray): X_processed_full = X_processed_full.toarray() if hasattr(X_processed_full, "toarray") else np.array(X_processed_full)
                st.session_state.main_preprocessor_fitted_on_full_X = main_preprocessor
                st.session_state.feature_names_processed = main_preprocessor.get_feature_names_out()
            except Exception as e: st.error(f"❌ Preprocessing failed: {e}."); can_proceed_with_training = False

        if can_proceed_with_training and X_processed_full is not None and X_processed_full.shape[0] > 0 and X_processed_full.shape[1] >= 3 and y_original_full is not None:
            st.subheader(" dimensionality_reduction PCA Visualization")
            try:
                pca = PCA(n_components=3, random_state=42); X_pca_full = pca.fit_transform(X_processed_full)
                color_labels_full = y_original_full; color_labels_legend_name = st.session_state.target
                if st.session_state.task_type == "classification" and st.session_state.le_target:
                    try: color_labels_full = st.session_state.le_target.inverse_transform(y_original_full); color_labels_legend_name = f"{st.session_state.target} (labels)"
                    except Exception: pass # Agar decode na ho paye toh encoded use karo
                pca_fig = px.scatter_3d(x=X_pca_full[:, 0], y=X_pca_full[:, 1], z=X_pca_full[:, 2], color=color_labels_full, title="3D PCA Plot (Full Data)", labels={'color': color_labels_legend_name})
                st.plotly_chart(pca_fig, use_container_width=True)
            except Exception as e_pca: st.warning(f"Could not generate PCA plot: {e_pca}.")

        st.subheader("🤖 Model Training & Comparison")
        if can_proceed_with_training and X_processed_full is not None and y_original_full is not None:
            if st.button("Train & Tune Models", key="automl_train_models_btn_main_panel"):
                st.session_state.train_button_clicked_once = True
                with st.spinner("⏳ Training models... This will take some time... Grab a chai! ☕"):
                    # --- Full Model Training Loop ---
                    try:
                        X_train_p, X_test_p, y_train, y_test = train_test_split(
                            X_processed_full, y_original_full, test_size=0.25, random_state=42,
                            stratify=y_original_full if st.session_state.task_type == "classification" else None
                        )
                        st.session_state.X_train_processed_for_fitting_smote_model = X_train_p
                        st.session_state.X_test_processed_for_predict = X_test_p
                        st.session_state.y_train_encoded_for_fitting_smote_model = y_train
                        st.session_state.y_test_encoded_for_predict = y_test
                    except ValueError as e:
                        if "The least populated class in y has only" in str(e) or "n_splits=" in str(e):
                             st.error(f"💡 Data Issue for Splitting: {str(e)}. Ensure each class has enough samples."); st.stop()
                        else: st.error(f"Splitting Error: {e}"); st.stop()
                    
                    common_cv = 3 # Cross-validation folds
                    # Updated model definitions with more standard parameters
                    classification_models_params = [
                        ('Logistic Regression', LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'), {'C': [0.01, 0.1, 1, 10, 100], 'max_iter': [1000, 2000]}),
                        ('KNN Classifier', KNeighborsClassifier(), {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance'], 'metric': ['minkowski', 'euclidean']}),
                        ('Decision Tree Classifier', DecisionTreeClassifier(random_state=42, class_weight='balanced'), {'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 5]}),
                        ('Random Forest Classifier', RandomForestClassifier(random_state=42, class_weight='balanced'), {'n_estimators': [100, 200], 'max_depth': [5, 10, 20, None], 'min_samples_leaf': [1, 2, 4]}),
                        ('Gradient Boosting Classifier', GradientBoostingClassifier(random_state=42), {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5]}),
                        ('Gaussian Naive Bayes', GaussianNB(), {}),
                        ('SVC', SVC(random_state=42, class_weight='balanced', probability=True), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}) # Added SVC
                    ]
                    regression_models_params = [
                        ('Linear Regression', LinearRegression(), {}),
                        ('Ridge Regression', Ridge(random_state=42, max_iter=2000), {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}),
                        ('Lasso Regression', Lasso(random_state=42, max_iter=2000), {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]}),
                        ('KNN Regressor', KNeighborsRegressor(), {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}),
                        ('Decision Tree Regressor', DecisionTreeRegressor(random_state=42), {'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 5]}),
                        ('Random Forest Regressor', RandomForestRegressor(random_state=42), {'n_estimators': [100, 200], 'max_depth': [5, 10, 20, None], 'min_samples_leaf': [1, 2, 4]}),
                        ('Gradient Boosting Regressor', GradientBoostingRegressor(random_state=42), {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5]}),
                        ('SVR', SVR(), {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'epsilon': [0.1, 0.2]}) # Added SVR
                    ]

                    models_to_run = classification_models_params if st.session_state.task_type == "classification" else regression_models_params
                    temp_results, temp_trained_models = [], []
                    progress_bar, status_text = st.progress(0.0), st.empty()

                    for i, (name, model_instance, params) in enumerate(models_to_run):
                        status_text.info(f"Training {name} ({i+1}/{len(models_to_run)})...")
                        try:
                            train_pipeline_steps = []
                            if st.session_state.task_type == "classification" and st.session_state.use_smote_for_classification and IMBLEARN_AVAILABLE:
                                smote_k_neighbors = 5 
                                class_dist_train = Counter(y_train)
                                if class_dist_train and min(class_dist_train.values()) > smote_k_neighbors :
                                    train_pipeline_steps.append(('smote', SMOTE(random_state=42, k_neighbors=smote_k_neighbors)))
                                elif class_dist_train:
                                    adjusted_k = min(class_dist_train.values()) - 1
                                    if adjusted_k >=1: train_pipeline_steps.append(('smote', SMOTE(random_state=42, k_neighbors=adjusted_k))); st.info(f"SMOTE for {name} using k_neighbors={adjusted_k}.")
                                    else: st.warning(f"Cannot apply SMOTE for {name} (smallest class too small). Training without SMOTE.")
                            
                            train_pipeline_steps.append(('model', model_instance))
                            current_train_pipeline = ImblearnPipeline(train_pipeline_steps) if 'smote' in dict(train_pipeline_steps) else SklearnPipeline(train_pipeline_steps)
                            pipeline_params = {f'model__{k}': v for k, v in params.items()} if params else {}

                            if not pipeline_params: # No tuning parameters for this model (e.g. GaussianNB)
                                final_fitted_train_pipeline = current_train_pipeline.fit(X_train_p, y_train)
                                best_params_display = 'default (no tuning)'
                            else:
                                grid_search = GridSearchCV(current_train_pipeline, pipeline_params, cv=common_cv, 
                                                           scoring='accuracy' if st.session_state.task_type == "classification" else 'r2', 
                                                           n_jobs=-1, error_score='raise') 
                                grid_search.fit(X_train_p, y_train)
                                final_fitted_train_pipeline = grid_search.best_estimator_
                                best_params_display = grid_search.best_params_
                            
                            y_pred_loop = final_fitted_train_pipeline.predict(X_test_p)
                            temp_trained_models.append({'name': name, 'fitted_smote_model_pipeline': final_fitted_train_pipeline, 'params': best_params_display})
                            if st.session_state.task_type == "classification":
                                temp_results.append([name, accuracy_score(y_test, y_pred_loop), f1_score(y_test, y_pred_loop, average='weighted', zero_division=0)])
                            else:
                                temp_results.append([name, r2_score(y_test, y_pred_loop), np.sqrt(mean_squared_error(y_test, y_pred_loop))])
                        except Exception as e_model_train:
                            error_str_model = str(e_model_train)
                            common_error_substrings = ["The least populated class in y has only", "n_splits=", "Expected n_neighbors <=", "is not supported for multi-output"]
                            if any(sub in error_str_model for sub in common_error_substrings):
                                st.warning(f"Skipping {name}: Data issue for CV/SMOTE. Error: {error_str_model[:100]}...")
                            else: st.error(f"Error training {name}: {error_str_model[:150]}...")
                        progress_bar.progress((i + 1) / len(models_to_run))
                    
                    if not temp_trained_models: status_text.error("No models trained successfully. Check data/config and error messages.")
                    else: status_text.success("Training complete! ✅")
                    st.session_state.results, st.session_state.trained_models_for_saving = temp_results, temp_trained_models
                    st.rerun() # Refresh to show results
                    # --- End of Full Model Training Loop ---
        elif not can_proceed_with_training and st.session_state.df is not None:
            st.warning("Model training is disabled. Please resolve data/preprocessing issues.")

        if st.session_state.results:
            st.subheader("🏆 AutoML Model Performance")
            cols_automl = ["Model", "Accuracy", "F1 Score (Weighted)"] if st.session_state.task_type == "classification" else ["Model", "R² Score", "RMSE"]
            results_df_automl = pd.DataFrame(st.session_state.results, columns=cols_automl)
            if not results_df_automl.empty:
                sort_by_col_automl = "Accuracy" if st.session_state.task_type == "classification" else "R² Score"
                results_df_automl = results_df_automl.sort_values(by=sort_by_col_automl, ascending=False).reset_index(drop=True)
                st.dataframe(results_df_automl.style.format({cols_automl[1]: "{:.3f}", cols_automl[2]: "{:.3f}"}))
                
                # Best model details and plots (as in your original app)
                best_model_name_automl = results_df_automl.iloc[0]['Model']
                best_model_entry_automl = next((m for m in st.session_state.trained_models_for_saving if m['name'] == best_model_name_automl), None)
                if best_model_entry_automl:
                    st.subheader(f"🔎 Detailed Look: {best_model_name_automl} (Best AutoML Model)")
                    # ... (Your logic for Confusion Matrix/Predicted vs Actual, Feature Importances/Coefficients plots)
                    # This part remains the same as your full original code.

                st.subheader("💾 Save Trained AutoML Pipeline")
                sel_model_name_save_automl = st.selectbox("Choose AutoML model pipeline to save:", results_df_automl['Model'].tolist(), key="automl_model_save_select")
                model_entry_save_automl = next((m for m in st.session_state.trained_models_for_saving if m['name'] == sel_model_name_save_automl), None)

                if model_entry_save_automl and st.session_state.main_preprocessor_fitted_on_full_X and \
                   st.session_state.original_X_for_pipeline_fitting is not None and \
                   st.session_state.original_y_for_pipeline_fitting is not None:
                    with st.spinner(f"Re-fitting final pipeline for {sel_model_name_save_automl} on ALL data..."):
                        main_preprocessor_final = clone(st.session_state.main_preprocessor_fitted_on_full_X)
                        smote_model_pipeline_config = model_entry_save_automl['fitted_smote_model_pipeline']
                        final_pipeline_steps = [('preprocessor', main_preprocessor_final)]
                        if 'smote' in smote_model_pipeline_config.named_steps: # Check if smote was part of the best pipeline
                            final_pipeline_steps.append(('smote', clone(smote_model_pipeline_config.named_steps['smote'])))
                        final_pipeline_steps.append(('model', clone(smote_model_pipeline_config.named_steps['model'])))
                        full_pipeline_to_save = ImblearnPipeline(final_pipeline_steps) if 'smote' in dict(final_pipeline_steps) else SklearnPipeline(final_pipeline_steps)
                        full_pipeline_to_save.fit(st.session_state.original_X_for_pipeline_fitting, st.session_state.original_y_for_pipeline_fitting)
                    
                    pipeline_bytes = BytesIO(); joblib.dump(full_pipeline_to_save, pipeline_bytes); pipeline_bytes.seek(0)
                    model_filename = f"{sel_model_name_save_automl.replace(' ', '_').lower()}_pipeline.pkl"
                    st.download_button(label=f"⬇️ Download {sel_model_name_save_automl} Pipeline (.pkl)", data=pipeline_bytes, file_name=model_filename, mime="application/octet-stream", key="automl_dl_pkl_btn")
                    st.info(f"💡 **Test your downloaded pipeline!** Scroll down to the **'Model Pipeline Checker'** section on this page to upload '{model_filename}' and test it with new data.")
                    if st.session_state.task_type == "classification" and st.session_state.le_target:
                        le_bytes = BytesIO(); joblib.dump(st.session_state.le_target, le_bytes); le_bytes.seek(0)
                        st.download_button(label="⬇️ Download Target Encoder (.pkl)", data=le_bytes, file_name="target_label_encoder.pkl", mime="application/octet-stream", key="automl_dl_le_btn")
            else: st.markdown("Train models to enable saving.")
    else: 
        st.info("Welcome to the AutoML Workflow! Please upload a CSV file or load a sample dataset from the sidebar to get started.")


# --- Main App Execution ---
def main():
    st.set_page_config(page_title="AutoML Suite 🚀", page_icon="🧪", layout="wide")
    st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)
    init_session_state() # Initialize all session states (will set show_welcome_guide)

    # --- Conditional Rendering for Welcome Guide ---
    if st.session_state.get('show_welcome_guide', True): # Default to showing guide if key somehow missing
        render_welcome_guide() # Yeh function welcome screen dikhayega
    else:
        # --- Main Application UI (as previously structured) ---
        st.sidebar.title("⚙️ Controls & Settings")
        st.sidebar.markdown("Use the options below to manage the AutoML workflow.")
        st.sidebar.markdown("---")

        render_automl_workflow() # AutoML section will render its sidebar controls and main content
        render_model_checker()   # Model Checker section will render its UI elements and logic below AutoML

        # Common Sidebar Footer
        st.sidebar.markdown("---")
        st.sidebar.header("📦 App Info")
        with st.sidebar.expander("Show App Requirements"):
            st.code("pip install streamlit pandas numpy scikit-learn plotly joblib imbalanced-learn")
            st.download_button("⬇️ Download requirements.txt",
                               "streamlit\npandas\nnumpy\nscikit-learn\nplotly\njoblib\nimbalanced-learn",
                               "requirements.txt", key="main_req_dl_btn")
        st.sidebar.markdown("---")
        st.sidebar.info("AutoML Suite: Your ML companion. Always validate results critically.")


if __name__ == "__main__":
    main()