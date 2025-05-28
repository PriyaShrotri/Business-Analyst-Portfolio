🩺 Breast Cancer Analytics Dashboard

A modern, interactive dashboard for the Breast Cancer Wisconsin Diagnostic Dataset built with Streamlit.
This project is designed for business analytics portfolios, showcasing advanced data exploration, visualization, and feature analysis for a real-world healthcare dataset.
🚀 Features

    KPI Summary:
    Instantly see the count and proportion of benign and malignant diagnoses.

    Correlation Heatmap:
    Visualize relationships between the most diagnosis-relevant features.

    Feature Importance:
    See which features drive diagnosis using a Random Forest model.

    Boxplots:
    Compare the distribution of top features by diagnosis group.

    Pairplot:
    Explore interrelationships between the most important features, colored by diagnosis.

    Custom Scatterplot:
    Select any two features for a dynamic scatterplot by diagnosis.

🗂️ Dataset

    Source: Breast Cancer Wisconsin Diagnostic Dataset

    Features: 30 numeric features computed from digitized images of breast mass FNA.

    Target: Diagnosis (malignant or benign).

🛠️ How It Works

    Load Data:
    The dataset is loaded using scikit-learn and converted to a pandas DataFrame.

    KPI Summary:
    The dashboard displays counts and proportions of each diagnosis class.

    Correlation Heatmap:
    Calculates the correlation of all features with the diagnosis, then displays a heatmap of the top 10.

    Feature Importance:
    Trains a Random Forest classifier and displays the ten most important features for diagnosis prediction.

    Boxplots:
    Shows the distribution of the top four features (by importance) for each diagnosis group.

    Pairplot:
    Uses seaborn to plot pairwise relationships among the top features, colored by diagnosis.

    Custom Scatterplot:
    Lets the user select any two features for a scatterplot, colored by diagnosis.
