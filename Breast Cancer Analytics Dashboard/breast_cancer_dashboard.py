import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Breast Cancer Analytics Dashboard", layout="wide")

# Load dataset
@st.cache_data
def load_data():
    cancer = load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['diagnosis'] = pd.Categorical.from_codes(cancer.target, cancer.target_names)
    return df, cancer

df, cancer = load_data()

st.title("Breast Cancer Wisconsin Diagnostic Dataset Dashboard")

# KPI Summary
st.header("KPI Summary")
col1, col2 = st.columns(2)
with col1:
    st.metric("Benign", int((df['diagnosis'] == 'benign').sum()))
with col2:
    st.metric("Malignant", int((df['diagnosis'] == 'malignant').sum()))

st.write("**Proportion of Diagnoses:**")
st.write(df['diagnosis'].value_counts(normalize=True).rename("proportion").to_frame())

# Correlation Heatmap
st.header("Correlation Heatmap")
df_corr = df.copy()
df_corr['diagnosis_num'] = df_corr['diagnosis'].cat.codes
corr = df_corr.corr(numeric_only=True)
top_corr_features = corr['diagnosis_num'].abs().sort_values(ascending=False)[1:11].index

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df_corr[top_corr_features].corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Feature Importance
st.header("Feature Importance (Random Forest)")
X = df[cancer.feature_names]
y = df['diagnosis'].cat.codes
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=cancer.feature_names)
top_importances = importances.sort_values(ascending=False)[:10]

fig2, ax2 = plt.subplots(figsize=(8, 6))
top_importances.plot(kind='barh', ax=ax2)
ax2.set_title('Top 10 Feature Importances (Random Forest)')
ax2.set_xlabel('Importance Score')
ax2.invert_yaxis()
st.pyplot(fig2)

# Boxplots for top features by diagnosis
st.header("Boxplots of Top Features by Diagnosis")
top_features = top_importances.index[:4]
fig3, axes = plt.subplots(2, 2, figsize=(14, 8))
for i, feature in enumerate(top_features):
    sns.boxplot(x='diagnosis', y=feature, data=df, ax=axes[i//2, i%2])
    axes[i//2, i%2].set_title(f'{feature} by Diagnosis')
plt.tight_layout()
st.pyplot(fig3)

# Pairplot for top features
st.header("Pairplot of Top Features by Diagnosis")
st.info("This plot may take a few seconds to render.")
import itertools
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Streamlit can't show seaborn pairplot directly, so we save to buffer
import io
import PIL.Image

pairplot_fig = sns.pairplot(df, vars=top_features, hue='diagnosis', diag_kind='kde', plot_kws={'alpha': 0.6})
buf = io.BytesIO()
pairplot_fig.savefig(buf, format="png", bbox_inches="tight")
st.image(buf)

# Optional: Add feature selectors for scatterplots
st.header("Custom Scatterplot")
feature_x = st.selectbox("Select X-axis feature:", cancer.feature_names, index=0)
feature_y = st.selectbox("Select Y-axis feature:", cancer.feature_names, index=1)

fig4, ax4 = plt.subplots(figsize=(7, 5))
sns.scatterplot(x=feature_x, y=feature_y, hue='diagnosis', data=df, alpha=0.7, ax=ax4)
ax4.set_title(f'{feature_x} vs {feature_y} by Diagnosis')
st.pyplot(fig4)
