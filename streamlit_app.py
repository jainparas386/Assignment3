import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def generate_data(dataset):
    if dataset == 'Dataset 1':
        X, y = make_classification(n_samples=1000, n_classes=2, weights=[
                                   0.9, 0.1], random_state=1)
    elif dataset == 'Dataset 2':
        X, y = make_classification(n_samples=1000, n_classes=2, weights=[
                                   0.8, 0.2], random_state=2)
    else:
        X, y = make_classification(n_samples=1000, n_classes=2, weights=[
                                   0.7, 0.3], random_state=3)
    return X, y


@st.cache_data
def get_data(dataset):
    X, y = generate_data(dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def plot_roc_curve(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    st.write('### ROC Curve')
    st.write('AUC: {:.3f}'.format(roc_auc))
    st.line_chart(pd.DataFrame({'fpr': fpr, 'tpr': tpr}).set_index('fpr'))


def plot_precision_recall_curve(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(
        y_test, y_pred_proba)

    st.write('### Precision-Recall Curve')
    st.line_chart(pd.DataFrame(
        {'precision': precision, 'recall': recall}).set_index('recall'))


def run():
    st.title('Classification Models')

    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Multi-layer Perceptron': MLPClassifier()
    }

    datasets = ['Dataset 1', 'Dataset 2', 'Dataset 3']
    dataset = st.sidebar.selectbox('Select dataset', options=datasets)

    X_train, X_test, y_train, y_test = get_data(dataset)

    model = st.sidebar.selectbox('Select model', options=list(models.keys()))

    clf = models[model]
    clf.fit(X_train, y_train)

    st.write('### Results')
    st.write('Accuracy: {:.3f}'.format(clf.score(X_test, y_test)))

    plot_roc_curve(clf, X_test, y_test)
    plot_precision_recall_curve(clf, X_test, y_test)


if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        st.write('An error occurred:', e)
