import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_moons, make_circles, make_classification
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import plotly as px
import matplotlib.pyplot as plt
from sklearn import metrics 

def generate_data(dataset):
    if dataset == 'Dataset 1':
        X, y = make_classification(n_samples=1000, n_features=2,n_redundant=0,n_classes=2, weights=[0.9, 0.1], random_state=1)
    elif dataset == 'Dataset 2':
        X, y = make_classification(n_samples=1000, n_features=2,n_redundant=0,n_classes=2, weights=[0.1, 0.9], random_state=2)
    elif dataset == 'Dataset 3':
        X, y = make_classification(n_samples=1000, n_features=2,n_redundant=0,n_classes=2, class_sep=1, weights=[0.5, 0.5], random_state=3)
    elif dataset == 'Dataset 4':
        X, y = make_classification(n_samples=1000, n_features=2,n_classes=2, n_redundant=0,class_sep=0.1, weights=[0.5, 0.5], random_state=3)
    
    return X, y


@st.cache_data
def get_data(dataset):
    X, y = generate_data(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    return X, y, X_train, X_test, y_train, y_test


def plot_roc_curve(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    st.write('### ROC Curve')
    st.write('ROC(Receiver Operating Characteristics curve) is a graph of TPR(True Positive rate) vs FPR(False Positive Rate) at different classification thresholds')
    st.write('AUC(Area Under the Curve): {:.3f}'.format(roc_auc))
    # st.line_chart(pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'x': np.linspace(0, 1, 10)}).set_index('fpr').style.set_caption('fpr vs tpr'))
    st.line_chart(pd.DataFrame({'fpr': fpr, 'tpr': tpr}).set_index('fpr'))
    st.write('x-axis: fpr, y-axis: tpr')


def plot_precision_recall_curve(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:,1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    # pr_auc = auc(precision, recall)

    
    st.write('### Precision-Recall Curve')
    st.write('Precision vs Recall at all classification thresholds')
    # st.write('AUC: {:.3f}'.format(pr_auc))
    st.line_chart(pd.DataFrame(
        {'precision': precision, 'recall': recall}).set_index('recall'))
    # st.columns
    st.write('x-axis: recall, y-axis: precision')
    # st.line_chart(pd.DataFrame({'precision': precision, 'recall': recall, 'x': np.linspace(0, 1, 10)}).set_index('recall').style.set_caption('recall vs precision'))
    

def plot_data(X, y):
    st.write('### Dataset')
    df = pd.DataFrame({'X0': X[:,0], 'X1': X})

def run():
    st.set_page_config(page_title='Classification Models', layout='wide')
    
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Multi-layer Perceptron': MLPClassifier(),
        'KNN': KNeighborsClassifier()
    }
    
    datasets = ['Dataset 1', 'Dataset 2', 'Dataset 3', 'Dataset 4']
    
    st.sidebar.title('Classification Models')
    
    dataset = st.sidebar.selectbox('Select dataset', options=datasets)
    if dataset == 'Dataset 1':
        st.sidebar.write('### Imbalanced Dataset')
    if dataset == 'Dataset 2':
        st.sidebar.write('### Imbalanced Dataset')
    if dataset == 'Dataset 3':
        st.sidebar.write('### Easy to classify(class_sep = 1')
    if dataset == 'Dataset 4':
        st.sidebar.write('### Hard to classify(class_sep = 0.1)')
    # X, y = generate_data(**datasets[dataset])
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X, y, X_train, X_test, y_train, y_test = get_data(dataset)
    model = st.sidebar.selectbox('Select model', options=list(models.keys()))
    

    clf = models[model]
    if model == 'Decision Tree':
        max_depth = st.sidebar.slider('Max Depth', 1, 10, 1)
        clf.set_params(max_depth = max_depth)

    if model == 'Random Forest':
        n_estimators = st.sidebar.slider('n_estimators', 20, 210, 20)
        max_depth = st.sidebar.slider('Max Depth', 1, 10, 1)
        clf.set_params(max_depth = max_depth, n_estimators= n_estimators)

    if model == 'Multi-layer Perceptron':
        alpha_L2_regu = st.sidebar.slider('alpha', 0.00, 100.00, 0.5)
        learning_rate_init = st.sidebar.slider('learning_rate_init', 0.00, 0.1, 0.02)
        max_iter = st.sidebar.slider('max_iter', 20, 210, 20)
        batch_size = st.sidebar.slider('batch_size', 10, 110, 5)
        # don't know how to add multiple layers
        # hidden_layer_sizes = st.sidebar.slider('hidden_layer_sizes', 20, 210, 20)
        if learning_rate_init == 0.00:
            learning_rate_init=0.001 #default
        
        if alpha_L2_regu == 0.00:
            alpha_L2_regu = 0.0001 #default
        clf.set_params(max_iter=max_iter, alpha= alpha_L2_regu, learning_rate_init=learning_rate_init, batch_size = batch_size)

    if model == 'KNN':
        n_neighbors = st.sidebar.slider('n_neighbors', min_value = 1, max_value = 50, step = 5)
        # p is for type of distance used
        p = st.sidebar.slider('power_parameters', 1, 5, 1)
        clf.set_params(n_neighbors = n_neighbors, p = p)
        
    clf.fit(X_train, y_train)

    st.title('Classification Results')
    st.write('### This app show the ROC curve and Precision-Recall curve for different classification datasets using different classification models')

    col1, col2 = st.columns([1, 1])

    with col1:
        st.write('### Dataset')
        # st.pyplot(pd.DataFrame({'x': X[:,0], 'y': X[:,1], 'label': y}).set_index('x'))
        fig, ax = plt.subplots()
        ax.scatter(X[:,0], X[:,1], label= 'Classification_Data', c=y, cmap='bwr')#, color = X[:,1])
        ax.set_xlabel('Feature-1')
        ax.set_ylabel('Feature-2')
        fig.legend()
        # fig.set_size_inches(8,4)
        st.pyplot(fig)
        

        st.write('### Training Dataset')
        # st.pyplot(pd.DataFrame({'x': X[:,0], 'y': X[:,1], 'label': y}).set_index('x'))
        fig, ax = plt.subplots()
        ax.scatter(X_train[:,0], X_train[:,1], label= 'Training_Data', c=y_train, cmap='bwr')#, color = X[:,1])
        ax.set_xlabel('Feature-1')
        ax.set_ylabel('Feature-2')
        fig.legend()
        st.pyplot(fig)

        st.write('### Testing Dataset')
        # st.pyplot(pd.DataFrame({'x': X[:,0], 'y': X[:,1], 'label': y}).set_index('x'))
        fig, ax = plt.subplots()
        ax.scatter(X_test[:,0], X_test[:,1], label= 'Testing_Data', c=y_test, cmap='bwr')#, color = X[:,1])
        ax.set_xlabel('Feature-1')
        ax.set_ylabel('Feature-2')
        fig.legend()
        st.pyplot(fig)
        
        # st.plotly_chart(px.line(pd.DataFrame({'x': [0, 1], 'y': [0, 1]}), x='x', y='y', line_dash='dash'), use_container_width=True)

    with col2:
        st.write('### Model Performance')
        # y_pred = clf.predict(X_test)

        # st.write('Training Accuracy:{:.3f}'.format(metrics.accuracy_score(y_test, y_pred)))
        st.write('Train Accuracy:{:.3f}'.format(clf.score(X_train, y_train)))
        st.write('Test Accuracy: {:.3f}'.format(clf.score(X_test, y_test)))

        plot_roc_curve(clf, X_test, y_test)
        plot_precision_recall_curve(clf, X_test, y_test)
        # st.plotly_chart(px.line(pd.DataFrame({'x': [0, 1], 'y': [0, 1]}), x='x', y='y', line_dash='dash'), use_container_width=True)

if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        st.write('An error occurred:', e)