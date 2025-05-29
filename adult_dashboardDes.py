import dash
from dash import dcc, html, dash_table, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, 
    confusion_matrix, roc_curve, auc, precision_score, recall_score
)
from ucimlrepo import fetch_ucirepo
import pickle
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# ===== CONFIGURACIÓN Y CACHE OPTIMIZADO =====
CACHE_DIR = "model_cache"
MODEL_CACHE_FILE = os.path.join(CACHE_DIR, "all_trained_models.pkl")
DATA_CACHE_FILE = os.path.join(CACHE_DIR, "processed_data.pkl")
RESULTS_CACHE_FILE = os.path.join(CACHE_DIR, "model_results.pkl")

def ensure_cache_dir():
    """Crear directorio de cache si no existe"""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

def load_and_preprocess_data():
    """Carga y preprocesa el dataset Adult con cache optimizado"""
    ensure_cache_dir()
    
    if os.path.exists(DATA_CACHE_FILE):
        print("📁 Cargando datos desde cache...")
        with open(DATA_CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    
    print("🔄 Procesando datos por primera vez...")
    start_time = time.time()
    
    # Cargar dataset
    adult = fetch_ucirepo(id=2)
    X = adult.data.features
    y = adult.data.targets
    
    # Concatenar features y target
    df = pd.concat([X, y], axis=1)
    
    # Optimización: Procesamiento vectorizado
    # Limpiar datos
    df = df[df["income"].notnull()].reset_index(drop=True)
    df = df.drop_duplicates().reset_index(drop=True)
    
    # Normalizar strings de forma vectorizada
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df[col] = df[col].str.strip().str.lower()
    
    # Filtrar solo clases válidas
    valores_esperados = ['<=50k', '>50k']
    df = df[df["income"].isin(valores_esperados)].reset_index(drop=True)
    
    # Eliminar columna redundante si existe
    if 'education' in df.columns:
        df = df.drop(columns=["education"])
    
    # Codificar variable objetivo
    le = LabelEncoder()
    df["income"] = le.fit_transform(df["income"])
    
    # Guardar en cache
    cache_data = (df, le)
    with open(DATA_CACHE_FILE, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"✅ Datos procesados en {time.time() - start_time:.2f} segundos")
    return df, le

def split_data_optimized(df):
    """División optimizada de datos con estratificación"""
    X_pre = df.drop(columns=["income"])
    y_pre = df["income"]
    
    # División única con estratificación
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_pre, y_pre, test_size=0.30, stratify=y_pre, random_state=42
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_optimized_preprocessor(X_train):
    """Crea preprocessor optimizado con menos transformaciones"""
    numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Preprocessor optimizado
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=20))
            ]), categorical_cols)
        ],
        remainder='drop',
        n_jobs=-1
    )
    
    return preprocessor

def train_single_model(model_config, X_train, X_val, y_train, y_val, preprocessor):
    """Entrena un solo modelo (para paralelización)"""
    nombre, modelo = model_config
    
    try:
        print(f"  🔄 Entrenando {nombre}...")
        start_time = time.time()
        
        # Crear pipeline
        pipeline = Pipeline(steps=[
            ("preprocessing", preprocessor),
            ("classifier", modelo)
        ])
        
        # Entrenar
        pipeline.fit(X_train, y_train)
        
        # Predicciones
        y_pred = pipeline.predict(X_val)
        y_proba = None
        
        # Obtener probabilidades si el modelo las soporta
        if hasattr(pipeline.named_steps['classifier'], "predict_proba"):
            try:
                y_proba = pipeline.predict_proba(X_val)[:, 1]
            except:
                y_proba = None
        
        # Métricas
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro")
        precision = precision_score(y_val, y_pred, average="macro")
        recall = recall_score(y_val, y_pred, average="macro")
        cm = confusion_matrix(y_val, y_pred)
        
        training_time = time.time() - start_time
        print(f"  ✅ {nombre} completado en {training_time:.2f}s - F1: {f1:.4f}")
        
        return nombre, {
            'accuracy': acc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_proba': y_proba,
            'training_time': training_time,
            'pipeline': pipeline
        }
        
    except Exception as e:
        print(f"  ❌ Error entrenando {nombre}: {str(e)}")
        return nombre, None

def train_all_models_optimized(X_train, X_val, y_train, y_val):
    """Entrena todos los modelos con optimizaciones y paralelización"""
    ensure_cache_dir()
    
    if os.path.exists(MODEL_CACHE_FILE):
        print("🤖 Cargando modelos entrenados desde cache...")
        with open(MODEL_CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    
    print("🚀 Entrenando todos los modelos con optimizaciones...")
    start_time = time.time()
    
    # Crear preprocessor una sola vez
    preprocessor = create_optimized_preprocessor(X_train)
    
    # Modelos optimizados con mejores hiperparámetros
    modelos = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, n_jobs=-1, solver='liblinear'
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=15, min_samples_split=10,
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=8, learning_rate=0.1,
            random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=7, algorithm='ball_tree', n_jobs=-1
        ),
        "Naive Bayes": GaussianNB(),
        "SVM Linear": SVC(
            kernel='linear', probability=True, random_state=42, max_iter=1000
        ),
        "SVM RBF": SVC(
            kernel='rbf', probability=True, random_state=42, max_iter=1000,
            gamma='scale', C=1.0
        ),
        "MLP Classifier": MLPClassifier(
            hidden_layer_sizes=(100, 50), max_iter=500, random_state=42,
            alpha=0.01, learning_rate='adaptive'
        )
    }
    
    resultados = {}
    
    # Entrenamiento secuencial optimizado
    for nombre, modelo in modelos.items():
        resultado = train_single_model(
            (nombre, modelo), X_train, X_val, y_train, y_val, preprocessor
        )
        if resultado[1] is not None:
            resultados[resultado[0]] = resultado[1]
    
    total_time = time.time() - start_time
    print(f"🎉 Todos los modelos entrenados en {total_time:.2f} segundos")
    
    # Guardar en cache
    with open(MODEL_CACHE_FILE, 'wb') as f:
        pickle.dump(resultados, f)
    
    return resultados

def calculate_additional_metrics(results):
    """Calcula métricas adicionales para análisis"""
    for model_name, model_result in results.items():
        cm = model_result['confusion_matrix']
        
        # Calcular métricas adicionales desde la matriz de confusión
        tn, fp, fn, tp = cm.ravel()
        
        model_result.update({
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0
        })
    
    return results

def create_roc_curve(results, model_name, y_true):
    """Crea curva ROC si las probabilidades están disponibles"""
    if results['y_proba'] is None:
        return go.Figure().add_annotation(
            text="ROC Curve not available for this model",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        ).update_layout(title=f"ROC Curve - {model_name}")
    
    fpr, tpr, _ = roc_curve(y_true, results['y_proba'])
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        line=dict(color='#e74c3c', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='#95a5a6', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f'ROC Curve - {model_name}',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        height=400
    )
    
    return fig

# ===== INICIALIZACIÓN DE DATOS =====
print("🔄 Inicializando Dashboard Optimizado...")
df, le = load_and_preprocess_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data_optimized(df)
model_results = train_all_models_optimized(X_train, X_val, y_train, y_val)
model_results = calculate_additional_metrics(model_results)

# Estadísticas generales
total_models = len(model_results)
best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['f1_score'])
avg_accuracy = np.mean([r['accuracy'] for r in model_results.values()])

print(f"✅ Dashboard listo: {total_models} modelos entrenados")
print(f"🏆 Mejor modelo: {best_model_name} (F1: {model_results[best_model_name]['f1_score']:.4f})")

# ===== CONFIGURACIÓN DASH =====
app = dash.Dash(__name__)
server = app.server  # Para deployment en Heroku/Render
app.title = "ML Models Comparison - Adult Dataset"

# Layout principal
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🤖 Machine Learning Models Comparison", 
                style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': '5px'}),
        html.P(f"Adult Dataset Analysis - {total_models} Models Trained",
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '16px'}),
        html.P(f"Best Model: {best_model_name} | Avg Accuracy: {avg_accuracy:.3f}",
               style={'textAlign': 'center', 'color': '#27ae60', 'fontSize': '14px', 'fontWeight': 'bold'})
    ], style={'backgroundColor': '#ecf0f1', 'padding': '15px', 'marginBottom': '20px', 'borderRadius': '8px'}),
    
    # Tabs principales
    dcc.Tabs(id='main-tabs', value='tab-overview', 
             style={'height': '44px'}, children=[
        dcc.Tab(label='📊 Overview', value='tab-overview'),
        dcc.Tab(label='🏆 Model Comparison', value='tab-comparison'),
        dcc.Tab(label='📈 Performance Analysis', value='tab-performance'),
        dcc.Tab(label='🔍 Detailed Metrics', value='tab-detailed')
    ]),
    
    html.Div(id='tabs-content', style={'marginTop': '20px'})
], style={'margin': '0 auto', 'maxWidth': '1400px', 'padding': '20px'})

@app.callback(Output('tabs-content', 'children'),
              Input('main-tabs', 'value'))
def render_content(tab):
    if tab == 'tab-overview':
        return html.Div([
            # Estadísticas generales
            html.Div([
                html.Div([
                    html.H4("📊 Dataset Size", style={'textAlign': 'center', 'color': '#3498db'}),
                    html.P(f"{df.shape[0]:,}", style={'fontSize': '28px', 'fontWeight': 'bold', 'textAlign': 'center', 'margin': '0'}),
                    html.P("Records", style={'textAlign': 'center', 'color': '#7f8c8d'})
                ], className='stat-card'),
                
                html.Div([
                    html.H4("🔢 Features", style={'textAlign': 'center', 'color': '#e74c3c'}),
                    html.P(f"{df.shape[1]-1}", style={'fontSize': '28px', 'fontWeight': 'bold', 'textAlign': 'center', 'margin': '0'}),
                    html.P("Variables", style={'textAlign': 'center', 'color': '#7f8c8d'})
                ], className='stat-card'),
                
                html.Div([
                    html.H4("🤖 Models", style={'textAlign': 'center', 'color': '#27ae60'}),
                    html.P(f"{total_models}", style={'fontSize': '28px', 'fontWeight': 'bold', 'textAlign': 'center', 'margin': '0'}),
                    html.P("Algorithms", style={'textAlign': 'center', 'color': '#7f8c8d'})
                ], className='stat-card'),
                
                html.Div([
                    html.H4("🏆 Best F1", style={'textAlign': 'center', 'color': '#f39c12'}),
                    html.P(f"{model_results[best_model_name]['f1_score']:.4f}", 
                           style={'fontSize': '28px', 'fontWeight': 'bold', 'textAlign': 'center', 'margin': '0'}),
                    html.P(best_model_name[:15], style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '12px'})
                ], className='stat-card')
            ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '30px'}),
            
            # Dataset distribution
            html.Div([
                html.Div([
                    dcc.Graph(
                        figure=px.pie(
                            values=df['income'].value_counts().values,
                            names=['≤ $50K', '> $50K'],
                            title="Income Distribution",
                            color_discrete_sequence=['#e74c3c', '#27ae60']
                        ).update_traces(textposition='inside', textinfo='percent+label')
                    )
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                html.Div([
                    dcc.Graph(
                        figure=go.Figure(data=[
                            go.Bar(x=['Training', 'Validation', 'Test'], 
                                   y=[len(X_train), len(X_val), len(X_test)],
                                   text=[f'{len(X_train)/len(df)*100:.1f}%', 
                                         f'{len(X_val)/len(df)*100:.1f}%', 
                                         f'{len(X_test)/len(df)*100:.1f}%'],
                                   textposition='auto',
                                   marker_color=['#3498db', '#e74c3c', '#27ae60'])
                        ]).update_layout(
                            title="Data Split Distribution",
                            xaxis_title="Dataset",
                            yaxis_title="Number of Records"
                        )
                    )
                ], style={'width': '50%', 'display': 'inline-block'})
            ])
        ])
    
    elif tab == 'tab-comparison':
        return html.Div([
            # Tabla de comparación completa
            html.Div([
                html.H3("📋 Complete Model Comparison", style={'color': '#2c3e50'}),
                dash_table.DataTable(
                    data=[
                        {
                            'Model': modelo,
                            'Accuracy': f"{resultados['accuracy']:.4f}",
                            'F1-Score': f"{resultados['f1_score']:.4f}",
                            'Precision': f"{resultados['precision']:.4f}",
                            'Recall': f"{resultados['recall']:.4f}",
                            'Training Time (s)': f"{resultados['training_time']:.2f}",
                            'Rank': i+1
                        }
                        for i, (modelo, resultados) in enumerate(
                            sorted(model_results.items(), key=lambda x: x[1]['f1_score'], reverse=True)
                        )
                    ],
                    columns=[
                        {'name': 'Rank', 'id': 'Rank', 'type': 'numeric'},
                        {'name': 'Model', 'id': 'Model'},
                        {'name': 'Accuracy', 'id': 'Accuracy', 'type': 'numeric'},
                        {'name': 'F1-Score', 'id': 'F1-Score', 'type': 'numeric'},
                        {'name': 'Precision', 'id': 'Precision', 'type': 'numeric'},
                        {'name': 'Recall', 'id': 'Recall', 'type': 'numeric'},
                        {'name': 'Training Time (s)', 'id': 'Training Time (s)', 'type': 'numeric'}
                    ],
                    style_cell={'textAlign': 'center', 'padding': '10px'},
                    style_header={'backgroundColor': '#34495e', 'color': 'white', 'fontWeight': 'bold'},
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{Rank} = 1'},
                            'backgroundColor': '#d5f4e6',
                            'color': 'black',
                            'fontWeight': 'bold'
                        }
                    ],
                    sort_action="native",
                    page_size=10
                )
            ], style={'marginBottom': '30px'}),
            
            # Gráfico de barras comparativo
            html.Div([
                dcc.Graph(
                    figure=go.Figure(data=[
                        go.Bar(name='Accuracy', 
                               x=list(model_results.keys()), 
                               y=[res['accuracy'] for res in model_results.values()],
                               marker_color='#3498db',
                               yaxis='y1'),
                        go.Bar(name='F1-Score', 
                               x=list(model_results.keys()), 
                               y=[res['f1_score'] for res in model_results.values()],
                               marker_color='#e74c3c',
                               yaxis='y1')
                    ]).update_layout(
                        title="Model Performance Comparison",
                        xaxis_title="Models",
                        yaxis_title="Score",
                        barmode='group',
                        xaxis_tickangle=-45,
                        height=500
                    )
                )
            ])
        ])
    
    elif tab == 'tab-performance':
        return html.Div([
            # Selector de modelo para análisis detallado
            html.Div([
                html.H3("🔍 Detailed Model Analysis", style={'color': '#8e44ad'}),
                html.P("Select a model to view detailed performance metrics:"),
                dcc.Dropdown(
                    id='performance-model-dropdown',
                    options=[{'label': f"{i+1}. {modelo}", 'value': modelo} 
                            for i, modelo in enumerate(sorted(model_results.keys(), 
                                                             key=lambda x: model_results[x]['f1_score'], reverse=True))],
                    value=best_model_name,
                    style={'marginBottom': '20px'}
                )
            ]),
            
            html.Div(id='performance-analysis-content')
        ])
    
    elif tab == 'tab-detailed':
        return html.Div([
            html.H3("📊 Detailed Metrics Analysis", style={'color': '#2c3e50'}),
            
            # Heatmap de todas las métricas
            html.Div([
                dcc.Graph(
                    figure=px.imshow(
                        pd.DataFrame({
                            model: [results['accuracy'], results['f1_score'], 
                                   results['precision'], results['recall']]
                            for model, results in model_results.items()
                        }, index=['Accuracy', 'F1-Score', 'Precision', 'Recall']).T,
                        text_auto=True,
                        aspect="auto",
                        title="All Models - All Metrics Heatmap",
                        color_continuous_scale='Viridis',
                        height=600
                    )
                )
            ], style={'marginBottom': '30px'}),
            
            # Tabla detallada con matriz de confusión
            html.Div([
                html.H4("Confusion Matrix Summary"),
                dash_table.DataTable(
                    data=[
                        {
                            'Model': modelo,
                            'True Positives': resultados.get('tp', 'N/A'),
                            'True Negatives': resultados.get('tn', 'N/A'),
                            'False Positives': resultados.get('fp', 'N/A'),
                            'False Negatives': resultados.get('fn', 'N/A'),
                            'Sensitivity': f"{resultados.get('sensitivity', 0):.4f}",
                            'Specificity': f"{resultados.get('specificity', 0):.4f}"
                        }
                        for modelo, resultados in model_results.items()
                    ],
                    columns=[
                        {'name': 'Model', 'id': 'Model'},
                        {'name': 'TP', 'id': 'True Positives'},
                        {'name': 'TN', 'id': 'True Negatives'},
                        {'name': 'FP', 'id': 'False Positives'},
                        {'name': 'FN', 'id': 'False Negatives'},
                        {'name': 'Sensitivity', 'id': 'Sensitivity'},
                        {'name': 'Specificity', 'id': 'Specificity'}
                    ],
                    style_cell={'textAlign': 'center', 'padding': '8px'},
                    style_header={'backgroundColor': '#2c3e50', 'color': 'white'},
                    sort_action="native"
                )
            ])
        ])

@app.callback(
    Output('performance-analysis-content', 'children'),
    Input('performance-model-dropdown', 'value')
)
def update_performance_analysis(selected_model):
    if not selected_model or selected_model not in model_results:
        return html.Div("Please select a valid model")
    
    results = model_results[selected_model]
    
    return html.Div([
        # Métricas principales
        html.Div([
            html.Div([
                html.H4("Accuracy", style={'textAlign': 'center', 'color': '#3498db'}),
                html.P(f"{results['accuracy']:.4f}", 
                       style={'fontSize': '24px', 'fontWeight': 'bold', 'textAlign': 'center'})
            ], className='metric-card'),
            
            html.Div([
                html.H4("F1-Score", style={'textAlign': 'center', 'color': '#e74c3c'}),
                html.P(f"{results['f1_score']:.4f}", 
                       style={'fontSize': '24px', 'fontWeight': 'bold', 'textAlign': 'center'})
            ], className='metric-card'),
            
            html.Div([
                html.H4("Precision", style={'textAlign': 'center', 'color': '#27ae60'}),
                html.P(f"{results['precision']:.4f}", 
                       style={'fontSize': '24px', 'fontWeight': 'bold', 'textAlign': 'center'})
            ], className='metric-card'),
            
            html.Div([
                html.H4("Recall", style={'textAlign': 'center', 'color': '#f39c12'}),
                html.P(f"{results['recall']:.4f}", 
                       style={'fontSize': '24px', 'fontWeight': 'bold', 'textAlign': 'center'})
            ], className='metric-card')
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '30px'}),
        
        # Matriz de confusión y curva ROC
        html.Div([
            html.Div([
                dcc.Graph(
                    figure=px.imshow(
                        results['confusion_matrix'],
                        text_auto=True,
                        aspect="auto",
                        title=f"Confusion Matrix - {selected_model}",
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['≤$50K', '>$50K'],
                        y=['≤$50K', '>$50K'],
                        color_continuous_scale='Blues'
                    )
                )
            ], style={'width': '50%', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Graph(
                    figure=create_roc_curve(results, selected_model, y_val)
                )
            ], style={'width': '50%', 'display': 'inline-block'})
        ])
    ])

# CSS personalizado
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .stat-card, .metric-card {
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin: 10px;
                flex: 1;
                min-width: 150px;
            }
            .stat-card:hover, .metric-card:hover {
                transform: translateY(-2px);
                transition: transform 0.2s ease;
                box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ===== EJECUCIÓN DE LA APLICACIÓN =====
if __name__ == '__main__':
    print("🚀 Iniciando servidor Dash...")
    print("📍 Accede a: http://127.0.0.1:8050")
    print("🔄 Para detener: Ctrl+C")
    
    # Configuración del servidor
    app.run(
        debug=False,           # Modo debug para desarrollo
        host='0.0.0.0',      # Permite conexiones externas
        port=int(os.environ.get('PORT', 8050)),
        dev_tools_hot_reload=True,  # Recarga automática
        dev_tools_ui=True,   # Herramientas de desarrollo
        threaded=True        # Soporte para múltiples usuarios
    )