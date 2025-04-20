import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Tema padrão com tons de verde
pio.templates.default = "plotly"

st.set_page_config(page_title="HR Analytics Dashboard", layout="wide")

# Carregando os dados
@st.cache_data
def carregar_dados():
    return pd.read_csv('HR_Analytics.csv')

df = carregar_dados()

st.markdown("""
    <style>
    .main {
        background-color: #f6fff8;
    }
    .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3, .stMetricLabel, .stMetricValue {
        color: #2d6a4f;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 HR Analytics Dashboard")

# Preparando colunas de mês e ano
if 'HireDate' in df.columns:
    df['HireDate'] = pd.to_datetime(df['HireDate'])
    df['Ano'] = df['HireDate'].dt.year
    df['Mes'] = df['HireDate'].dt.month_name()
else:
    df['Ano'] = 2023
    df['Mes'] = 'Janeiro'

# Sidebar - Filtros
st.sidebar.header("Filtros")
departamentos = st.sidebar.multiselect("Departamento", options=df['Department'].unique(), default=list(df['Department'].unique()))
atrition = st.sidebar.multiselect("Attrition", options=df['Attrition'].unique(), default=list(df['Attrition'].unique()))
anos = st.sidebar.multiselect("Ano de Contratação", options=sorted(df['Ano'].unique()), default=sorted(df['Ano'].unique()))
meses = st.sidebar.multiselect("Mês de Contratação", options=df['Mes'].unique(), default=list(df['Mes'].unique()))

# Se nenhum filtro for selecionado, usar todos os valores como padrão
departamentos = departamentos if departamentos else list(df['Department'].unique())
atrition = atrition if atrition else list(df['Attrition'].unique())
nanos = anos if anos else list(df['Ano'].unique())
meses = meses if meses else list(df['Mes'].unique())

# Aplicando filtros
df_filtrado = df[(df['Department'].isin(departamentos)) &
                 (df['Attrition'].isin(atrition)) &
                 (df['Ano'].isin(nanos)) &
                 (df['Mes'].isin(meses))]

# Paleta de cores verdes para os gráficos
colors_green = px.colors.sequential.Greens

# Abas de navegação
aba = st.tabs(["📈 Visão Geral", "🚀 Desenvolvimento", "🎯 OKRs", "🧭 Jornada do Colaborador"])

# Aba: Visão Geral
with aba[0]:
    # KPIs principais
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Funcionários", df_filtrado.shape[0])

    with col2:
        taxa_attrition = df_filtrado[df_filtrado['Attrition'] == 'Yes'].shape[0] / df_filtrado.shape[0] * 100 if df_filtrado.shape[0] > 0 else 0
        st.metric("Taxa de Evasão", f"{taxa_attrition:.2f}%")

    with col3:
        media_salario = df_filtrado['MonthlyIncome'].mean() if df_filtrado.shape[0] > 0 else 0
        st.metric("Salário Médio", f"${media_salario:,.2f}")

    st.markdown("---")
    col4, col5 = st.columns(2)
    with col4:
        st.subheader("Distribuição de Idade")
        fig_idade = px.histogram(df_filtrado, x='Age', nbins=20, color='Attrition', barmode='overlay', title='Distribuição de Idade', color_discrete_sequence=['#2d6a4f', '#74c69d'])
        st.plotly_chart(fig_idade, use_container_width=True)

    with col5:
        st.subheader("Salário vs Satisfação no Trabalho")
        fig_salario = px.scatter(df_filtrado, x='MonthlyIncome', y='JobSatisfaction', color='Attrition', hover_data=['JobRole'], title='Relação entre Salário e Satisfação', color_discrete_sequence=['#2d6a4f', '#74c69d'])
        st.plotly_chart(fig_salario, use_container_width=True)

    col6, col7 = st.columns(2)
    with col6:
        st.subheader("Funcionários por Departamento")
        fig_departamento = px.histogram(df_filtrado, x='Department', color='Attrition', barmode='group', title='Distribuição por Departamento', color_discrete_sequence=['#2d6a4f', '#74c69d'])
        st.plotly_chart(fig_departamento, use_container_width=True)

    with col7:
        st.subheader("Distribuição por Gênero")
        fig_genero = px.histogram(df_filtrado, x='Gender', color='Attrition', barmode='group', title='Distribuição por Gênero', color_discrete_sequence=['#2d6a4f', '#74c69d'])
        st.plotly_chart(fig_genero, use_container_width=True)

    st.subheader("Idade média por Gênero")
    fig_idade_genero = px.box(df_filtrado, x='Gender', y='Age', color='Gender', title='Idade por Gênero', color_discrete_sequence=['#2d6a4f', '#95d5b2'])
    st.plotly_chart(fig_idade_genero, use_container_width=True)

    st.markdown("---")
    st.subheader("Tabela de Dados")
    st.dataframe(df_filtrado, use_container_width=True)

# Aba: Desenvolvimento
with aba[1]:
    st.header("🚀 Desenvolvimento")
    st.subheader("Satisfação no Trabalho por Educação")
    fig_satisf_edu = px.box(df_filtrado, x='EducationField', y='JobSatisfaction', color='EducationField', title='Satisfação no Trabalho por Área de Educação', color_discrete_sequence=colors_green)
    st.plotly_chart(fig_satisf_edu, use_container_width=True)

    st.subheader("Anos na Empresa vs Performance")
    if 'PerformanceRating' in df_filtrado.columns:
        fig_perf = px.scatter(df_filtrado, x='YearsAtCompany', y='PerformanceRating', color='Attrition', title='Performance vs Tempo de Empresa', color_discrete_sequence=colors_green)
        st.plotly_chart(fig_perf, use_container_width=True)

# Aba: OKRs
with aba[2]:
    st.header("🎯 OKRs")
    st.subheader("Horas de Treinamento vs Performance")
    if 'PerformanceRating' in df_filtrado.columns:
        fig_treino = px.scatter(df_filtrado, x='TrainingTimesLastYear', y='PerformanceRating', color='Attrition', title='Treinamento vs Performance', color_discrete_sequence=colors_green)
        st.plotly_chart(fig_treino, use_container_width=True)

    st.subheader("Ambição de Crescimento (JobLevel vs Satisfação)")
    fig_growth = px.box(df_filtrado, x='JobLevel', y='JobSatisfaction', color='Attrition', title='Crescimento e Satisfação', color_discrete_sequence=colors_green)
    st.plotly_chart(fig_growth, use_container_width=True)

# Aba: Jornada do Colaborador
with aba[3]:
    st.header("🧭 Jornada do Colaborador")
    st.subheader("Tempo de Empresa vs Número de Empresas Anteriores")
    fig_jornada = px.scatter(df_filtrado, x='YearsAtCompany', y='NumCompaniesWorked', color='Attrition', title='Jornada Profissional', color_discrete_sequence=colors_green)
    st.plotly_chart(fig_jornada, use_container_width=True)

    st.subheader("Previsão de Evasão de Colaborador")
    df_modelo = df.copy()
    colunas_utilizadas = ['Age', 'DistanceFromHome', 'MonthlyIncome', 'YearsAtCompany', 'JobSatisfaction', 'EnvironmentSatisfaction']

    df_modelo = df_modelo.dropna(subset=colunas_utilizadas + ['Attrition'])
    df_modelo = df_modelo[colunas_utilizadas + ['Attrition']]

    le = LabelEncoder()
    df_modelo['Attrition'] = le.fit_transform(df_modelo['Attrition'])

    X = df_modelo[colunas_utilizadas]
    y = df_modelo['Attrition']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    modelo = RandomForestClassifier()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    acuracia = accuracy_score(y_test, y_pred)
    st.metric("Acurácia do Modelo", f"{acuracia:.2%}")
    st.text("Relatório de Classificação")
    st.text(classification_report(y_test, y_pred))
