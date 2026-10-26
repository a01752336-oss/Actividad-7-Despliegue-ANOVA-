#Creamos el archivo de la APP en el interprete principal (Phyton)
#####################################################
#Importamos librerias
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
######################################################
#Definimos la instancia
@st.cache_resource
######################################################
#Creamos la función de carga de datos
def load_data():
   #Lectura de los archivos csv
   df=pd.read_csv("Existencias_04_09_2026.csv")
   clasificacion=pd.read_csv("Clasificación_Articulos.csv")

   #Quitamos espacios en la marca
   df['marca'] = df['marca'].str.strip()

   #Top 6 marcas y el resto como 'Otras'
   top_marcas = df['marca'].value_counts().head(6).index.tolist()
   df['marca_grupo'] = np.where(df['marca'].isin(top_marcas), df['marca'], 'Otras')

   #Transformacion logaritmica
   df['log_existencias'] = np.log1p(df['existencias'])

   #Flujo: Digital o Analogo
   flujo = clasificacion[clasificacion['categoria'].isin(['Digital', 'Analogo'])]
   flujo = flujo.drop_duplicates('articulo_id')[['articulo_id', 'categoria']]
   flujo = flujo.rename(columns={'categoria': 'flujo'})

   #Tipo: Consumible, Equipo, Accesorio, Refaccion o Software
   tipo = clasificacion[clasificacion['categoria'].isin(['Consumible', 'Equipo', 'Accesorio', 'Refaccion', 'Software'])]
   tipo = tipo.drop_duplicates('articulo_id')[['articulo_id', 'categoria']]
   tipo = tipo.rename(columns={'categoria': 'tipo'})

   #Left join: conservamos todos los articulos de existencias
   df = pd.merge(df, flujo, on='articulo_id', how='left')
   df = pd.merge(df, tipo, on='articulo_id', how='left')

   #Consumible contra Equipo y refacciones
   df['tipo_grupo'] = np.where(df['tipo'] == 'Consumible', 'Consumible', 'Equipo y refacciones')
   df.loc[df['tipo'].isnull(), 'tipo_grupo'] = np.nan

   #Topamos los outliers con IQR
   y = df['log_existencias']
   percentile25 = y.quantile(0.25)
   percentile75 = y.quantile(0.75)
   iqr = percentile75 - percentile25
   Limite_Superior_iqr = percentile75 + 1.5 * iqr
   Limite_Inferior_iqr = percentile25 - 1.5 * iqr
   df['log_existencias'] = df['log_existencias'].clip(lower=Limite_Inferior_iqr, upper=Limite_Superior_iqr)

   #Transformacion logaritmica del valor del inventario
   df['log_valor_inventario'] = np.log1p(df['valor_total_inventario'])

   #Topamos los outliers del valor con IQR
   y = df['log_valor_inventario']
   percentile25 = y.quantile(0.25)
   percentile75 = y.quantile(0.75)
   iqr = percentile75 - percentile25
   Limite_Superior_iqr = percentile75 + 1.5 * iqr
   Limite_Inferior_iqr = percentile25 - 1.5 * iqr
   df['log_valor_inventario'] = df['log_valor_inventario'].clip(lower=Limite_Inferior_iqr, upper=Limite_Superior_iqr)
   return df
###############################################################################
#Cargo los datos obtenidos de la función "load_data"
df = load_data()
###############################################################################
#CREACIÓN DEL DASHBOARD
##############################################################################
#Generamos los encabezados para la barra lateral (sidebar)
st.sidebar.title("CLV - EXISTENCIAS")

#Widget 1: Selectbox
#Menu desplegable de opciones de la página
View= st.sidebar.selectbox(label= "Tipo de Análisis", options= ["ANOVA"])

 ####################################################################################

 # CONTENIDO DE LA VISTA 5
if View == "ANOVA":
    #ANOVA
    #Generamos la lista de variables numéricas
    Lista_num= ['log_existencias', 'existencias', 'log_valor_inventario', 'valor_total_inventario']   #Devuelve lista de Columnas numéricas
    Lista_cat= ['marca_grupo', 'flujo', 'tipo_grupo', 'tipo']                 #Devuelve lista de Columnas categóricas
    #Select box
    Variable_y= st.sidebar.selectbox(label= "Variable dependiente (y)", options= Lista_num)
    #Select box
    Variable_x= st.sidebar.selectbox(label="Variable independiente (x)", options= Lista_cat)

    #Generamos los encabezados para el dashboard
    st.title("ANOVA")

    #Generamos el diseño del Layout deseado
    # Fila 1
    Contenedor_A, Contenedor_B = st.columns(2)
    #Redefinimos las variables
    Var_num= df[Variable_y]
    Var_cat= df[Variable_x]

    # Contenedor A: ANOVA y resultados
    with Contenedor_A:
        st.subheader("Resultados del ANOVA")
        grupos = [grupo[Variable_y].dropna() for nombre, grupo in df.groupby(Variable_x)]
        f_valor, p_valor = stats.f_oneway(*grupos)

        st.write(f"**F-valor:** {f_valor:.3f}")
        st.write(f"**p-valor:** {p_valor:.4e}")
        if p_valor < 0.05:
            st.success("¡Hay diferencias significativas entre los grupos!")
        else:
            st.warning("No hay diferencias significativas entre los grupos.")

    # Contenedor B: Gráfico de barras
    with Contenedor_B:
        st.subheader("Gráfico de Barras: Media por grupo")

        resumen = df.groupby(Variable_x)[Variable_y].agg(['mean', 'sem']).reset_index()

        fig, ax = plt.subplots()

        # Dibujo de las barras manualmente con matplotlib para controlar yerr
        x = resumen[Variable_x].astype(str)
        y = resumen['mean']
        yerr = resumen['sem']

        ax.bar(x, y, yerr=yerr, capsize=5, color='skyblue', edgecolor='black')
        ax.set_ylabel(f"Media de {Variable_y}")
        ax.set_xlabel(Variable_x)
        ax.set_title("Media por grupo con Error Estándar")
        plt.xticks(rotation=30, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
