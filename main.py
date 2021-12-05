import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import pickle
#librerías de pre-processing
import datetime as dt

#librería del modelo
from ngboost import NGBRegressor



pickle_in = open('modelo.pickle', 'rb')
#Unpickle modelo
best_model = pickle.load(pickle_in)

#Unpickle temperatura
media_train_temp = pickle.load(pickle_in)
std_train_temp = pickle.load(pickle_in)
ult_temp = pickle.load(pickle_in)

#Unpickle poblacion
media_train_pob = pickle.load(pickle_in)
std_train_pob = pickle.load(pickle_in)
ult_pob = pickle.load(pickle_in)

#Unpickle PIB
media_train_pib = pickle.load(pickle_in)
std_train_pib = pickle.load(pickle_in)
ult_pib = pickle.load(pickle_in)

#Unpickle desempleo
media_train_des = pickle.load(pickle_in)
std_train_des = pickle.load(pickle_in)
ult_des = pickle.load(pickle_in)

#Unpickle rezagos
media_train_rez = pickle.load(pickle_in)
std_train_rez = pickle.load(pickle_in)
ult_rez = pickle.load(pickle_in)

#Unpickle muestra
x_muestra = pickle.load(pickle_in)

#Unpickle festivos
base_festivos = pickle.load(pickle_in)


def predecir(x_pred, modelo = best_model):
    y_dist = modelo.pred_dist(x_pred)
    media = y_dist.params['loc'][0]
    std = y_dist.params['scale'][0]
    return(media, std)


def get_predictores(inc_temp, fecha, inc_pib,inc_pob, inc_des, med_temp=media_train_temp, std_temp=std_train_temp, ult_temp=ult_temp, med_pob=media_train_pob, std_pob=std_train_pob, ult_pob=ult_pob, med_pib=media_train_pib, std_pib=std_train_pib, ult_pib=ult_pib,  med_des=media_train_des, std_des=std_train_des, ult_des=ult_des, med_rez=media_train_rez, std_rez=std_train_rez, ult_rez=ult_rez, base_festivos=base_festivos, x_muestra=x_muestra ):
    # Tratamiento de temperaturas
    temp = ult_temp+ inc_temp
    temp_est = (temp - med_temp)/std_temp
    temp_est = temp_est.to_list()
    
    # Población
    poblacion = ult_pob*(1+inc_pob)
    poblacion = (poblacion-med_pob)/std_pob
    
    # Tratamiento del PIB
    pib = ult_pib*(1+inc_pib)
    pib_est = (pib-med_pib)/std_pib
    pib_est = pib_est.to_list()
    
    # Tratamiento del desempleo
    des = ult_des*(1+inc_des)
    des_est = (des-med_des)/std_des
    
    # Rezagos
    rez = (ult_rez-med_rez)/std_rez
    
    list_predictores = [fecha]+[rez] + temp_est + [poblacion] + pib_est + [des_est]
    predictores = pd.DataFrame(list_predictores, index= ['fecha']+['rez'] + pd.DataFrame(ult_temp).T.columns.to_list()+['Población económicamente activa']+pd.DataFrame(ult_pib).T.columns.to_list() + ['tasa_desempleo'] ).T
    
    # Tratamiento de fechas
    predictores['fecha'] = pd.to_datetime(predictores['fecha'], format='%Y-%m-%d')
    
    dias = []
    for i in predictores['fecha']:
        dias.append(pd.Timestamp(i).day_name())
    meses = []
    for i in predictores['fecha']:
        meses.append(pd.Timestamp(i).month_name())
    predictores['dia_semana'] = dias
    predictores['nombre_mes'] = meses
    
    base_festivos['fecha'] = pd.to_datetime(base_festivos['fecha'], format='%d/%m/%Y')
    
    predictores['week_end'] = [1 if x == 'Sunday' or x == 'Saturday' else 0 for x in predictores['dia_semana']]
    predictores = pd.merge(predictores,base_festivos, how='left', left_on='fecha', right_on='fecha',right_index=False)
    predictores.drop(columns=['fecha'], inplace=True)
    predictores['festivo_we']=predictores.apply(lambda x: 1 if x['festivo']==1 or x['week_end']==1 else 0,axis=1)
    
    predictores['August'] = np.where(predictores['nombre_mes']=='August', 1, 0)
    predictores['December'] = np.where(predictores['nombre_mes']=='December', 1, 0)
    predictores['February'] = np.where(predictores['nombre_mes']=='February', 1, 0)
    predictores['January'] = np.where(predictores['nombre_mes']=='January', 1, 0)
    predictores['July'] = np.where(predictores['nombre_mes']=='July', 1, 0)
    predictores['June'] = np.where(predictores['nombre_mes']=='June', 1, 0)
    predictores['March'] = np.where(predictores['nombre_mes']=='March', 1, 0)
    predictores['May'] = np.where(predictores['nombre_mes']=='May', 1, 0)
    predictores['November'] = np.where(predictores['nombre_mes']=='November', 1, 0)
    predictores['October'] = np.where(predictores['nombre_mes']=='October', 1, 0)
    predictores['September'] = np.where(predictores['nombre_mes']=='September', 1, 0)
        
    predictores.drop(columns=['week_end','dia_semana','festivo','nombre_mes'], inplace=True)
    
    orden = pd.DataFrame(x_muestra).T.columns.to_list()
    predictores = predictores[orden]
    
    return predictores

## MAIN

st.title("Prediciendo la demanda de Energia")

st.write("Nuestro proposito principal con este proyecto es poder disponibilizar una herramienta que nos permita realizar predicciones de la demanda de energia en Colombia que tenga en cuenta la incertidumbre alrededor de esta tarea. A través de la utilización de un modelo de Natural Gradient Boosting. En la barra lateral izquierda encuentran un menú de opciones en los cuales podemos escoger el escenario futuro sobre el cual realizaremos nuestra predicción la distribución de densidad la probabilidad la demanda energetica en ese periodo.")

#chart_data = pd.DataFrame(
  #   np.random.normal(1, 5,1000))




col1,col2=st.columns([4,1])


#col1.line_chart(chart_data)


## SIDEBAR
st.sidebar.subheader(" Inputs del Modelo")

st.sidebar.write("Fecha")

fecha= st.sidebar.date_input("Selecciona la fecha en la cual quieras realizar tu predicción")

#st.sidebar.write("La fecha seleccionada actualmente es: ",d)

st.sidebar.write("Cambio en temperatura")

inc_temp = st.sidebar.number_input('Inserte el incremento esperado en grados celsius de temperatura para esa fecha')
#st.sidebar.write('El incremento seleccionado actualmente es: ', number)

st.sidebar.write("Cambio en PIB")

pib = st.sidebar.number_input('Inserte el incremento porcentual esperado en el PIB para esa fecha')
#st.sidebar.write('El incremento seleccionado actualmente es: ', pib)


st.sidebar.write("Cambio en Desempleo")

u = st.sidebar.number_input('Inserte el incremento esperado en el desempleo para esa fecha')
#st.sidebar.write('El incremento seleccionado actualmente es: ', u)

st.sidebar.write("Cambio en población")

pob = st.sidebar.number_input('Inserte el incremento esperado en la población para esa fecha')
#st.sidebar.write('El incremento seleccionado actualmente es: ', pob)
st.set_option('deprecation.showPyplotGlobalUse', False)

boton_actualizar = st.sidebar.button("¡Predecir!")

if boton_actualizar:

    mean_,std_=predecir(get_predictores(inc_temp,fecha,pib,pob,u))
    chart_data = pd.DataFrame(
     (np.random.normal(mean_, std_, 1000)*std_train_rez)+media_train_rez)
 
    hist_plot=sns.distplot(chart_data)
    hist_plot.set(xlabel='Megavatios/hora', ylabel='Densidad de probabilidad')
    col1.pyplot()


col1.markdown("## El Modelo: NG-Boost")
col1.write("""Natural Gradient Boosting es una metodologia desarrollada para poder realizar predicción probabilistica. 
Esto nos permite no solamente realizar predicciones precisas de los valores reales sino también realizar una estimación de la incertidumbre. 
A diferencia de otra metodologias que pueden ser utilizadas para este tipo de predicciones, esta utiliza un enfoque modular. """)

col1.markdown("### ¿Cómo funciona?")
from PIL import Image
image = Image.open('how_the_model_works.PNG')

col1.image(image, caption='How the model works')
col1.write("Este modelo toma un enfoque modular que se separa en 3 componentes")
col1.markdown(""" - **Base Learner:** Al igual que todo modelo de boosting partimos de un modelo sencillo de aprendizaje que va mejorando iteración a iteración.
                """)
col1.markdown(""" 
                - **Distribución parámetrica:** En este caso no estamos entrenando un modelo para predicción punto, sino predicción probabilistica. Por lo cual este modelo parte
                del supuesto de que nuestra variable de interés sigue alguna distribución conocida y nosotros estimaremos los parámetros que caracterizan dicha distribución.
                En este caso suponemos que la demanda de energia sigue una distribución normal y estimaremos la media y la desviación estandar que la caracterizan.""")

col1.markdown(""" - **Función objetivo:** Este modelo parte de una función objetivo. En este caso utilizamos un maxima verosimilitud, 
                      como función objetivo. La regla recibe como insumo una distribución de probabilidad predicha y un valor verdadero de la variable de interés
                      y asigna un valor a través de la función objetivo de poder replicar el valor verdadero con la distribución de probabilidad predicha.""")

