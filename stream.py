import streamlit as st
import pandas as pd
import numpy as np
#import unidecode as uni
#import Util as utl
#from sklearn.cross_validation import cross_val_score
#from sklearn import metrics
#from sklearn import linear_model
#from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
import pickle
import matplotlib.pyplot as plt

 

def crear_dummis_modelos(modelo, precio):
    df = pd.DataFrame({'moto':pd.Series('moto'), 'samsung':pd.Series('samsung'),'tcl':pd.Series('tcl'),'xiaomi':pd.Series('xiaomi'),'iphone':pd.Series('iphone'),'lg':pd.Series('lg'),'sony':pd.Series('sony'),'nokia':pd.Series('nokia'),'blackberry':pd.Series('blackberry'),'huawei':pd.Series('huawei'),'otros':pd.Series('otros')})
    df0 = pd.DataFrame({'precio':pd.Series(precio)})
    df = df.applymap(lambda x: 1 if x==modelo else 0)
    df2 = pd.concat([df0,df],axis=1)
    return df2



def regresion_logistica(p_data, v_target=True):
    # DEPENDENCIAS
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix
    import unidecode as ud
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix,accuracy_score
    import seaborn as sns
    from sklearn.metrics import roc_auc_score
    from matplotlib import pyplot as plt
    
    data = p_data
    
    if v_target:
        data_aux = data[data.ventas==0].sample(200)# BALANCEAR LA CANTIDAD DE VENTAS = "0"
        data = data[data.ventas!=0]
        data =  pd.concat([data,data_aux],axis=0)
        data.ventas = data.ventas.apply(lambda x: 0 if (x<=1) else 1 if ((x>1)&(x<=5)) else 2)
    else:
        data.ventas = pd.qcut(data.ventas,2,labels=[0,1])
    
    
    data.precio = pd.qcut(data.precio,5,labels=[0,1,2,3,4])
    X = pd.DataFrame({'precio':data.precio,'moto':data.moto,'samsung':data.samsung,'tcl':data.tcl,'xiaomi':data.xiaomi,'iphone':data.iphone,'lg':data.lg,'sony':data.sony,'nokia':data.nokia,'huawei':data.huawei,'blackberry':data.blackberry,'otros':data.otros})
    y = data.ventas
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=50)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    modelo = LogisticRegression(C=1e10)
    modelo.fit(X_train,y_train)
    X_test = scaler.transform(X_test)
    y_predict = modelo.predict(X_test)

    y_test = np.array(y_test)
    y_predict = np.array(y_predict)

    if y.value_counts().shape[0] == 2:
        print('ROC_AUC_SCORE: ',roc_auc_score(y_test, y_predict))
    print(' ')
    print(classification_report(y_test,y_predict))
    print('')
    print(confusion_matrix(y_test,y_predict))
    mat = confusion_matrix(y_test, y_predict)
    sns.heatmap(mat, square=True, annot=True, fmt='d')
    plt.xlabel('Etiquetas predichas')
    plt.ylabel('Etiquetas verdaderas')
        
    return {'modelo':modelo,'scaler':scaler}





def naive_bayes(p_df, predecir=[], v_target=3):
    #DEPENDENCIAS
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix,accuracy_score 
    from sklearn.naive_bayes import MultinomialNB
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    scaler = StandardScaler()
    from sklearn.feature_extraction.text import TfidfVectorizer
    import unidecode as ud
    
    df = p_df
    
    if v_target==3:
        df_aux = df[df.ventas==0].sample(190)
        df = df[df.ventas!=0]
        df =  pd.concat([df,df_aux],axis=0)
        df.ventas = df.ventas.apply(lambda x: 0 if (x<=1) else 1 if ((x>1)&(x<=5)) else 2)
        df.ventas.value_counts()
    else:
        df.ventas = pd.qcut(df.ventas,2,labels=[0,1])
        
    X = pd.DataFrame({'titulo':df.titulo})
    X = X.titulo.apply(lambda x : ud.unidecode(x).lower())
    y = df.ventas

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=50)
    modelo_tfidf = TfidfVectorizer(ngram_range=(2,3), max_df=0.65)
    X_train = modelo_tfidf.fit_transform(X_train)
    y_train = np.array(y_train)
    if v_target==3:
        modelo_NB = MultinomialNB(alpha=0.001,class_prior=[.75,.99,.80])
    else:
        modelo_NB = MultinomialNB(alpha=0.001,class_prior=[.60,.30])
    modelo_NB.fit(X_train, y_train)
    X_test = modelo_tfidf.transform(X_test)
    X_test.get_shape()
    predicciones = modelo_NB.predict(X_test)
    predicciones_proba = modelo_NB.predict_proba(X_test)
    accuracy = accuracy_score(y_test, predicciones)
    #print('ROC_AUC_SCORE: ',roc_auc_score(y_test, predicciones))
    print(' ')
    print(classification_report(y_test,predicciones))
    print('')
    print(confusion_matrix(y_test,predicciones))
    mat = confusion_matrix(y_test, predicciones)
    sns.heatmap(mat, square=True, annot=True, fmt='d')
    plt.xlabel('Etiquetas predichas')
    plt.ylabel('Etiquetas verdaderas')
    salida = []
    if pd.Series(predecir).shape[0] != 0:
        for i in predecir:
            nuevos_X = pd.Series(i)
            nuevos_X = modelo_tfidf.transform(nuevos_X)
            if v_target==3:
                dic = {0:'pocas',1:'medias',2:'altas'}
            else:
                dic = {0:'pocas',1:'muchas'}
            pd.DataFrame(modelo_NB.predict_proba(nuevos_X))

            if (modelo_NB.predict_proba(nuevos_X).max())>0.45:
                print('')
                print('Prediccion de ventas para '+i+': ',dic[modelo_NB.predict(nuevos_X)[0]])
                print(modelo_NB.predict_proba(nuevos_X))
                if v_target==3:
                    salida_valor = modelo_NB.predict(nuevos_X)[0]
                    bar_df = pd.DataFrame(modelo_NB.predict_proba(nuevos_X),columns=['bajas ventas','medias ventas','altas ventas'])
                else:
                    salida_valor = modelo_NB.predict(nuevos_X)[0]
                    bar_df = pd.DataFrame(modelo_NB.predict_proba(nuevos_X),columns=['bajas ventas','altas ventas'])
                    
                salida.append((i,bar_df))
            else:
                salida_valor = -1
                print('')
                print('No se puede estimar correctamente')
    else:
        salida_valor = -2
        salida='n/p'
    return {'modeloPalabras':modelo_tfidf,'modelo':modelo_NB,'salida':salida,'salida_valor':salida_valor}





st.markdown('<style>body{}</style>', unsafe_allow_html=True) 

st.markdown('<style>.st-b9{}</style>', unsafe_allow_html=True) 


#st.markdown('<style>.st-dz{background:black;color:white;}</style>', unsafe_allow_html=True) 
#st.markdown('<style>.st-dk{background:white;color:white;}</style>', unsafe_allow_html=True) 
#st.markdown('<style>.st-e4{background:white;color:white;}</style>', unsafe_allow_html=True) 



st.write(
      '<div class=bloque_titulo><h1 class="titulo">CLASIFICADOR DE VENTAS</h1><h3 class="subtitulo">Celulares en MercadoLibre</h3></div>',
      unsafe_allow_html=True
)


st.markdown('<style>div.bloque_titulo{padding:5%;text-align:center;border-radius:15%;}</style>', unsafe_allow_html=True)

#st.write(
#      '<h3 class="subtitulo">Celulares en MercadoLibre</h3>',
#      unsafe_allow_html=True
#)

st.markdown('<style>h3.titulo{margin-top:0;margin-botton:-5px;}</style>', unsafe_allow_html=True)
st.markdown('<style>h3.subtitulo{margin-top:0;}</style>', unsafe_allow_html=True) 

dfm = pd.DataFrame({
  'Modelos': ['Regresion Logistica', 'Naive Bayes'],
  'second column': [1, 2]
})


st.write(
      '<h3 class="tipo_modelo">Seleccione el Modelo a Utilizar...</h3>',
      unsafe_allow_html=True
)


var_modelo = st.selectbox('   ',dfm['Modelos'])

var_modelo = var_modelo[0]




if var_modelo == 'R':
  st.write(
      '<h1 class="c_precio">Predecir con Regresion Logistica...</h3>',
      unsafe_allow_html=True
  ) 


  st.write(
      '<h3 class="c_precio">Ingrese el Precio...</h3>',
      unsafe_allow_html=True
  ) 


  dfm_precios = pd.DataFrame({
  'Precios': ['$9.999-$22999.0', '$22999.0-$45069.0','$45069.0-$78989.0','$78989.0-$154666.0','$154666.0-$449999.0'],
  })

  var_precios_celulares = st.selectbox('   ',dfm_precios['Precios'])

  dic_precios={'$9.999-$22999.0':0,'$22999.0-$45069.0':1,'$45069.0-$78989.0':2,'$78989.0-$154666.0':3,'$154666.0-$449999.0':4}



  st.write(
      '<h3 class="c_precio">Seleccione el Modelo...</h3>',
      unsafe_allow_html=True
  ) 

  dfm_celular = pd.DataFrame({
  'Modelos': ['MOTO', 'SAMSUNG','IPHONE','LG','TCL','XIAOMI','SONY','NOKIA','BLACKBERRY','HUAWEI','OTROS'],
  })

  var_modelo_celular = st.selectbox('   ',dfm_celular['Modelos'])

  
  var_modelo_celular = var_modelo_celular.lower()



  if st.button('Predecir Ventas'):
    
    nuevos_x = crear_dummis_modelos(var_modelo_celular,dic_precios[var_precios_celulares])


    data = pd.read_csv('data_celulares_modelo.csv')
    dic_modelo = regresion_logistica(data, v_target=False)
    modelo = dic_modelo['modelo']
    scaler = dic_modelo['scaler']
    nuevos_x = scaler.transform(nuevos_x)
        
    predict = modelo.predict(nuevos_x)[0]

    if predict == 0:
      st.write(
        '<h3 class="c_prediccion">El Modelo Estima Pocas Ventas...</h3>',
          unsafe_allow_html=True
      ) 
      st.markdown('<style>h3.c_prediccion{color:white;font-size:2em;background:#F12424;padding:10%;}</style>', unsafe_allow_html=True)
  

    else:
      st.write(
          '<h3 class="c_prediccion">El Modelo Estima Muchas Ventas...</h3>',
          unsafe_allow_html=True
      )
      st.markdown('<style>h3.c_prediccion{color:white;font-size:2em;background:#28E21B;padding:10%;}</style>', unsafe_allow_html=True) 





else:
  st.write(
      '<h1 class="c_precio">Predictor Naive Bayes...</h3>',
      unsafe_allow_html=True
  ) 

  st.write(
    '<h3 class="c_precio">Ingrese caracteristicas del Celular...</h3>',
    unsafe_allow_html=True
  ) 


  var_titulo = ''
  var_titulo = st.text_input('')

  v_dimension = st.checkbox("Decision Binaria")

  if v_dimension:
    v_target=2
  else:
    v_target=3

  if st.button('Predecir Ventas'):
    df = pd.read_csv('data_celulares_modelo.csv').iloc[:,1:]

    l_celulares=[var_titulo]
    df_bar = naive_bayes(df,l_celulares,v_target=v_target)['salida_valor']

    
    df = pd.read_csv('data_celulares_modelo.csv').iloc[:,1:]
    l_celulares=[var_titulo]
    
    modelo = naive_bayes(df,l_celulares,v_target=v_target)
    modeloBayes = modelo['modelo']
    modeloPalabras = modelo['modeloPalabras']

    var_titulo = modeloPalabras.transform(pd.Series(var_titulo))
    predictproba = modeloBayes.predict_proba(var_titulo)
    if v_dimension:
      bar_df = pd.DataFrame(predictproba,columns=['bajas ventas','altas ventas'])
    else:
      bar_df = pd.DataFrame(predictproba,columns=['bajas ventas','medias ventas', 'altas ventas'])
    bar_df.plot.bar(rot=0)
    st.pyplot()



st.title('')
st.write('<div class="github"><a href="https://github.com/emanuelbaquero/celulares-ml"><img src="https://img2.freepng.es/20180331/udw/kisspng-social-media-github-computer-icons-logo-github-5ac0188083c4f5.8572681115225386245397.jpg"><p class="cuenta de git"></p></a></div>', unsafe_allow_html=True)
st.markdown('<style>div.github a{text-aling:center;}</style>',unsafe_allow_html=True)
st.markdown('<style>div.github img{ width:10%;opacity:.2;padding:.5%;text-align:center;border-radius:50%;margin-left:47%;}</style>',unsafe_allow_html=True)
st.markdown('<style>div.github img:hover{padding:.1%;opacity: 1;}</style>',unsafe_allow_html=True)
