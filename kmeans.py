import pandas as pd
import yfinance as yf
import numpy as np
import warnings

import plotly.express as px  #Criação de graficos dinâmnicos
import plotly.offline as py
import plotly.graph_objects as go #Para criação e concatenização de graficos

from plotly.subplots import make_subplots
from pandas_datareader import data as pdr
from sklearn.cluster import KMeans

import pandas_datareader, sklearn, plotly
warnings.filterwarnings("ignore")


# Função para calcular valores de WCSS
def calcular_wcss(dados_ativos):
    wcss = []
    for k in range(1,11):
        kmeans = KMeans(n_clusters = k, random_state=0, init='k-means++' )
        kmeans.fit(X=dados_ativos)        
        wcss.append(kmeans.inertia_)
    return wcss

#Função gera grafico
def gera_grafico (X, Y, color, X_centroide, Y_centroide, name):
    grafico = px.scatter(x=X ,
                         y=Y,
                         color = color
                         )
    grafico_centroide = px.scatter( x = X_centroide , y=Y_centroide, size=[7,7,7])
    grafico_centroide = go.Figure(data=grafico.data + grafico_centroide.data)
    grafico_centroide.update_xaxes(title_text='Largura')
    grafico_centroide.update_yaxes(title_text='Altura')
    grafico_centroide.update_layout(title_text="Clusters de " + name, title_x=0.5)
    grafico_centroide.show()



print(f''' Relação das bibliotecas e suas respectivas versões
------------------------------------------------------------
  pandas: {pd.__version__}
  numpy: {np.__version__}
  yahoo ficances: {yf.__version__}
  pandas_datareader: {pandas_datareader.__version__}
  plotly: {plotly.__version__}
  sklearn: {sklearn.__version__}
------------------------------------------------------------
        '''
      )
#A primeira etapa é coletar a lista de ativos alvo.
dados_ativos = pd.read_csv(r'.\dataset\ativos.csv',
                            sep = ';', encoding='latin1'
                           )

# Cria lista de dataframe para coletar e armazenar dados de cada ativo
lista_df = []
for ativo in dados_ativos['ativo']:
    try:
        df = pdr.get_data_yahoo(ativo, 
                                start="2022-01-01", end="2022-12-31")       
        # Cria nova coluna com o nome do ativo
        df['ativo'] = ativo
        # Adiciona os dados em uma lista
        lista_df.append(df)
    except e:
        print(f'Foi encontrado um erro no ativo: {ativo} - erro: {e}')
              
df_ativos = pd.concat(lista_df) 

#Verificar quantidade de dados coletados
print(f'O dataset coletado possui {len(df_ativos)} linhas')

df_ativos['dif_percentual'] = df_ativos['Adj Close'].pct_change()
#Adicionar mais conteudo na base de dados
retorno = (df_ativos.groupby(['ativo'])
                             .agg(retorno=('dif_percentual', 'mean'))*252)
#Indica qual o retorno financeiro o ativo proporciona para a pessoa
volatividade = (df_ativos.groupby(['ativo'])
                             .agg(volatividade = ('dif_percentual', 'std'))* np.sqrt(252))
#indica o quanto que o ativo oscila. É calculado com o desvio padrão
analise_ativos = pd.merge(retorno, volatividade, how='inner', on ='ativo')

# Resetando o index da nova tabela e visualizando dados gerados
analise_ativos.reset_index(inplace=True)
analise_ativos.head()

# Seleciona as variáveis que serão utilizadas na clusterização
dados_ativos = analise_ativos[['retorno','volatividade']]

# Realiza o cálculo do WCSS
wcss_ativos = calcular_wcss(dados_ativos)

# Visualizando os dados obtidos do WCSS
for i in range(len(wcss_ativos)):
  print(f'O cluster {i} possui valor de WCSS de: {wcss_ativos[i]}')

#Criaçao do gráfico cotovelo, essencial para selecionar quantidade de clusters

grafico_wcss = px.line( x = range(1,11),
                        y = wcss_ativos
                       )
fig = go.Figure(grafico_wcss)

fig.update_layout(title='Calculando o WCSS',
                  xaxis_title= 'Número de clusters',
                  yaxis_title= 'Valor do Wcss', 
                  template =  'plotly_white'
                  ) 

fig.show() #mostrando gráfico cotovelo

kmeans_ativos = KMeans(n_clusters=5, 
                       random_state=0, 
                       init='k-means++')
analise_ativos['cluster'] = kmeans_ativos.fit_predict(dados_ativos)
centroides = kmeans_ativos.cluster_centers_

print('Construindo grafico')
gera_grafico(x=analise_ativos["volatilidade"], y=analise_ativos["retorno"], X_centroide=centroides[:1],
             Y_centroide=centroides[:2],
             name="Analise K-MEANS", color="")


fig = make_subplots(rows = 1, cols = 1,
                    shared_xaxes = True,
                    vertical_spacing = 0.08)

fig.add_trace(go.Scatter( x = analise_ativos["volatividade"], 
                          y = analise_ativos["retorno"],
                          name = "", mode = "markers",
                          text = analise_ativos['ativo'],
                          marker = dict(size = 14, color = analise_ativos["cluster"])
                        )
              )

fig.update_layout( height = 600, width = 900,
                   title_text = "Análise de Clusters",               
                   xaxis_title = "Volatividade",
                   yaxis_title = "Retorno"               
                 )


fig.show()