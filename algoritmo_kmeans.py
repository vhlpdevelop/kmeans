import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

iris = pd.read_csv('iris.csv', sep=',')

#iris.head() #primeiros registros
#iris.tail() #ultimos registros
# print(iris.head())
#print(iris.describe())

#print(iris.shape)

#print(iris.isna().sum()) verificar nulos

#print(iris.info())

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


x_petalas= iris.iloc[:, [2,3]].values
print(x_petalas[:10])

#normaliza os dados é necessário para eliminar discrepancias 
#no caso da iris nao é necessário mas na vida real é...
normalizar_dados = StandardScaler()
x_petalas= normalizar_dados.fit_transform(x_petalas)
print(x_petalas)
#Dados da X_petalas foram normalizados

#Calcular número de cluster
wcss_petala = []
#caso nao soubessemos o número ideal...
for i in range(1,11):
    kmeans_petala = KMeans(n_clusters=i, random_state=0) 
    kmeans_petala.fit(x_petalas) #Realiza o treinamento dos dados
    wcss_petala.append(kmeans_petala.inertia_) #inertia é o valor do dado css que é inserido na lista

for i in range(len(wcss_petala)):
    print('Cluster: ', i+1, '- Valor do wcss ', wcss_petala[i]) #resultado vai ser número ideal 3 ou 4 de cluster

grafico_cotovelo_petala = px.line(x=range(1,11), y=wcss_petala) #cria o grafico cotovelo, mostra o número ideal de grupos
grafico_cotovelo_petala.update_xaxes(title_text='Num de Clusters')
grafico_cotovelo_petala.update_yaxes(title_text='Valor WCSS')
grafico_cotovelo_petala.update_layout(title_text='Gráfico Cotovelo', title_x=0.5)
#grafico_cotovelo_petala.show() mostra o grafico cotovelo gerado

kmeans_petala = KMeans(n_clusters=3, random_state=0)
label_cluster_petala = kmeans_petala.fit_predict(x_petalas)
print(label_cluster_petala) #clusters criados

centroides_petala = kmeans_petala.cluster_centers_ #centralizando os clusters
print(centroides_petala) #Vetor que possui os centroides
grafico_petala = px.scatter(x=x_petalas[:,0], y=x_petalas[:,1], color=label_cluster_petala) #gráfico petala
grafico_centroide_petala = px.scatter( x = centroides_petala[:,0] , y=centroides_petala[:,1], size=[7,7,7]) #criação do grafio centroide maior para visualizar
grafico_final = go.Figure(data=grafico_petala.data + grafico_centroide_petala.data) #junção dos graficos centroide e petalas
 #grafico_final.show() #grupo de clusters feito com sucesso, todos agrupados com sua cor

#construindo as sepalas, mesma ideia das petalas porém valor 0,1 das colunas
x_sepala = iris.iloc[:, [0,1]].values
x_sepala = normalizar_dados.fit_transform(x_sepala)
wcss_sepala = []
for i in range(1,11):
    kmeans_sepala = KMeans(n_clusters=i, random_state=0) 
    kmeans_sepala.fit(x_sepala) #Realiza o treinamento dos dados
    wcss_sepala.append(kmeans_sepala.inertia_) #inertia é o valor do dado css que é inserido na lista

for i in range(len(wcss_sepala)):
    print('Cluster: ', i+1, '- Valor do wcss ', wcss_sepala[i])



kmeans_sepala = KMeans(n_clusters=3, random_state=0)
label_cluster_sepala = kmeans_sepala.fit_predict(x_sepala)

#construindo gráfico das sepalas
centroides_sepala = kmeans_sepala.cluster_centers_ #centralizando os clusters
print(centroides_sepala) #Vetor que possui os centroides
grafico_sepala = px.scatter(x=x_sepala[:,0], y=x_sepala[:,1], color=label_cluster_sepala) #gráfico sepala
grafico_centroide_sepala = px.scatter( x = centroides_sepala[:,0] , y=centroides_sepala[:,1], size=[7,7,7]) #criação do grafio centroide maior para visualizar
grafico_final_sepala = go.Figure(data=grafico_sepala.data + grafico_centroide_sepala.data) #junção dos graficos centroide e sepala
grafico_final_sepala.update_xaxes(title_text='Largura')
grafico_final_sepala.update_yaxes(title_text='Altura')
grafico_final_sepala.update_layout(title_text='Clusters de sepalas', title_x=0.5)
# grafico_final_sepala.show() #grupo de clusters feito com sucesso, todos agrupados com sua cor

#agora chamar função construida gera_grafico
X = x_petalas[:,0]
Y = x_petalas[:,1]
color = label_cluster_petala
X_centroide = centroides_petala[:,0]
Y_centroide = centroides_petala[:,1]
gera_grafico(X, Y, color, X_centroide, Y_centroide, "Petalas") #constroi grafico petala
X = x_sepala[:,0]
Y = x_sepala[:,1]
color = label_cluster_sepala
X_centroide = centroides_sepala[:,0]
Y_centroide = centroides_sepala[:,1]
gera_grafico(X, Y, color, X_centroide, Y_centroide, "Sepala") #constroi grafico sepala