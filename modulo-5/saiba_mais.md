# Para saber mais: Otimização de Parâmetros

Mesmo após comparar dois algoritmos e escolher qual é o melhor,  ainda existe espaço para explorar os algoritmos e encontrar o melhor  modelo, faremos isso através dos hiperparâmetros.

Os hiperparâmetros são características do seu modelo que podem ser definidas através dos parâmetros, por exemplo o [DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html) conta com parâmetros **max_depth** e **min_samples_split** que dependendo do valor vão te entregar um modelo melhor adaptado aos seus dados.

Mas isso nos trás uma questão, basta olhar para a documentação de  algum desses algoritmos e vamos notar a infinidade de parâmetros que  temos e assim muitas possibilidades. Para esse problema vamos ter duas  soluções, definir os valores e hiperparâmetros que vamos explorar ou  explorar aleatoriamente, essas duas estratégias estão implementadas no  scikit-learn, o [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) e o [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html) 

Vamos começar pelo **GridSearchCV**, ele nos permite definir um espaço que queremos explorar, por exemplo:

```python
espaco_de_parametros = {
  "max_depth" : [3, 5],
  "min_samples_split" : [32, 64, 128],
  "min_samples_leaf" : [32, 64, 128],
  "criterion" : ["gini", "entropy"]
}
```

Aqui definimos um dicionário onde a chave é o nome do parâmetro que queremos otimizar e o valor é uma lista com os valores que queremos que ele explore. Então  nesse caso ele vai treinar a árvore usando uma profundidade máxima, **max_depth**, 3 e 5.

Para usar o **GridSearchCV** vamos repetir alguns parâmetro que usamos no **cross_validate**.

Código da aula:

```python
cv = RepeatedStratifiedKFold(n_splits = n_splits, n_repeats=n_repeats)
resultados=cross_validate(modelo, x, y, cv=cv, scoring='roc_auc', return_train_score=True)
```

A novidade vai ser que não vamos mandar o x e o y, e vamos usar o parâmetro **param_grid** que será o nosso dicionário.

```python
cv = RepeatedStratifiedKFold(n_splits = n_splits, n_repeats=n_repeats)

busca = GridSearchCV(modelo,              
                     param_grid=espaco_de_parametros,
                     cv = cv, scoring='roc_auc',
                     return_train_score=True)
busca.fit(x, y)

resultados = pd.DataFrame(busca.cv_results_).iloc[busca.best_index_]
```

Outra diferença que temos é que teremos mais de um modelo, então precisamos  selecionar o que teve o melhor desempenho, temos acesso a ele por **busca.best_index_** e então selecionamos eles dentro dos nossos resultados.

```python
resultados = pd.DataFrame(busca.cv_results_).iloc[busca.best_index_]

auc_medio = resultados['mean_test_score']
auc_medio_treino = resultados['mean_train_score']

auc_std = resultados['std_test_score']
```

Com isso, conseguimos descobrir o melhor modelo dentre o grupo de  parâmetros que definimos, mas isso pode ser um problema, porque o espaço de possibilidade é bem maior, então a melhor combinação de parâmetros  pode estar em outro lugar nesse espaço, pensando nisso temos a segunda  solução, o **RandomizedSearchCV** nele vamos definir o  espaço que queremos explorar e ele aleatoriamente vai testar as  combinações de hiperparâmetros, e ele vai testar a quantidade que você  determinar pelo parâmetro **n_iter**, número de interações.

Primeiro vamos definir esse espaço, veja que agora podemos incluir um grande número de possibilidades, já que não serão todas testadas, por  exemplo no **min_samples_split** vamos testar números aleatórios entre 32 e 129.

```python
from scipy.stats import randint

espaco_de_parametros = {
    "n_estimators" :randint(10, 101),
    "max_depth" : randint(3, 6),
    "min_samples_split" : randint(32, 129),
    "min_samples_leaf" : randint(32, 129),
    "bootstrap" : [True, False],
    "criterion" : ["gini", "entropy"]
}
```

Com o espaço definido podemos seguir para utilização do **RandomizedSearchCV**, veja que vamos usar de maneira semelhante ao **GridSearchCV** porém com um parâmetro a mais o **n_iter**, que é a quantidade de combinações que ele vai tentar.

```python
cv = RepeatedStratifiedKFold(n_splits = n_splits, n_repeats=n_repeats)

busca = RandomizedSearchCV(modelo, param_distributions=espaco_de_parametros,
                           n_iter = n_iter, cv = cv, scoring='roc_auc',
                           return_train_score=True)
busca.fit(x, y)
```

Para poder comparar os resultados, modifiquei a função apresentada na aula para utilizar esses otimizadores de hiperparâmetros:

Primeiro o GridSearchCV:

```python
from sklearn.model_selection import GridSearchCV

def roda_modelo_GridSearchCV(modelo, dados, n_splits, n_repeats, espaco_de_parametros):

    np.random.seed(1231234)
    dados = dados.sample(frac=1).reset_index(drop=True)
    x_columns = dados.columns
    y = dados["ICU"]
    x = dados[x_columns].drop(["ICU","WINDOW"], axis=1)


    cv = RepeatedStratifiedKFold(n_splits = n_splits, n_repeats=n_repeats)

    busca = GridSearchCV(modelo, param_grid=espaco_de_parametros,
                         cv = cv, scoring='roc_auc',
                         return_train_score=True)
    busca.fit(x, y)

    resultados = pd.DataFrame(busca.cv_results_)

    auc_medio = resultados.iloc[busca.best_index_]['mean_test_score']
    auc_medio_treino = resultados.iloc[busca.best_index_]['mean_train_score']

    auc_std = resultados.iloc[busca.best_index_]['std_test_score']

    print(f'AUC  {auc_medio} - {auc_medio_treino}')
    return auc_medio, auc_medio_treino    
```

Depois o RandomizedSearchCV:

```python
from sklearn.model_selection import RandomizedSearchCV

def roda_modelo_RandomizedSearchCV(modelo, dados, n_splits, n_repeats, espaco_de_parametros, n_iter):

    np.random.seed(1231234)
    dados = dados.sample(frac=1).reset_index(drop=True)
    x_columns = dados.columns
    y = dados["ICU"]
    x = dados[x_columns].drop(["ICU","WINDOW"], axis=1)


    cv = RepeatedStratifiedKFold(n_splits = n_splits, n_repeats=n_repeats)

    busca = RandomizedSearchCV(modelo, param_distributions=espaco_de_parametros,
                                n_iter = n_iter, cv = cv, scoring='roc_auc',
                                return_train_score=True)
    busca.fit(x, y)

    resultados = pd.DataFrame(busca.cv_results_)

    auc_medio = resultados.iloc[busca.best_index_]['mean_test_score']
    auc_medio_treino = resultados.iloc[busca.best_index_]['mean_train_score']

    auc_std = resultados.iloc[busca.best_index_]['std_test_score']

    print(f'AUC  {auc_medio} - {auc_medio_treino}')
    return auc_medio, auc_medio_treino    
```

Com isso podemos notar que podemos explorar ainda mais os algoritmos,  recomendo explorar a documentação dos algoritmos que estiver trabalhando e verificar se há hiperparâmetros a serem explorados neles e encontrar o melhor para os seus dados e seu objetivo.

Qualquer dúvida que tenha sobre essas estratégias apresentadas é só nos procurar no Discord e esse é o [link](https://github.com/alura-cursos/covid-19-clinical-2/blob/main/Modulo_6_e_Saiba_Mais.ipynb) para o código completo.



# Para saber mais: Pipelines

Encontrar o melhor o modelo vai ficando mais complexo conforme  aprendemos novas etapas como normalização de dados, validação cruzada e  otimização de parâmetros. Pensando em resolver essa complexidade e  evitar erros nesses processos, vamos conhecer um recurso do [scikit-learn](https://scikit-learn.org/stable/), os [pipelines](https://scikit-learn.org/stable/modules/compose.html#pipeline) e o que eles nos permitem, como por exemplo fixar uma sequência linear de etapas, que são:

- [Dados], [processamento de dados], [seleção de características], [normalização], [classificação].

Vantagens:

- Código mais legível
- Dificulta o vazamento de dados

O scikit-learn nos permite construir essa estrutura através do [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline) e para utilizá-lo definimos a sequência de passos utilizando uma lista  que é composta de tuplas e essa por sua vez, terá o nome da etapa e a  classe:

```python
from sklearn.pipeline import Pipeline
pipe = Pipeline([('nome_etapa1', SelectKBest()),
                 ('nome_etapa2', Normalized()),
                 ('modelo',  LogisticRegression())])
```

Com o Pipeline criado, ele será utilizado no lugar do modelo, por exemplo no [cross_validate](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html) ou [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html).

No exemplo optei por criar a nossa própria etapa, e ela será baseada na função aprendida na aula a **remove_corr_var**

```python
def remove_corr_var(dados, valor_corte):

    matrix_corr = dados.iloc[:,4:-2].corr().abs()
    matrix_upper = matrix_corr.where(np.triu(np.ones(matrix_corr.shape), k=1).astype(np.bool))
    excluir = [coluna for coluna in matrix_upper.columns if any(matrix_upper[coluna] > valor_corte)]

    return dados.drop(excluir, axis=1)
```

Essa função identificou as colunas com alta correlação e as removeu, vamos  repetir ela, mas agora dentro do nosso processo de treinamento.

Quando o **cross_validation** separa nossos dados em  teste e treino, a ideia é que a nossa função leia os dados de treino,  identifique as colunas com alta correlação e as remova tanto dos dados  de treino quanto dos dados de teste.

Não vou me aprofundar muito nos conceitos de Orientação Objeto, mas para criarmos essa solução precisamos criar uma [classe](https://www.alura.com.br/artigos/poo-programacao-orientada-a-objetos), optei pelo nome **RemoveCorrVar** e nela vamos construir nossa estratégia.

Para o Pipeline aceitar nossa classe precisamos colocar entre parênteses **(BaseEstimator, TransformerMixin)** que são outras classes que o pipeline requisita.

Vamos também criar esse **__init__** e nele definir o **valor_corte**, que é o valor de correlação máxima que vamos manter.

A parte de encontrar as colunas com alta correlação vai ficar dentro do **fit** que por padrão vai receber os nossos dados X e Y.

```python
class RemoveCorrVar(BaseEstimator, TransformerMixin):
    def __init__( self, valor_corte = 0.95):
        self.valor_corte = valor_corte

    def fit( self, X, y = None ):
        matrix_corr = X.iloc[:,4:].corr().abs()
        matrix_upper = matrix_corr.where(np.triu(np.ones(matrix_corr.shape), k=1).astype(np.bool))
        self.excluir = [coluna for coluna in matrix_upper.columns if any(matrix_upper[coluna] > self.valor_corte)]
        return self
```

Você deve ter notado que não retiramos as colunas ainda, apenas salvamos elas na variável **self.excluir** . O motivo disso é que temos um trecho específico para isso, ele é chamado de **transform**, o drop vai ficar nesse trecho para que ele seja aplicado tanto nos dados de treino, quanto nos de teste.

```python
class RemoveCorrVar(BaseEstimator, TransformerMixin):
    def __init__( self, valor_corte = 0.95):
        self.valor_corte = valor_corte

    def fit( self, X, y = None ):
        matrix_corr = X.iloc[:,4:].corr().abs()
        matrix_upper = matrix_corr.where(np.triu(np.ones(matrix_corr.shape), k=1).astype(np.bool))
        self.excluir = [coluna for coluna in matrix_upper.columns if any(matrix_upper[coluna] > self.valor_corte)]
        return self 

    def transform(self, X, y = None):
        X = X.drop(self.excluir, axis=1)
        return X
```

Com a etapa pronta, podemos incluí-la no Pipeline, optei pelo nome **cat_selector**, seletor de características.

```python
 pipeline = Pipeline([('cat_selector', RemoveCorrVar()),
                         ('Modelo', modelo)])
```

Finalmente podemos utilizar agora o pipeline no lugar do modelo dentro da função **roda_modelo_cv** criada na aula.

```python
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

def roda_modelo_cv_pipeline(modelo, dados, n_splits, n_repeats):

    np.random.seed(1231234)
    dados = dados.sample(frac=1).reset_index(drop=True)
    x_columns = dados.columns
    y = dados["ICU"]
    x = dados[x_columns].drop(["ICU","WINDOW"], axis=1)

    pipeline = Pipeline([('cat_selector', RemoveCorrVar()),
                         ('Modelo', modelo)])    

    cv = RepeatedStratifiedKFold(n_splits = n_splits, n_repeats=n_repeats)
    resultados=cross_validate(pipeline, x, y, cv=cv, scoring='roc_auc', return_train_score=True)

    auc_medio = np.mean(resultados['test_score'])
    auc_medio_treino = np.mean(resultados['train_score'])

    auc_std = np.std(resultados['test_score'])

    print(f'AUC  {auc_medio} - {auc_medio_treino}')
    return auc_medio, auc_medio_treino)
```

Então agora cada vez que nosso modelo for treinado no **cross_validate** antes ele vai remover as colunas que julgamos desnecessárias.

Existem diversas etapas já prontas no sklearn para o [processamento dos dados](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-scaler), por exemplo o [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler) que vai escalar os valores do seus dados, então recomendo procurar por  outras soluções que fazem sentido para o seus dados e que podem ser  adicionadas ao seu Pipeline.

Qualquer dúvida que tenha sobre essas estratégias apresentadas é só nos procurar no Discord e esse é o [link](https://github.com/alura-cursos/covid-19-clinical-2/blob/main/Modulo_6_e_Saiba_Mais.ipynb) para o código completo.

# Para saber mais: Salvando seu Modelo

Depois de treinado seu modelo pode ser salvo, e com isso poderá  utilizar ele em outro projeto, por exemplo um projeto web. Existem  diversos caminhos para isso, como pode ver na [documentação](https://scikit-learn.org/stable/modules/model_persistence.html).

Hoje vou apresentar um exemplo usando a biblioteca [joblib](https://joblib.readthedocs.io/en/latest/)

```python
from sklearn.tree import DecisionTreeClassifier

modelo_arvore = DecisionTreeClassifier()
modelo_arvore.fit(x, y)

from joblib import dump, load
dump(modelo_arvore, 'filename.joblib')
```

Utilizamos a função **dump**, colocamos o primeiro parâmetro com a variável que contem nosso modelo treinado, no caso a **modelo_arvore** e depois o nome do arquivo que será gerado e guardará nosso modelo.

E quando estiver em um novo projeto e quiser usar aquele modelo já treinado poderá usar a função **load**

```python
modelo_arvore = load('filename.joblib')
```

Qualquer dúvida que tenha sobre essas estratégias apresentadas é só nos procurar no Discord.