"""
SCIC - Sistema de Comunicacao Interplanetaria da Colonia Aurora Siger
Fase 6
"""

import json
import heapq
import pandas as pd

# bibliotecas usadas apenas quando o modelo de previsao for implementado
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

path_dados = "dados_aurora_siger.json"


# ---------------------------------------------------------------------------
# 1. Carregamento e organizaçao dos dados
# ---------------------------------------------------------------------------

def carregar_dados(caminho=path_dados): # le o json com os dados da colonia e retorna um dataframe do Pandas
"""
to do:
- abrir o arquivo com json.load()
- converter a lista de dicionarios em DataFrame (pd.DataFrame)
- retornar o DataFrame
"""
    with open(caminho, "r", encoding="utf-8") as f:
        registros = json.load(f)
    df = pd.DataFrame(registros)
    return df


def resumo_dados(df): # mostra um resumo geral da base
"""
to do:
- imprimir df.shape, df['tipo_modulo'].value_counts(), etc.
- imprimir quantos registros estao em cada status (ativo/manutencao/alerta)
"""
    print(f"Total de registros: {len(df)}")
    print(f"Modulos unicos: {df['nome_modulo'].nunique()}")
    print(f"Ciclos disponiveis: {sorted(df['ciclo'].unique())}")
    print("\nRegistros por status:")
    print(df["status"].value_counts())


def consultar_registros(df, tipo_modulo=None, status=None): # filtra e retorna registros da base de acordo com tipo de modulo / status informado
"""
to do:
- aplicar filtros no DataFrame conforme os parametros informados
- retornar o DataFrame filtrado
"""
    resultado = df.copy()
    if tipo_modulo:
        resultado = resultado[resultado["tipo_modulo"] == tipo_modulo]
    if status:
        resultado = resultado[resultado["status"] == status]
    return resultado


# ---------------------------------------------------------------------------
# 2. Analise numerica: erro absoluto e erro relativo
# ---------------------------------------------------------------------------

def calcular_erros(df): #calcula o erro absoluto e o erro relativo entre as latencias e add duas novas colunas ao df
"""
to do:
- implementar o calculo vetorizado usando as colunas do DataFrame
- tratar possivel divisao por zero (prevista == 0), se houver
- retornar o DataFrame com as colunas novas
"""
    df = df.copy()
    df["erro_absoluto"] = (df["latencia_observada_ms"] - df["latencia_prevista_ms"]).abs()
    df["erro_relativo"] = df["erro_absoluto"] / df["latencia_prevista_ms"]
    return df


def interpretar_erros(df): # imprime uma interpretaçao textual dos erros calculados e uma discussao sobre quando o erro pode ser considerado aceitavel
"""
  to do:
- usar df['erro_absoluto'].mean(), df['erro_relativo'].mean()
- identificar o registro com maior erro relativo (idxmax)
- escrever um comentario/print explicando o que isso significa no contexto da colonia (ex: erro relativo > 0.30 pode indicar falha real, nao apenas arredondamento)
"""
    pass  # TODO


# ---------------------------------------------------------------------------
# 3. Heap de priorizaçao de alertas
# ---------------------------------------------------------------------------

def montar_heap_alertas(df): # cria um heap com os alertas da base, e queremos que a maior prioridade saia primeiro
"""
to do:
- filtrar df para status in ['alerta', 'manutencao']
- construir a lista de tuplas
- transformar em heap com heapq.heapify()
- retornar o heap (lista)
"""
    alertas = df[df["status"].isin(["alerta", "manutencao"])]
    heap = []
    for _, linha in alertas.iterrows():
        item = (-linha["prioridade"], linha["nome_modulo"], linha["mensagem_alerta"])
        heap.append(item)
    heapq.heapify(heap)
    return heap


def proximo_alerta_urgente(heap): # remove e retorna o alerta mais urgente do heap
    """
TODO:
    - usar heapq.heappop(heap) e devolver o item removido
    - tratar o caso do heap estar vazio
    """
    if not heap:
        return None
    return heapq.heappop(heap)


# ---------------------------------------------------------------------------
# 4. Trie de busca por prefixo
# ---------------------------------------------------------------------------

class NoTrie: # guarda os filhos e se marca o fim de uma palavra
    def __init__(self):
        self.filhos = {}
        self.fim_de_palavra = False


class Trie: # busca por prefixo em nomes de modulos ou codigos de sensores
    def __init__(self):
        self.raiz = NoTrie()
 
    def inserir(self, palavra): # insere caractere por caractere de uma palavra no trie
"""
to do:
- percorrer a palavra, criando nos filhos quando necessario
- marcar fim_de_palavra = True no ultimo no
"""
        no = self.raiz
        for caractere in palavra:
            if caractere not in no.filhos:
                no.filhos[caractere] = NoTrie()
            no = no.filhos[caractere]
        no.fim_de_palavra = True

    def _coletar_palavras(self, no, prefixo_atual, resultados): # percorre a trie a partir de um no coletando palavras
        if no.fim_de_palavra:
            resultados.append(prefixo_atual)
        for caractere, proximo_no in no.filhos.items():
            self._coletar_palavras(proximo_no, prefixo_atual + caractere, resultados)

    def buscar_prefixo(self, prefixo): # retorna todas as palavras inseridas na trie que começam com o prefixo informado
"""
to do :
- navegar pela trie ate o no correspondente ao prefixo
- se o prefixo nao existir, retornar lista vazia
- se existir, usar _coletar_palavras para juntar todas as palavras
"""
        no = self.raiz
        for caractere in prefixo:
            if caractere not in no.filhos:
                return []
            no = no.filhos[caractere]
        resultados = []
        self._coletar_palavras(no, prefixo, resultados)
        return resultados


def montar_trie_modulos(df): # cria e popula uma trie com os nomes de modulo e codigos de sensor
"""
to do 
- criar uma Trie()
- inserir cada nome_modulo (df['nome_modulo'].unique())
- inserir cada codigo_sensor (df['codigo_sensor'].unique())
- retornar a trie
"""
    trie = Trie()
    for nome in df["nome_modulo"].unique():
        trie.inserir(nome)
    for codigo in df["codigo_sensor"].unique():
        trie.inserir(codigo)
    return trie


# ---------------------------------------------------------------------------
# 5. Modelo de previsão e métricas de avaliaçao
# ---------------------------------------------------------------------------

def preparar_dados_modelo(df): # prepara variaveis e alvo para o modelo de previsao de latencia observada, usando o n° do ciclo como variavel preditora
"""
to do:
- criar uma coluna numerica a partir de 'ciclo' (ex: int(ciclo[-2:]))
- definir X (ex: numero do ciclo, e opcionalmente tipo_modulo com one-hot)
- definir y = latencia_observada_ms
- retornar X, y
"""
    df = df.copy()
    df["numero_ciclo"] = df["ciclo"].str.extract(r"(\d+)").astype(int)
    X = df[["numero_ciclo"]]
    y = df["latencia_observada_ms"]
    return X, y


def treinar_avaliar_modelo(X, y): # divide os dados em treino, treina uma regressao linear simples e calcula MAE, MSE e RMSE e R2
"""
to do:
- usar train_test_split(X, y, test_size=0.2, random_state=42)
- treinar LinearRegression()
- prever no conjunto de teste
- calcular mae, mse, rmse (rmse = mse ** 0.5), r2
- imprimir os resultados com interpretacao
- retornar o modelo treinado e o dicionario de metricas
"""
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    modelo = LinearRegression()
    modelo.fit(X_treino, y_treino)
    y_previsto = modelo.predict(X_teste)

    mae = mean_absolute_error(y_teste, y_previsto)
    mse = mean_squared_error(y_teste, y_previsto)
    rmse = mse ** 0.5
    r2 = r2_score(y_teste, y_previsto)

    metricas = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

    print("Metricas do modelo de previsao de latencia:")
    for nome, valor in metricas.items():
        print(f"  {nome}: {valor:.4f}")

    # to do : escrever a interpretacao textual das metricas 

    return modelo, metricas


# ---------------------------------------------------------------------------
# 6. Bases numericas e eletricidade
# ---------------------------------------------------------------------------

def calcular_potencia(tensao_v, corrente_a): # calcula a potencia aproximada a partir de tensao e corrente
    return round(tensao_v * corrente_a, 3)


def converter_codigo_sensor(codigo_hex): # recebe um codigo de sensor em hexadecimal e retorna sua representacao em decimal e binario
"""
to do :
- usar int(codigo_hex, 16) para converter para decimal
- usar bin() para converter o valor decimal para binario
- retornar (valor_decimal, valor_binario)
"""
    valor_decimal = int(codigo_hex, 16)
    valor_binario = bin(valor_decimal)
    return valor_decimal, valor_binario


# ---------------------------------------------------------------------------
# 7. Menu principal 
# ---------------------------------------------------------------------------

def menu():
    df = carregar_dados()

    opcoes = """
========================================
 SCIC - Sistema de Comunicacao Interplanetaria
 Colonia Aurora Siger
========================================
1 - Ver resumo dos dados
2 - Consultar registros (por tipo/status)
3 - Calcular erros (absoluto e relativo)
4 - Priorizar alertas (heap)
5 - Buscar modulo/sensor por prefixo (trie)
6 - Rodar modelo de previsao de latencia
7 - Calcular potencia e converter codigo de sensor
8 - Analise final dos resultados
0 - Sair
========================================
"""

    while True:
        print(opcoes)
        escolha = input("Escolha uma opcao: ").strip()

        if escolha == "1":
            resumo_dados(df)

        elif escolha == "2":
            tipo = input("Tipo do modulo (enter para ignorar): ").strip() or None
            status = input("Status (enter para ignorar): ").strip() or None
            resultado = consultar_registros(df, tipo_modulo=tipo, status=status)
            print(resultado)

        elif escolha == "3":
            df_com_erros = calcular_erros(df)
            print(df_com_erros[["nome_modulo", "ciclo", "latencia_prevista_ms",
                                 "latencia_observada_ms", "erro_absoluto", "erro_relativo"]])
            interpretar_erros(df_com_erros)

        elif escolha == "4":
            heap = montar_heap_alertas(df)
            print(f"Total de alertas no heap: {len(heap)}")
            alerta = proximo_alerta_urgente(heap)
            print("Alerta mais urgente:", alerta)
            # TODO: opcionalmente, mostrar mais de um alerta em sequencia

        elif escolha == "5":
            trie = montar_trie_modulos(df)
            prefixo = input("Digite o prefixo de busca: ").strip()
            resultados = trie.buscar_prefixo(prefixo)
            print("Resultados encontrados:", resultados)

        elif escolha == "6":
            X, y = preparar_dados_modelo(df)
            treinar_avaliar_modelo(X, y)

        elif escolha == "7":
            tensao = float(input("Tensao (V): "))
            corrente = float(input("Corrente (A): "))
            print("Potencia calculada:", calcular_potencia(tensao, corrente), "W")
            codigo = input("Codigo do sensor (ex: 0x3C09): ").strip()
            decimal, binario = converter_codigo_sensor(codigo)
            print(f"Decimal: {decimal} | Binario: {binario}")

        elif escolha == "8":
            # to do: montar uma analise final que junte erros + alertas + modelo
            print("TODO: implementar analise final consolidada")

        elif escolha == "0":
            print("Encerrando o SCIC. Ate a proxima transmissao.")
            break

        else:
            print("Opcao invalida, tente novamente.")


if __name__ == "__main__":
    menu()
