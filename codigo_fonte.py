"""
SCIC - Sistema de Comunicacao Interplanetaria da Colonia
Aurora Siger

Este e o arquivo principal do sistema. Ele le a base de dados simulada
(dados_aurora_siger.json), calcula indicadores e erros, prioriza alertas
com heap, permite busca por prefixo com trie, monta um modelo simples de
previsao de latencia e exibe tudo em um menu no terminal.

ESTE ARQUIVO E UM ESQUELETO: as funcoes estao estruturadas e comentadas,
mas a logica interna de cada uma precisa ser implementada.
Procure por "TODO" para saber onde completar o codigo.
"""

import json
import heapq
import pandas as pd

# bibliotecas usadas apenas quando o modelo de previsao for implementado
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CAMINHO_DADOS = "dados_aurora_siger.json"


# ---------------------------------------------------------------------------
# 1. CARREGAMENTO E ORGANIZACAO DOS DADOS
# ---------------------------------------------------------------------------

def carregar_dados(caminho=CAMINHO_DADOS):
    """
    Le o arquivo JSON com os registros da Aurora Siger e retorna
    um DataFrame do Pandas.

    TODO:
    - abrir o arquivo com json.load()
    - converter a lista de dicionarios em DataFrame (pd.DataFrame)
    - retornar o DataFrame
    """
    with open(caminho, "r", encoding="utf-8") as f:
        registros = json.load(f)
    df = pd.DataFrame(registros)
    return df


def resumo_dados(df):
    """
    Mostra um resumo geral da base: quantidade de registros, modulos
    unicos, ciclos disponiveis e contagem por status.

    TODO:
    - imprimir df.shape, df['tipo_modulo'].value_counts(), etc.
    - imprimir quantos registros estao em cada status (ativo/manutencao/alerta)
    """
    print(f"Total de registros: {len(df)}")
    print(f"Modulos unicos: {df['nome_modulo'].nunique()}")
    print(f"Ciclos disponiveis: {sorted(df['ciclo'].unique())}")
    print("\nRegistros por status:")
    print(df["status"].value_counts())


def consultar_registros(df, tipo_modulo=None, status=None):
    """
    Filtra e retorna registros da base de acordo com tipo de modulo
    e/ou status informados.

    TODO:
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
# 2. ANALISE NUMERICA - ERRO ABSOLUTO E ERRO RELATIVO
# ---------------------------------------------------------------------------

def calcular_erros(df):
    """
    Calcula erro absoluto e erro relativo entre latencia_prevista_ms
    e latencia_observada_ms, adicionando duas novas colunas ao DataFrame:
    'erro_absoluto' e 'erro_relativo'.

    erro_absoluto = |observada - prevista|
    erro_relativo = erro_absoluto / prevista

    TODO:
    - implementar o calculo vetorizado usando as colunas do DataFrame
    - tratar possivel divisao por zero (prevista == 0), se houver
    - retornar o DataFrame com as colunas novas
    """
    df = df.copy()
    df["erro_absoluto"] = (df["latencia_observada_ms"] - df["latencia_prevista_ms"]).abs()
    df["erro_relativo"] = df["erro_absoluto"] / df["latencia_prevista_ms"]
    return df


def interpretar_erros(df):
    """
    Imprime uma interpretacao textual dos erros calculados: media,
    maior erro, modulo com maior erro relativo, e uma discussao sobre
    quando o erro pode ser considerado aceitavel.

    TODO:
    - usar df['erro_absoluto'].mean(), df['erro_relativo'].mean()
    - identificar o registro com maior erro relativo (idxmax)
    - escrever um comentario/print explicando o que isso significa
      no contexto da colonia (ex: erro relativo > 0.30 pode indicar
      falha real, nao apenas arredondamento)
    """
    pass  # TODO


# ---------------------------------------------------------------------------
# 3. HEAP DE PRIORIZACAO DE ALERTAS
# ---------------------------------------------------------------------------

def montar_heap_alertas(df):
    """
    Cria um heap (fila de prioridade) com os alertas da base.
    Cada item do heap deve ser uma tupla, por exemplo:
        (-prioridade, nome_modulo, mensagem_alerta, erro_relativo)
    Usamos -prioridade porque heapq do Python e um min-heap, e queremos
    que a MAIOR prioridade saia primeiro.

    Considere incluir apenas registros com status 'alerta' ou 'manutencao'.

    TODO:
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


def proximo_alerta_urgente(heap):
    """
    Remove e retorna o alerta mais urgente do heap (heapq.heappop).

    TODO:
    - usar heapq.heappop(heap) e devolver o item removido
    - tratar o caso do heap estar vazio
    """
    if not heap:
        return None
    return heapq.heappop(heap)


# ---------------------------------------------------------------------------
# 4. TRIE DE BUSCA POR PREFIXO
# ---------------------------------------------------------------------------

class NoTrie:
    """Um no da trie: guarda os filhos e se marca o fim de uma palavra."""
    def __init__(self):
        self.filhos = {}
        self.fim_de_palavra = False


class Trie:
    """
    Estrutura de trie para busca por prefixo em nomes de modulos ou
    codigos de sensores.
    """
    def __init__(self):
        self.raiz = NoTrie()

    def inserir(self, palavra):
        """
        Insere uma palavra na trie, caractere por caractere.

        TODO:
        - percorrer a palavra, criando nos filhos quando necessario
        - marcar fim_de_palavra = True no ultimo no
        """
        no = self.raiz
        for caractere in palavra:
            if caractere not in no.filhos:
                no.filhos[caractere] = NoTrie()
            no = no.filhos[caractere]
        no.fim_de_palavra = True

    def _coletar_palavras(self, no, prefixo_atual, resultados):
        """Funcao auxiliar: percorre a trie a partir de um no coletando palavras."""
        if no.fim_de_palavra:
            resultados.append(prefixo_atual)
        for caractere, proximo_no in no.filhos.items():
            self._coletar_palavras(proximo_no, prefixo_atual + caractere, resultados)

    def buscar_prefixo(self, prefixo):
        """
        Retorna todas as palavras inseridas na trie que comecam com
        o prefixo informado.

        TODO:
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


def montar_trie_modulos(df):
    """
    Cria e popula uma trie com os nomes de modulo e codigos de sensor
    presentes na base.

    TODO:
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
# 5. MODELO SIMPLES DE PREVISAO + METRICAS DE AVALIACAO
# ---------------------------------------------------------------------------

def preparar_dados_modelo(df):
    """
    Prepara variaveis (X) e alvo (y) para o modelo de previsao de
    latencia observada, usando o numero do ciclo como variavel preditora.

    Sugestao simples: X = numero do ciclo (extraido de 'ciclo_01' -> 1),
    y = latencia_observada_ms.

    TODO:
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


def treinar_avaliar_modelo(X, y):
    """
    Divide os dados em treino/teste, treina uma regressao linear simples
    e calcula MAE, MSE, RMSE e R2.

    TODO:
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

    # TODO: escrever a interpretacao textual das metricas (ver item 1.3 do edital)

    return modelo, metricas


# ---------------------------------------------------------------------------
# 6. BASES NUMERICAS E ELETRICIDADE BASICA
# ---------------------------------------------------------------------------

def calcular_potencia(tensao_v, corrente_a):
    """
    Calcula a potencia aproximada a partir de tensao e corrente (P = V * I).
    """
    return round(tensao_v * corrente_a, 3)


def converter_codigo_sensor(codigo_hex):
    """
    Recebe um codigo de sensor em hexadecimal (ex: '0x3C09') e retorna
    sua representacao em decimal e binario.

    TODO:
    - usar int(codigo_hex, 16) para converter para decimal
    - usar bin() para converter o valor decimal para binario
    - retornar (valor_decimal, valor_binario)
    """
    valor_decimal = int(codigo_hex, 16)
    valor_binario = bin(valor_decimal)
    return valor_decimal, valor_binario


# ---------------------------------------------------------------------------
# 7. MENU PRINCIPAL
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
            # TODO: montar uma analise final que junte erros + alertas + modelo
            print("TODO: implementar analise final consolidada")

        elif escolha == "0":
            print("Encerrando o SCIC. Ate a proxima transmissao.")
            break

        else:
            print("Opcao invalida, tente novamente.")


if __name__ == "__main__":
    menu()
