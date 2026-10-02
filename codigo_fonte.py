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
    with open(caminho, "r", encoding="utf-8") as f:
        registros = json.load(f)
    df = pd.DataFrame(registros)
    return df

def cadastrar_registro(df):
    """
    Permite cadastrar manualmente um novo registro no sistema, a partir
    de dados digitados pelo usuario, e retorna o DataFrame atualizado.
    """
    print("\nCadastro de novo registro:")
    novo_id = int(df["id"].max()) + 1 if len(df) > 0 else 1
    nome_modulo = input("Nome do modulo: ").strip()
    tipo_modulo = input("Tipo do modulo (habitacao/agricultura/comunicacao/"
                         "laboratorio/suporte_medico/armazenamento_dados): ").strip()
    ciclo = input("Ciclo (ex: ciclo_05): ").strip()
    latencia_observada = float(input("Latencia observada (ms): "))
    latencia_prevista = float(input("Latencia prevista (ms): "))
    tensao = float(input("Tensao (V): "))
    corrente = float(input("Corrente (A): "))
    potencia = calcular_potencia(tensao, corrente)
    status = input("Status (ativo/manutencao/alerta): ").strip()
    prioridade = int(input("Prioridade (1 a 5): "))
    codigo_sensor = input("Codigo do sensor (ex: 0x3C1F): ").strip()
    mensagem_alerta = input("Mensagem de alerta: ").strip()
 
    novo_registro = {
        "id": novo_id,
        "nome_modulo": nome_modulo,
        "tipo_modulo": tipo_modulo,
        "ciclo": ciclo,
        "latencia_observada_ms": latencia_observada,
        "latencia_prevista_ms": latencia_prevista,
        "tensao_v": tensao,
        "corrente_a": corrente,
        "potencia_w": potencia,
        "status": status,
        "prioridade": prioridade,
        "codigo_sensor": codigo_sensor,
        "mensagem_alerta": mensagem_alerta,
    }
 
    df = pd.concat([df, pd.DataFrame([novo_registro])], ignore_index=True)
    print(f"\nRegistro cadastrado com sucesso (id {novo_id}).")
    return df
 
def resumo_dados(df): # mostra um resumo geral da base
    print(f"Total de registros: {len(df)}")
    print(f"Modulos unicos: {df['nome_modulo'].nunique()}")
    print(f"Ciclos disponiveis: {sorted(df['ciclo'].unique())}")
    print("\nRegistros por status:")
    print(df["status"].value_counts())


def consultar_registros(df, tipo_modulo=None, status=None): # filtra e retorna registros da base de acordo com tipo de modulo / status informado
    resultado = df.copy()
    if tipo_modulo:
        resultado = resultado[resultado["tipo_modulo"] == tipo_modulo]
    if status:
        resultado = resultado[resultado["status"] == status]
    return resultado

def exibir_registros(df_resultado): # exibe os registros filtrados de forma legivel no terminal
    if df_resultado.empty:
        print("\nNenhum registro encontrado com os filtros informados.")
        return
 
    print(f"\n{len(df_resultado)} registro(s) encontrado(s):\n")
    for _, linha in df_resultado.iterrows():
        print(f"- [{linha['ciclo']}] {linha['nome_modulo']} ({linha['tipo_modulo']}) "
              f"| status: {linha['status']} | prioridade: {linha['prioridade']} "
              f"| latencia prevista: {linha['latencia_prevista_ms']} ms "
              f"| latencia observada: {linha['latencia_observada_ms']} ms "
              f"| sensor: {linha['codigo_sensor']}")


# ---------------------------------------------------------------------------
# 2. Analise numerica: erro absoluto e erro relativo
# ---------------------------------------------------------------------------

def calcular_erros(df): #calcula o erro absoluto e o erro relativo entre as latencias e add duas novas colunas ao DF
    df = df.copy()
    df["erro_absoluto"] = (df["latencia_observada_ms"] - df["latencia_prevista_ms"]).abs()
 
    # evita divisao por zero caso alguma latencia prevista seja 0
    df["erro_relativo"] = df.apply(
        lambda linha: linha["erro_absoluto"] / linha["latencia_prevista_ms"]
        if linha["latencia_prevista_ms"] != 0 else 0.0,
        axis=1
    )
    return df


def interpretar_erros(df): # imprime uma interpretaçao textual dos erros calculados e uma discussao sobre quando o erro pode ser considerado aceitavel

    media_absoluta = df["erro_absoluto"].mean()
    media_relativa = df["erro_relativo"].mean()
 
    indice_pior = df["erro_relativo"].idxmax()
    pior_registro = df.loc[indice_pior]
 
    print("\nInterpretacao dos erros:")
    print(f"  Erro absoluto medio: {media_absoluta:.2f} ms")
    print(f"  Erro relativo medio: {media_relativa * 100:.2f}%")
    print(f"  Maior erro relativo: {pior_registro['nome_modulo']} "
          f"({pior_registro['ciclo']}), {pior_registro['erro_relativo'] * 100:.2f}%")
 
    print("\n  Discussao: erros relativos abaixo de ~10% sao considerados dentro da")
    print("  margem esperada de arredondamento e precisao numerica dos sensores.")
    print("  Erros entre ~15% e 30% indicam necessidade de manutencao preventiva.")
    print("  Erros acima de ~30%, como o maior identificado acima, sao tratados como")
    print("  alerta critico, pois dificilmente sao explicados apenas por aproximacao")
    print("  numerica, indicando uma possivel falha real na comunicacao do modulo.")


# ---------------------------------------------------------------------------
# 3. Heap de priorizaçao de alertas
# ---------------------------------------------------------------------------

def montar_heap_alertas(df): # cria um heap com os alertas da base, e queremos que a maior prioridade saia primeiro
    alertas = df[df["status"].isin(["alerta", "manutencao"])]
    heap = []
    for _, linha in alertas.iterrows():
        item = (-linha["prioridade"], linha["nome_modulo"], linha["mensagem_alerta"])
        heap.append(item)
    heapq.heapify(heap)
    return heap


def proximo_alerta_urgente(heap): # remove e retorna o alerta mais urgente do heap
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
        no = self.raiz
        for caractere in prefixo:
            if caractere not in no.filhos:
                return []
            no = no.filhos[caractere]
        resultados = []
        self._coletar_palavras(no, prefixo, resultados)
        return resultados


def montar_trie_modulos(df): # cria e popula uma trie com os nomes de modulo e codigos de sensor
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
    df = df.copy()
    df["numero_ciclo"] = df["ciclo"].str.extract(r"(\d+)").astype(int)
    X = df[["numero_ciclo"]]
    y = df["latencia_observada_ms"]
    return X, y

def treinar_avaliar_modelo(X, y): # divide os dados em treino, treina uma regressao linear simples e calcula MAE, MSE e RMSE e R2
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
 
    print("\nMetricas do modelo de previsao de latencia:")
    for nome, valor in metricas.items():
        print(f"  {nome}: {valor:.4f}")
 
    print("\nInterpretacao das metricas:")
    print(f"  O modelo erra, em media, {mae:.2f} ms na previsao da latencia (MAE).")
    if rmse > mae * 1.3:
        print(f"  O RMSE ({rmse:.2f}) e consideravelmente maior que o MAE ({mae:.2f}),")
        print("  o que indica que existem alguns registros com erro de previsao bem")
        print("  maior que a media, normalmente ligados a modulos ja criticos na base.")
    else:
        print(f"  O RMSE ({rmse:.2f}) esta proximo do MAE ({mae:.2f}), sugerindo que")
        print("  os erros do modelo sao relativamente uniformes entre os registros.")
 
    if r2 >= 0.7:
        print(f"  O R2 de {r2:.2f} indica que o modelo explica boa parte da variacao")
        print("  da latencia observada, mas isso nao significa que ele seja perfeito")
        print("  para todos os modulos individualmente.")
    else:
        print(f"  O R2 de {r2:.2f} e baixo, indicando que o numero do ciclo sozinho")
        print("  nao explica bem a variacao da latencia — seria necessario incluir")
        print("  outras variaveis (como tipo_modulo ou tensao) para melhorar o modelo.")
 
    return modelo, metricas


# ---------------------------------------------------------------------------
# 6. Bases numericas e eletricidade
# ---------------------------------------------------------------------------

def calcular_potencia(tensao_v, corrente_a): # calcula a potencia aproximada a partir de tensao e corrente
    return round(tensao_v * corrente_a, 3)


def converter_codigo_sensor(codigo_hex): # recebe um codigo de sensor em hexadecimal e retorna sua representacao em decimal e binario
    valor_decimal = int(codigo_hex, 16)
    valor_binario = bin(valor_decimal)
    return valor_decimal, valor_binario

# ---------------------------------------------------------------------------
# 7. Análise final
# ---------------------------------------------------------------------------

def analise_final(df): # junta erros, heap de alertas e metricas do modelo em um resumo de apoio a decisao para a equipe da colonia
    print("\n========== ANALISE FINAL DOS RESULTADOS ==========")
 
    df_erros = calcular_erros(df)
    print(f"\nTotal de modulos monitorados: {df_erros['nome_modulo'].nunique()}")
    print(f"Erro relativo medio geral: {df_erros['erro_relativo'].mean() * 100:.2f}%")
 
    heap = montar_heap_alertas(df_erros)
    print(f"\nTotal de alertas/manutencoes pendentes: {len(heap)}")
    if heap:
        alerta_mais_urgente = heap[0]
        print(f"Alerta mais urgente no momento: {alerta_mais_urgente[1]} "
              f"— {alerta_mais_urgente[2]} (prioridade {-alerta_mais_urgente[0]})")
 
    X, y = preparar_dados_modelo(df_erros)
    print("\nDesempenho do modelo de previsao de latencia:")
    _, metricas = treinar_avaliar_modelo(X, y)
 
    print("\nConclusao:")
    if metricas["R2"] >= 0.7 and df_erros["erro_relativo"].mean() < 0.15:
        print("  A rede de comunicacao da Aurora Siger apresenta, de forma geral,")
        print("  erro controlado e previsibilidade razoavel, com poucos modulos")
        print("  exigindo atencao imediata.")
    else:
        print("  A rede de comunicacao da Aurora Siger apresenta erro relativo")
        print("  significativo e/ou baixa previsibilidade, recomendando-se priorizar")
        print("  os alertas no topo do heap e revisar os modulos mais criticos.")
    print("====================================================")

# ---------------------------------------------------------------------------
# 8. Menu principal 
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
3 - Cadastrar novo registro manualmente
4 - Calcular erros (absoluto e relativo)
5 - Priorizar alertas (heap)
6 - Buscar modulo/sensor por prefixo (trie)
7 - Rodar modelo de previsao de latencia
8 - Calcular potencia e converter codigo de sensor
9 - Analise final dos resultados
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
            exibir_registros(resultado)
 
        elif escolha == "3":
            df = cadastrar_registro(df)
 
        elif escolha == "4":
            df_com_erros = calcular_erros(df)
            exibir_registros(df_com_erros)
            interpretar_erros(df_com_erros)
 
        elif escolha == "5":
            heap = montar_heap_alertas(df)
            print(f"\nTotal de alertas no heap: {len(heap)}")
            alerta = proximo_alerta_urgente(heap)
            if alerta:
                print(f"Alerta mais urgente: {alerta[1]} — {alerta[2]} "
                      f"(prioridade {-alerta[0]})")
            else:
                print("Nenhum alerta pendente no momento.")
 
        elif escolha == "6":
            trie = montar_trie_modulos(df)
            prefixo = input("Digite o prefixo de busca: ").strip()
            resultados = trie.buscar_prefixo(prefixo)
            print("Resultados encontrados:", resultados if resultados else "nenhum")
 
        elif escolha == "7":
            X, y = preparar_dados_modelo(df)
            treinar_avaliar_modelo(X, y)
 
        elif escolha == "8":
            tensao = float(input("Tensao (V): "))
            corrente = float(input("Corrente (A): "))
            print("Potencia calculada:", calcular_potencia(tensao, corrente), "W")
            codigo = input("Codigo do sensor (ex: 0x3C09): ").strip()
            decimal, binario = converter_codigo_sensor(codigo)
            print(f"Decimal: {decimal} | Binario: {binario}")
 
        elif escolha == "9":
            analise_final(df)
 
        elif escolha == "0":
            print("Encerrando o SCIC. Ate a proxima transmissao.")
            break
 
        else:
            print("Opcao invalida, tente novamente.")
 
 
if __name__ == "__main__":
    menu()
