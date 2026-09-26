# SCIC — Sistema de Comunicação Interplanetária da Colônia - FASE 6

Protótipo desenvolvido por Eduardo Alves e Lisandra Araujo para a colônia **Aurora Siger**, como atividade integradora.

## Objetivo do projeto

O SCIC organiza os dados operacionais e de comunicação dos módulos da colônia
(habitação, agricultura, comunicação, laboratório, suporte médico e armazenamento
de dados), calcula indicadores e erros numéricos entre latência prevista e
observada, prioriza alertas críticos usando uma estrutura de heap, permite busca
rápida por prefixo com uma trie, e simula uma previsão simples de latência com
regressão linear, apoiando a tomada de decisão da equipe de operação.

## Arquivos da entrega

- `codigo_fonte.py` — arquivo principal do sistema, com todo o código em Python;
- `dados_aurora_siger.json` — base de dados simulada da Aurora Siger, usada pelo sistema;
- `relatorio_tecnico.pdf` — explicação técnica completa do projeto (contexto, dados, erros, modelo, heap, trie, gerenciamento inteligente da comunicação, reflexão social/cultural/sustentável, limitações e melhorias);
- `README.md` — este arquivo;
- `link_video.txt` — link do vídeo de apresentação do projeto, publicado no YouTube como "Não listado".

## Dependências

O projeto foi desenvolvido em **Python 3**, utilizando as seguintes bibliotecas:

- `pandas` — leitura, organização e manipulação dos dados;
- `scikit-learn` — modelo de regressão linear e métricas de avaliação (MAE, MSE, RMSE, R²);
- `json` e `heapq` — bibliotecas padrão do Python, não exigem instalação.

Para instalar as dependências externas:

```bash
pip install pandas scikit-learn
```

## Modo de execução

1. Certifique-se de que os arquivos `codigo_fonte.py` e `dados_aurora_siger.json` estão na mesma pasta;
2. Instale as dependências listadas acima, caso ainda não estejam instaladas;
3. Execute o sistema pelo terminal:

```bash
python codigo_fonte.py
```

4. O sistema exibirá um menu com as seguintes opções:

```
1 - Ver resumo dos dados
2 - Consultar registros (por tipo/status)
3 - Calcular erros (absoluto e relativo)
4 - Priorizar alertas (heap)
5 - Buscar modulo/sensor por prefixo (trie)
6 - Rodar modelo de previsao de latencia
7 - Calcular potencia e converter codigo de sensor
8 - Analise final dos resultados
0 - Sair
```

5. Basta digitar o número da opção desejada e seguir as instruções exibidas no terminal.

## Observações

- Todos os dados utilizados são simulados, coerentes com o contexto da missão da Aurora Siger;
- O código contém comentários explicando as principais etapas de cada funcionalidade;
- Não é necessário nenhum hardware físico, sensor real, API externa ou conexão com a internet para executar o sistema.
