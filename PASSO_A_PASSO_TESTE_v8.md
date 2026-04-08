# Passo a passo de teste — PID Tuner Pro v8

## 1. Instalação

No terminal:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 2. Carregar o histórico

Na tela inicial:
- clique em **Carregue um CSV ou Excel**
- selecione o arquivo da malha
- confira a prévia dos dados

## 3. Mapear as colunas

Escolha:
- **Tempo**
- **PV**
- **SP**
- **MV**

Observação:
- para a versão robusta de modelagem e sintonia, o ideal é ter **SP, PV e MV** medidos

## 4. Preencher os metadados da visita

Informe:
- cliente
- planta
- nome da malha
- tag
- tipo de malha
- tipo de controlador proposto: **PI** ou **PID**

## 5. Configurar a limpeza de outliers

Comece assim:
- método: **Auto**
- reamostragem: **60s**
- aplicar em: **PV e MV**
- tratamento: **Interpolar**
- intensidade: **1.6 a 1.8**
- passadas: **3**

Se ainda houver muitos picos:
- aumente a intensidade para **2.0**, **2.2** ou **2.5**
- compare novamente os gráficos

## 6. Rodar a análise

Clique em **Rodar análise**.

Depois verifique, nesta ordem:
- **Resumo executivo**
- **Qualidade do dado**
- **Modelagem**
- **Sintonia e simulação**

## 7. Conferir a qualidade dos dados

Na aba **Qualidade do dado**, confira:
- PV bruta x PV limpa
- pontos sinalizados como outlier
- tabela comparando os 5 métodos de limpeza

Se a PV limpa ainda estiver ruim:
- aumente a intensidade
- troque o método de outlier manualmente
- rode novamente

## 8. Conferir a modelagem

Na aba **Modelagem**, veja:
- ranking dos modelos
- **PV real x PV prevista** de cada modelo
- ganho do processo
- tau
- tempo morto
- convenção do erro recomendada

Use preferencialmente:
- o melhor modelo preditivo para validar qualidade
- o modelo equivalente mostrado pelo sistema para a sintonia

## 9. Conferir as sintonias sugeridas

Na aba **Sintonia e simulação**, veja:
- tabela com **Kc, Ti, Ki, Td**
- gráfico principal: **como a PV deve se comportar para cada sintonia**
- gráfico de **MV prevista**
- gráfico de **erro previsto**
- tabela com **overshoot, settling time, IAE, ISE, ITAE e TV(MV)**

## 10. Testar um PID atual da planta

Se você tiver o PID atual:
- marque **Incluir PID atual na comparação**
- escolha a forma:
  - **Kc, Ti, Td**
  - ou **Kc, Ki, Kd**
- informe os valores

O sistema colocará a curva do PID atual junto com as demais.

## 11. Testar uma sintonia manual

Na seção **Teste manual de sintonia**:
- informe o nome do teste
- escolha **PI** ou **PID**
- informe **Kc, Ti e Td**
- clique em **Adicionar/atualizar teste manual**

Depois compare a nova curva com as outras.

## 12. Gerar relatório Word

Na aba **Relatório**:
- clique em **Gerar relatório desta malha**
- ou em **Gerar relatório do portfólio**

Use **Adicionar ao portfólio** quando quiser montar um ranking de várias malhas da visita.

## 13. Ordem prática recomendada em campo

Para uma primeira rodada:
- comece pelo método mais robusto
- valide o sentido de ação
- confira o esforço da MV
- depois teste a sintonia intermediária
- só teste as mais agressivas se a malha estiver segura e limpa

## 14. Recomendação para a sua visita

Para começar com segurança:
- outlier: **Auto**
- intensidade: **1.8**
- passadas: **3**
- tratamento: **Interpolar**
- aplicar em: **PV e MV**
- modelo: **Auto**
- sintonia: **Comparar todos**

Depois refine malha a malha.
