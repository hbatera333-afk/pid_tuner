# Passo a passo para testar o PID Tuner Pro v7

1. Extraia o arquivo ZIP.
2. Abra um terminal dentro da pasta `pid_tuner_field_pro_v7`.
3. Instale as dependências:

```bash
pip install -r requirements.txt
```

4. Inicie o software:

```bash
streamlit run app.py
```

5. No navegador, carregue seu arquivo Excel ou CSV.
6. Mapeie as colunas:
   - Tempo
   - PV
   - SP
   - MV
7. Preencha cliente, planta, nome da malha e tag.
8. Escolha o tipo da malha e se a proposta será PI ou PID.
9. Em outliers, faça assim no primeiro teste:
   - Método: `Auto`
   - Tratamento: `Interpolar`
   - Intensidade: `1.2`
   - Passadas: `2`
   - Aplicar em: `PV e MV`
10. Clique em `Rodar análise`.
11. Revise, nesta ordem:
   - nota da malha
   - comparação dos filtros de outliers
   - comparação dos modelos
   - PV real vs PV prevista na validação
   - parâmetros equivalentes do processo
   - tabela de sintonias
   - gráfico da PV prevista por método de sintonia
12. Se ainda houver muitos outliers, repita a análise mudando:
   - Intensidade para `1.6`, `2.0` ou `2.4`
   - Passadas para `3`
   - Método específico: `Hampel`, `Rolling IQR` ou `MAD global`
   - Tratamento para `Excluir`
13. Gere o relatório Word.

## Interpretação rápida

- `Kc`: ganho do controlador
- `Ti (s)`: tempo integral em segundos
- `Ki (1/s)`: ganho integral equivalente
- `Td (s)`: tempo derivativo em segundos
- `PB equivalente (%)`: só é válido quando PV e MV estão normalizados em %

## Ajuste recomendado para primeira tentativa

Comece com o método `IMC/SIMC robusto` e só depois teste os mais agressivos.
