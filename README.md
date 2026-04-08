# PID Tuner Pro v8

Ferramenta Python para diagnóstico de malhas, limpeza robusta de outliers, modelagem de processo, comparação de sintonia e geração de relatório Word.

## Recursos principais
- 5 métodos de outlier + modo Auto
- intensidade de limpeza parametrizável
- 1 a 3 passadas de limpeza
- interpolar ou excluir pontos sinalizados
- FOPDT, SOPDT, ARX(2,2) e ARX(3,3)
- PV real x PV prevista
- comparação gráfica da PV prevista por método de sintonia
- teste manual de Kc, Ti e Td
- inclusão do PID atual da planta
- relatório Word para uso em campo

## Execução
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Arquivos importantes
- `app.py`: interface Streamlit
- `robust_processing.py`: limpeza, modelagem, score e preparação da análise
- `pid_core.py`: regras de sintonia e simulação fechada
- `reporting.py`: geração do relatório Word
- `PASSO_A_PASSO_TESTE_v8.md`: roteiro rápido de uso
