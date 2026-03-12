# 📋 Plano de Desenvolvimento — Alinhado ao Plano Mestre

> Documento operacional do repositório. A referência estratégica continua sendo `documentation/PLANO_MESTRE.md`, que permanece inalterado.

## 1. Objetivo

Este documento traduz o `PLANO_MESTRE.md` para o **estado real atual do projeto**, deixando explícito:

- o que já está implementado;
- o que está operacional no app atual;
- o que segue como evolução futura.

## 2. Princípios de alinhamento

1. `PLANO_MESTRE.md` é o documento autoritativo de direção.
2. A documentação operacional deve refletir o repositório como ele existe hoje.
3. Direção conceitual e implementação corrente não devem ser confundidas.
4. Toda atualização documental deve permanecer coerente com `app/`, `tests/` e `streamlit_app.py`.

## 3. Estado atual do projeto

### 3.1 Implementado

- aplicação Streamlit com páginas de **Predição Individual**, **Análise da Turma** e **Sobre o Projeto**;
- cálculo de **INDE dinâmico por fase**;
- consolidação em três dimensões:
  - `dim_academica`
  - `dim_psicossocial`
  - `dim_psicopedagogica`
- extração de sinais textuais das observações:
  - `sent_score`
  - `sent_len`
- predição com **probabilidade calibrada**;
- fallback heurístico para contingência quando o artefato do modelo não fica utilizável;
- deploy preparado via `streamlit_app.py`.

### 3.2 Operacional no app

- formulário individual com `IAN`, `IDA`, `IEG`, `IAA`, `IPS`, `IPP`, `IPV`, `Fase`, pedras, número de avaliações e observação;
- upload em lote com derivação automática de dimensões e INDE;
- classificação em risco **baixo**, **médio** e **alto**;
- recomendações e leitura visual do risco;
- uso do threshold vindo do metadado do modelo.

### 3.3 Em evolução

- aprofundar a leitura longitudinal prevista no plano mestre;
- expandir NLP para representações textuais mais ricas, se validado;
- evoluir a frente preditiva sem romper a interface gerencial atual;
- consolidar materiais finais de apresentação e vídeo.

## 4. Estrutura corrente do repositório

```text
passos_magicos/
├── app/
│   ├── app.py
│   ├── risk_calibration.py
│   ├── ui_helpers.py
│   ├── modelo_risco_clean.pkl
│   ├── scaler_clean.pkl
│   └── modelo_meta_clean.json
├── data/
├── documentation/
│   ├── PLANO_MESTRE.md
│   ├── PLANO_DESENVOLVIMENTO.md
│   └── INDICE_DATATHON.md
├── notebooks/
├── tests/
├── streamlit_app.py
├── requirements.txt
└── requirements-dev.txt
```

## 5. Tradução dos eixos do plano mestre

### Eixo 1 — Engenharia de dados e harmonização

**Direção no plano mestre:** estruturar a leitura do PEDE e sustentar análises reaplicáveis.

**Estado atual:**

- notebooks e base tratada organizados no repositório;
- figuras analíticas já disponíveis em `data/`;
- cálculo do INDE dinâmico incorporado ao app.

### Eixo 2 — NLP em dados não estruturados

**Direção no plano mestre:** extrair valor das observações textuais.

**Estado atual:**

- pipeline lexical operacional no app;
- uso de `sent_score` e `sent_len` como sinais complementares;
- possibilidade de informar esses valores manualmente para simulação e revisão.

### Eixo 3 — Predição de risco educacional

**Direção no plano mestre:** usar IA como sentinela de risco.

**Estado atual:**

- modelo tabular serializado em `joblib` com features consolidadas;
- calibração de probabilidade aplicada antes da exibição do resultado;
- threshold lido de `modelo_meta_clean.json`;
- no artefato clean atual, o threshold é **0.30**.

**Nota de alinhamento:** o plano mestre enquadra essa frente dentro da trilha de Deep Learning; o estado operacional atual publicado utiliza um modelo probabilístico tabular calibrado, aderente ao código e aos artefatos presentes no repositório.

### Eixo 4 — Solução tecnológica e deploy

**Direção no plano mestre:** entregar ferramenta utilizável pela equipe.

**Estado atual:**

- `app/app.py` contém a implementação principal;
- `streamlit_app.py` funciona como entrypoint para o Streamlit Community Cloud;
- os fluxos individual e em lote estão alinhados com a lógica atual do projeto.

### Eixo 5 — Storytelling “A Jornada da Pedra”

**Direção no plano mestre:** transformar análise técnica em narrativa executiva.

**Estado atual:**

- perguntas do case já estão mapeadas documentalmente;
- notebooks e figuras sustentam a narrativa;
- apresentação final e vídeo ainda dependem de consolidação final.

## 6. Backlog executivo

### Concluído

- estrutura principal do repositório;
- app Streamlit funcional;
- deploy com entrypoint na raiz;
- calibração de probabilidade;
- melhorias recentes de UX na predição individual;
- testes automatizados para calibração e helpers de UI;
- atualização de `README.md` e `datathon.md`.

### Em andamento

- alinhamento completo da documentação ao plano mestre;
- revisão final do pacote executivo do Datathon.

### Pendente

- consolidar apresentação storytelling;
- consolidar vídeo final;
- revisar pacote final antes do fechamento.

## 7. Validações recentes

Verificações já executadas no estado atual do código:

- `python -m py_compile app/app.py app/ui_helpers.py`
- `python -m pytest tests/test_risk_calibration.py tests/test_ui_helpers.py`
- `python -m pytest tests -q`

Resultado mais recente conhecido:

- **7 testes passando**.

## 8. Regras para próximas atualizações

- não editar `PLANO_MESTRE.md` ao atualizar a documentação operacional;
- descrever sempre o estado real do repositório;
- não tratar artefatos legados como padrão atual;
- manter coerência entre documentação, interface do app e artefatos em `app/`.