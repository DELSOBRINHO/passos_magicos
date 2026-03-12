# Plano Mestre: Solução de Inteligência Educacional Passos Mágicos

Este documento consolida o Plano de Execução mestre, alinhado aos PDFs presentes no repositório (`Dicionário Dados Datathon.pdf` e `PEDE_ Pontos importantes.pdf`). O objetivo é transformar o trabalho realizado na Fase 5 em uma solução sustentável e reaplicável para a Associação Passos Mágicos.

## 1. Engenharia de Dados e Harmonização (Base PEDE)

- Utilizar ciclos 2020–2023 para construir visão longitudinal dos alunos.
- Motor de Cálculo Dinâmico do INDE: lógica que ajusta automaticamente pesos dos indicadores (IAN, IDA, IEG, IAA, IPS, IPP, IPV) segundo a fase do aluno (Fases 0–7 vs. Fase 8), de acordo com as diretrizes do PEDE.
- Tratamento Longitudinal: cruzar dados anuais para identificar trajetórias de cada "Pedra" (evolução Quartzo → Topázio), tratar categóricas e normalizar notas.

## 2. Análise de Dados Não Estruturados (NLP)

- Extrair valor das observações textuais (campos de "Destaque"/recomendações).
- Embeddings de Sentimento: converter textos psicopedagógicos em vetores numéricos para análise e detecção de similaridades/associações.
- Detectar padrões silenciosos: correlações entre termos em IPS/IPP e quedas futuras de IDA.

## 3. Modelo Preditivo de Deep Learning (Previsão de Risco)

- Arquitetura: MLP robusto para prever probabilidade de risco ou regressão na classificação educacional.
- Inputs: dimensões Acadêmica, Psicossocial e Psicopedagógica (features consolidadas e normalizadas).
- Output: probabilidade 0–100% de risco; identificar variáveis chave para o Ponto de Virada (IPV).

## 4. Solução Tecnológica e Deploy (Streamlit)

- App web para uso diário da equipe: dashboards gerenciais e calculadora preditiva.
- Interface gerencial: evolução do INDE, efetividade por fase (Ágata, Ametista, etc.).
- Calculadora preditiva: usuário insere indicadores e obtém INDE calculado, probabilidade de risco e sugestões de intervenção.

## 5. Estratégia de Storytelling: "A Jornada da Pedra"

- Diagnóstico: perfil atual de defasagem e desempenho.
- Alavancas de mudança: como engajamento e psicossocial impactam o acadêmico.
- Coração da solução: IA como sentinela contra a defasagem.
- Impacto real: projeções de melhoria e recomendações práticas.

## Itens de Entrega (Checklist Final)

- Repositório GitHub: código versionado e documentado.
- Notebooks: EDA e ML documentados (feature engineering → modelo).
- App Streamlit: publicado no Streamlit Community Cloud.
- Apresentação Executiva (PDF): storytelling orientado a resultados.
- Vídeo de demonstração (≈5 minutos): visão gerencial + tour técnico.

---

> Observação: após validar este Plano Mestre, atualizar `documentation/PLANO_DESENVOLVIMENTO.md` para referenciar e alinhar as tarefas com este plano e com as regras de negócio encontradas nos PDFs de referência.
