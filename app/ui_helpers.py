RISK_DETAILS = {
    'alto': {
        'level': 'ALTO',
        'css_class': 'risk-alto',
        'emoji': '🔴',
        'accent_color': '#E8562A',
        'recommendation': (
            "**Ação imediata recomendada:**\n\n"
            "- 🧠 Encaminhar para avaliação psicopedagógica urgente\n"
            "- 📞 Contato com responsáveis esta semana\n"
            "- 📖 Plano de reforço acadêmico personalizado\n"
            "- 💙 Apoio psicossocial – verificar contexto familiar"
        ),
    },
    'medio': {
        'level': 'MÉDIO',
        'css_class': 'risk-medio',
        'emoji': '🟡',
        'accent_color': '#F4A259',
        'recommendation': (
            "**Monitoramento intensificado:**\n\n"
            "- 📊 Acompanhar indicadores semanalmente\n"
            "- 🎯 Reforçar engajamento em atividades extracurriculares\n"
            "- 💬 Conversa individual com o aluno sobre motivação\n"
            "- 👀 Observar IPS nas próximas semanas"
        ),
    },
    'baixo': {
        'level': 'BAIXO',
        'css_class': 'risk-baixo',
        'emoji': '🟢',
        'accent_color': '#4CAF9A',
        'recommendation': (
            "**Manter e potencializar:**\n\n"
            "- ✅ Aluno apresenta bom desempenho geral\n"
            "- 🏆 Avaliar indicação para bolsa ou destaque\n"
            "- 🚀 Desafiar com atividades de liderança\n"
            "- 📈 Manter monitoramento mensal regular"
        ),
    },
}


def clamp_probability(probability):
    return min(max(float(probability), 0.0), 1.0)


def describe_risk(probability, threshold):
    prob = clamp_probability(probability)
    if prob >= 0.65:
        details = dict(RISK_DETAILS['alto'])
    elif prob >= float(threshold):
        details = dict(RISK_DETAILS['medio'])
    else:
        details = dict(RISK_DETAILS['baixo'])
    details['progress_pct'] = int(round(prob * 100))
    return details


def build_risk_progress_html(probability, accent_color):
    pct = int(round(clamp_probability(probability) * 100))
    return (
        '<div class="risk-progress-track">'
        f'<div class="risk-progress-fill" style="width:{pct}%; background:{accent_color};"></div>'
        '</div>'
    )


def resolve_sentiment_values(observacoes, score_fn, usar_manual=False, sent_score_manual=None, sent_len_manual=None):
    text = '' if observacoes is None else str(observacoes)
    auto_score = int(score_fn(text))
    auto_len = len(text)

    if usar_manual:
        if sent_score_manual is None or sent_len_manual is None:
            raise ValueError('Valores manuais de sentimento e comprimento são obrigatórios quando o modo manual está ativo.')
        return {
            'mode': 'manual',
            'sent_score': int(sent_score_manual),
            'sent_len': int(sent_len_manual),
            'preview': f"Manual ativo: score {int(sent_score_manual)} · comprimento {int(sent_len_manual)}",
        }

    return {
        'mode': 'automatico',
        'sent_score': auto_score,
        'sent_len': auto_len,
        'preview': f"Automático: score {auto_score} · comprimento {auto_len}",
    }