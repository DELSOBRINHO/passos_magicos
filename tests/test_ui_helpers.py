from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))

from ui_helpers import build_risk_progress_html, describe_risk, resolve_sentiment_values


def test_resolve_sentiment_values_automatico_usa_heuristica_e_tamanho_texto():
    result = resolve_sentiment_values('Aluno muito engajado', lambda text: 3)

    assert result['mode'] == 'automatico'
    assert result['sent_score'] == 3
    assert result['sent_len'] == len('Aluno muito engajado')
    assert 'Automático' in result['preview']


def test_resolve_sentiment_values_manual_sobrescreve_automatico():
    result = resolve_sentiment_values(
        'Texto qualquer',
        lambda text: -2,
        usar_manual=True,
        sent_score_manual=4,
        sent_len_manual=180,
    )

    assert result['mode'] == 'manual'
    assert result['sent_score'] == 4
    assert result['sent_len'] == 180
    assert 'Manual ativo' in result['preview']


def test_describe_risk_classifica_faixas_e_percentual():
    medio = describe_risk(0.58, 0.30)
    alto = describe_risk(0.95, 0.30)
    baixo = describe_risk(0.01, 0.30)

    assert medio['level'] == 'MÉDIO'
    assert medio['progress_pct'] == 58
    assert alto['level'] == 'ALTO'
    assert baixo['level'] == 'BAIXO'


def test_build_risk_progress_html_reflete_percentual():
    html = build_risk_progress_html(0.58, '#F4A259')

    assert 'width:58%' in html
    assert '#F4A259' in html