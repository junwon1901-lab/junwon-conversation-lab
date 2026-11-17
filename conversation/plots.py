"""A conversation workspace: commands and real Python results share one thread."""
from html import escape
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from llm_lab.client import LLMError
from llm_lab.agent import handle
from llm_lab import analysis, session as memory

COLORS = ['#5281d6', '#6aab9b', '#ad8acb']



def chart_style(fig, **kwargs):
    fig.update_layout(template='plotly_white', height=360, font={'family': 'Noto Sans KR, sans-serif', 'color': '#76859a', 'size': 11},
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#fafbfd',
                      margin={'l': 25, 'r': 15, 't': 45, 'b': 45},
                      legend={'orientation': 'h', 'y': -0.2}, **kwargs)
    fig.update_xaxes(gridcolor='#edf1f7', zerolinecolor='#e3eaf4')
    fig.update_yaxes(gridcolor='#edf1f7', zerolinecolor='#e3eaf4')
    st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})


def network_diagram(model):
    cfg = model['config']
    layers = [len(model['data']['features']), *cfg.get('hidden_layers', (64,32,16)), 1]
    items = []
    for i, size in enumerate(layers):
        if i:
            items.append('<span class="arrow">→</span>')
        label = '입력' if i == 0 else '출력' if i == len(layers)-1 else f'은닉층 {i}'
        items.append(f'<div class="lab-layer {"output" if i == len(layers)-1 else ""}"><strong>{size}</strong>{label}</div>')
    st.markdown('<div class="lab-network">' + ''.join(items) + '</div>', unsafe_allow_html=True)
    st.caption(f"활성화 함수: {cfg.get('activation', 'relu')} · 입력을 가중합한 뒤 비선형 활성화를 거쳐 출력합니다.")


def draw_result(run, *, errors_only=False):
    custom = run.get('custom_data', False)
    evaluation = run.get('evaluation_label', '테스트')
    selected = next(r for r in run['models'] if r['kind'] == run['selected'])
    if errors_only:
        st.markdown('#### 오차가 큰 예측')
        table = pd.DataFrame(analysis.worst_errors(run)).drop(columns=['hull','Fn'], errors='ignore').rename(columns={
            'row':'원자료 행','actual':'실제값','predicted':'예측값','residual':'잔차','absolute_error':'절대오차','predicted_actual_ratio':'예측/실제'})
        st.dataframe(table, hide_index=True, width='stretch')
        return
    st.markdown(f"### {selected['name']}" + (' · 비교 결과' if len(run['models']) > 1 else ' · 분석 결과'))
    st.caption(f"{selected['data']['source'] if custom else '공개 선체 저항 데이터'} · {selected['data']['rows']:,}행 · {evaluation}")
    if selected.get('equation'):
        st.markdown(f'<div class="lab-equation">{escape(selected["equation"])}</div>', unsafe_allow_html=True)
    metric = next(m for m in selected['metrics'] if m['split'] == evaluation)
    for column, title, key in zip(st.columns(3), ['R² · 결정계수', 'RMSE · 제곱평균제곱근오차', 'MAE · 평균절대오차'], ['R2','RMSE','MAE']):
        column.metric(title, f'{metric[key]:.4f}')
    fig = go.Figure()
    if selected.get('curve'):
        fig.add_trace(go.Scatter(x=[p['x'] for p in selected['input_points']], y=[p['y'] for p in selected['input_points']], mode='markers', name='입력한 점', marker={'size':8,'color':'#263c5b'}))
        for color, model in zip(COLORS, run['models']):
            fig.add_trace(go.Scatter(x=[p['x'] for p in model['curve']], y=[p['y'] for p in model['curve']], mode='lines', name=model['name'],line={'color':color,'width':2}))
        chart_style(fig, title='입력 데이터와 적합 곡선', xaxis_title=selected['data']['features'][0], yaxis_title=selected['data']['target'])
        st.caption('곡선은 전체 입력 데이터에 적합했습니다. 위 지표는 각 행을 학습에서 제외한 교차검증 예측으로 계산합니다.')
    else:
        values = []
        for color, model in zip(COLORS, run['models']):
            points = model['test_points']
            y, p = [x['actual'] for x in points], [x['predicted'] for x in points]
            values.extend(y+p)
            fig.add_trace(go.Scatter(x=y,y=p,mode='markers',name=model['name'],marker={'size':6,'opacity':.65,'color':color}))
        lo,hi = min(values),max(values)
        fig.add_trace(go.Scatter(x=[lo,hi],y=[lo,hi],mode='lines',name='예측 = 실제',line={'dash':'dash','color':'#a9b7cb'}))
        chart_style(fig,title='실제값과 예측값',xaxis_title='실제값',yaxis_title='예측값')
    if not custom:
        st.caption('목표값: Rt/Δ × 10³ · 힘의 단위 N이 아닙니다. 같은 선박의 행은 학습·검증·테스트에 섞이지 않습니다.')
    rows = [m for r in run['models'] for m in r['metrics'] if m['split'] == evaluation]
    st.dataframe(pd.DataFrame(rows).rename(columns={'model':'모델','R2':'R²','rows':'평가 행 수','negative_predictions':'음수 예측 수'})[['모델','R²','RMSE','MAE','평가 행 수','음수 예측 수']], hide_index=True, width='stretch')
    neural = next((r for r in run['models'] if r['kind'] == 'neural'), None)
    if neural:
        st.markdown('#### 신경망은 이렇게 연결됩니다')
        network_diagram(neural)
        if neural['history']:
            history = pd.DataFrame(neural['history'])
            loss = go.Figure()
            for color, col in zip(COLORS, [c for c in history.columns if c != 'epoch']):
                loss.add_trace(go.Scatter(x=history['epoch'],y=history[col],name=col,mode='lines',line={'color':color,'width':2}))
            chart_style(loss,title='실제 학습 손실 곡선',xaxis_title='학습 횟수 (epoch)',yaxis_title='표준화한 목표 척도의 손실')
            if neural['best_epoch'] is not None:
                st.caption(f"검증 오차가 가장 작았던 {neural['best_epoch']} epoch의 가중치를 채택했습니다.")
    if run.get('epoch_candidates'):
        st.markdown('#### 50 / 500 epoch 상한 비교')
        rows = [{k:v for k,v in r.items() if k!='history'} for r in run['epoch_candidates']]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
    st.caption(run['selection_basis'])
    st.caption(run['note'])
    st.caption('이어가기: x가 10일 때 예측해줘 · 다른 모델로 학습해줘 · 오차가 큰 예측 10개 보여줘 · 보고서 만들어줘' if custom and len(selected['data']['features']) == 1 else '이어가기: 오차가 큰 예측 10개 보여줘 · 보고서 만들어줘 · 이전 결과 보여줘')

