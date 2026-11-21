"""Chat first: one network switch, persistent guidance, inline analysis results."""
from html import escape
from pathlib import Path
import time
import streamlit as st
from llm_lab import session as memory
from llm_lab.client import LLMError
from conversation.engine import handle
from conversation.model import ConversationModel
from conversation.plots import draw_result
from conversation.web import NetworkPolicy, probe

REPO = 'https://github.com/junwon1901-lab/junwon-conversation-lab'
REPORT = 'https://app.notion.com/p/41cd259f3ec781d2a072c3bc566fafd3?pvs=204'


@st.fragment(run_every='30s')
def connection_status():
    enabled = st.session_state.get('internet_enabled', False)
    if not enabled:
        st.markdown('<div class="connection off">● INTERNET OFF <span>외부 요청 차단 · 기본 분석 가능</span></div>', unsafe_allow_html=True)
        return
    checked = st.session_state.get('connection_check', {})
    if time.monotonic() - checked.get('tick', 0) > 30:
        checked = {**probe(NetworkPolicy(True)), 'tick': time.monotonic()}
        st.session_state.connection_check = checked
    color = 'on' if checked.get('connected') else 'uncertain'
    title = 'INTERNET ON · 연결 확인' if checked.get('connected') else 'INTERNET ON · 점검 서버 응답 미확인'
    st.markdown(f'<div class="connection {color}">● {title}<span>{escape(checked.get("checked_at", ""))} · 기본 명령은 계속 사용 가능</span></div>', unsafe_allow_html=True)
    st.caption(checked.get('detail', ''))
    last = st.session_state.get('conversation_state', {}).get('search_status')
    if last:
        st.caption(('최근 검색 성공' if last['ok'] else '최근 검색 실패 · 서비스 일시 제한 가능') + ' · ' + last['checked_at'])


def guide(state):
    data = state.get('dataset')
    with st.container(key='persistent_guide'):
        st.markdown('**채팅 가이드 · 항상 표시**')
        st.markdown('자유롭게 질문하고 `더 쉽게 설명해줘`로 이어가세요. 웹 자료가 필요하면 `검색해줘`를 덧붙이세요.')
        st.markdown('`선형회귀 해줘` · `비선형회귀 해줘` · `신경망 학습해줘` · `모델 3가지 비교해줘`')
        st.caption('문서: 자료 등록: 제목 + 다음 줄에 본문 · 자료 목록 · 기억해: 내용 · 기억 보여줘')
        if data:
            st.caption(f"내 데이터 {len(data['rows'])}행 · 입력 {', '.join(data['columns'][:-1])} · 목표 {data['columns'][-1]} · 예측: " + ', '.join(k+'=값' for k in data['columns'][:-1]))
        else:
            st.caption('내 숫자 입력: x: 1,2,3,4,5,6 y: 2,4,6,8,10,12 선형회귀 해줘 · 도움말 · 새 대화 시작')


def render():
    st.set_page_config(page_title='Conversation Lab · 대화로 시작하는 데이터 탐구', page_icon='💬', layout='wide')
    st.markdown('<style>' + Path(__file__).with_name('ui.css').read_text() + '</style>', unsafe_allow_html=True)
    if 'conversation_state' not in st.session_state:
        st.session_state.conversation_state = memory.new_session()
    state = st.session_state.conversation_state
    with st.sidebar:
        st.markdown('<div class="wordmark">C<span>↗</span></div><h2>Conversation Lab</h2><p class="muted">서준원의 대화형 데이터 실험실</p>', unsafe_allow_html=True)
        st.markdown('---')
        st.markdown('**대화부터 시작하세요**\n\n정해진 문장 밖의 질문도 괜찮습니다. 앞서 나눈 대화를 이어서 설명하거나 자료를 찾아볼 수 있습니다.')
        st.markdown('**분석도 대화 안에서**\n\n선형·비선형 회귀, 인공신경망과 직접 입력한 데이터 분석을 지원합니다.')
        st.markdown('**기록 다루기**\n\n`이전 결과 보여줘`\n\n`보고서 만들어줘`\n\n`새 대화 시작`')
        st.markdown('---')
        st.caption('무료 CPU 공개 모델 · 유료 API 키 불필요\n\n모델 최초 준비에는 시간이 걸립니다. 서버가 재시작되면 대화가 초기화될 수 있습니다.')
        st.caption('ON/OFF는 앱의 외부 접근 설정입니다. 웹사이트 접속에는 인터넷이 필요하며, 실제 인터넷 단절 환경에서는 PC 로컬 실행을 사용합니다.')
        st.markdown(f'[소스 코드 ↗]({REPO})' + (f'　[보고서 ↗]({REPORT})' if REPORT else ''))
    left, right = st.columns([3, 1])
    with left:
        st.markdown('<div class="eyebrow">CONVERSATION × DATA × DISCOVERY</div>', unsafe_allow_html=True)
    with right:
        enabled = st.toggle('인터넷 ON / OFF', value=True, key='internet_enabled', help='앱의 외부 검색·모델 다운로드를 제어합니다. 기기의 Wi-Fi 설정을 바꾸지는 않습니다.')
    connection_status()
    if not state['messages']:
        st.markdown('<section class="hero"><div class="hero-kicker">질문 하나에서, 새로운 이해로.</div><h1>무엇이 궁금한가요?</h1><p>대화를 나누고, 자료를 찾고, 숫자로 확인하세요.<br>한 번의 질문을 다음 발견으로 이어갑니다.</p></section>', unsafe_allow_html=True)
        st.markdown('<div class="ideas"><div><small>01 · TALK</small><b>생각을 함께 정리하기</b><p>“발표 전에 긴장될 때는<br>어떻게 준비하면 좋을까?”</p></div><div><small>02 · EXPLORE</small><b>웹 자료와 연결하기</b><p>“배가 물에 뜨는 원리를<br>검색해서 쉽게 설명해줘.”</p></div><div><small>03 · ANALYZE</small><b>내 데이터로 확인하기</b><p>“이 숫자들에 어떤 모델이<br>가장 잘 맞을까?”</p></div></div>', unsafe_allow_html=True)
    runs = {r['id']: r for r in state['runs'] if r.get('id')}
    for message in state['messages'][-40:]:
        with st.chat_message(message['role'], avatar='💬' if message['role'] == 'assistant' else None):
            content = message['content']
            if message['role'] == 'user' and len(content) > 1800:
                content = content[:1200] + '\n\n… (입력 데이터 등록됨)'
            st.markdown(content)
            receipt = message.get('tool_receipt')
            if receipt:
                st.caption(f"실행 확인 {receipt['id']} · {receipt['action']} · {receipt['status']} · {receipt['dataset']['source']} {receipt['dataset']['rows']}행 · LLM 결과 수신: {receipt.get('llm_feedback', 'pending')}")
            if message.get('interpretation'):
                st.markdown('**실행 결과에 대한 LLM 해석**')
                st.markdown(message['interpretation'])
            for i, doc in enumerate(message.get('knowledge', []), 1):
                st.markdown(f'**[자료{i}] {doc["title"]}** · 조각 {doc["chunk"]}')
                st.caption(doc['source'])
                st.text(doc['text'])
            if message.get('report'):
                st.markdown(message['report'])
            elif message.get('run_id') in runs:
                draw_result(runs[message['run_id']], errors_only=message.get('action') == 'errors')
            for index, source in enumerate(message.get('sources', []), 1):
                label = escape(source['title']).replace('[', '').replace(']', '')
                st.markdown(f"[{index}. {label}]({source['url']})")
                st.caption(source['kind'] + ' · 확인 ' + source['retrieved_at'])
            evidence = message.get('evidence')
            if evidence:
                mode = '이전 검색 자료 참고' if evidence.get('reused_sources') else '웹 검색 참고' if evidence.get('network_used') and not evidence.get('search_error') else '추가 웹 검색 없음'
                st.caption(f"{mode} · {evidence.get('model', '')}")
    with st.bottom:
        guide(state)
        prompt = st.chat_input('궁금한 것을 편하게 물어보세요', max_chars=100000)
    if prompt:
        if prompt.strip().rstrip('.!') in ('새 대화 시작', '새 대화', '대화 초기화'):
            st.session_state.conversation_state = memory.new_session()
            st.rerun()
        with st.chat_message('user'):
            st.markdown(prompt[:1800])
        with st.chat_message('assistant', avatar='💬'):
            status = st.empty()
            try:
                handle(prompt, state, enabled, progress=lambda value: status.info(value))
                status.empty()
                st.rerun()
            except (LLMError, ValueError, OSError) as exc:
                status.empty()
                st.error(str(exc))
