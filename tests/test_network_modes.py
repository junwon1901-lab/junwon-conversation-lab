from pathlib import Path
from unittest.mock import Mock
import socket
import pytest
from streamlit.testing.v1 import AppTest
from conversation.engine import handle
from conversation.web import NetworkPolicy, OfflineError, search, probe, excerpt
from llm_lab import session, cloud_client


@pytest.fixture(autouse=True)
def isolated_sessions(tmp_path, monkeypatch):
    monkeypatch.setenv('LLM_SESSION_DIR', str(tmp_path / 'sessions'))


def no_socket(*args, **kwargs):
    raise AssertionError('OFF mode made a network request')


def test_offline_policy_guards_before_any_network(monkeypatch):
    monkeypatch.setattr(socket, 'getaddrinfo', no_socket)
    for func, args in [(search, ('hello', NetworkPolicy())), (probe, (NetworkPolicy(),)), (excerpt, ('https://example.com', NetworkPolicy()))]:
        with pytest.raises(OfflineError):
            func(*args)


def test_offline_without_llm_runs_all_three_models_and_prediction(monkeypatch):
    monkeypatch.setattr(socket.socket, 'connect', no_socket)
    monkeypatch.setattr(socket, 'getaddrinfo', no_socket)
    monkeypatch.setattr(cloud_client, '_load_locked', no_socket)
    model = Mock()
    model.available.return_value = False
    state = session.new_session()
    handle('x: 1,2,3,4,5,6 y: 5,7,9,11,13,15 선형회귀 해줘', state, False, model)
    result = handle('x가 10일 때 예측해줘', state, False, model)
    assert result['evidence']['prediction']['value'] == pytest.approx(23)
    for prompt in ['비선형회귀 해줘', '신경망 학습해줘']:
        handle(prompt, state, False, model)
    assert [r['action'] for r in state['runs']] == ['linear', 'polynomial', 'neural']
    assert 'OFF' in handle('새로운 질문이야', state, False, model)['reply']
    model.route.assert_not_called()


def test_offline_never_downloads_missing_model(tmp_path, monkeypatch):
    monkeypatch.setenv('LLM_MODEL_DIR', str(tmp_path / 'empty'))
    monkeypatch.setattr(cloud_client, 'urlopen', no_socket)
    with pytest.raises(cloud_client.LLMError, match='OFF'):
        cloud_client.ensure_model(allow_download=False)


def model_for(action):
    model = Mock()
    model.available.return_value = True
    model.route.return_value = ({'action': action, 'query': '부력 원리', 'degree': 2, 'values': {}, 'reply': ''}, {})
    model.answer.return_value = ('참고 자료로 설명합니다 [1].', {'model': '테스트 대역'})
    return model


def test_search_failure_does_not_fake_evidence():
    state = session.new_session()
    model = model_for('search')
    researcher = Mock(side_effect=TimeoutError('timeout'))
    result = handle('부력 원리를 검색해줘', state, True, model, researcher)
    assert '최신 사실은 확인되지 않았습니다' in result['reply']
    assert result['sources'] == [] and state['search_status']['ok'] is False
    assert not state['runs']


def test_followup_retains_source_and_session_isolation():
    state, other = session.new_session(), session.new_session()
    source = {'title': 'Reference', 'url': 'https://example.com', 'text': '부력 설명', 'kind': '검색 요약', 'retrieved_at': 'test'}
    model = model_for('search')
    handle('검색해줘', state, True, model, Mock(return_value=[source]))
    model.route.return_value[0]['action'] = 'chat'
    result = handle('그걸 쉽게 설명해줘', state, False, model, no_socket)
    assert result['evidence']['reused_sources'] and not result['evidence']['network_used']
    assert model.answer.call_args.kwargs['sources'] == [source]
    assert not other['messages'] and 'last_sources' not in other


def test_ui_switch_blocks_probe_and_commands_need_no_llm(monkeypatch):
    monkeypatch.setattr('conversation.ui.probe', lambda _: {'connected': True, 'checked_at': 'test'})
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=60).run()
    app.toggle[0].set_value(False).run()
    monkeypatch.setattr('conversation.ui.probe', no_socket)
    monkeypatch.setattr(cloud_client, '_load_locked', no_socket)
    app.chat_input[0].set_value('x: 1,2,3,4,5,6 y: 5,7,9,11,13,15 선형회귀 해줘').run()
    assert not app.exception and not app.error and not app.button
    assert any('INTERNET OFF' in m.value for m in app.markdown)
    assert len(app.get('plotly_chart')) >= 1
    assert any('항상 표시' in m.value for m in app.markdown)
    app.chat_input[0].set_value('새 대화 시작').run()
    assert not app.session_state['conversation_state']['messages']
    assert app.toggle[0].value is False
