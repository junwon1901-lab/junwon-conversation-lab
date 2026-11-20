from unittest.mock import Mock
import socket
import pytest
from conversation.engine import handle
from conversation.model import ConversationModel
from conversation.tools import validate
from conversation.knowledge import manage, retrieve
from llm_lab import session
from llm_lab.client import LLMError

@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv('LLM_SESSION_DIR', str(tmp_path))

def model():
    m = Mock()
    m.available.return_value = True
    m.explain_tool.return_value = ('작은 표본이므로 외삽에는 주의해야 합니다.', {'seconds': .01})
    m.answer.return_value = ('정지한 배는 부력과 무게가 같습니다 [자료1].', {'model':'test'})
    m.route.return_value = ({'action':'chat','degree':2,'values':{},'query':'','reply':''}, {})
    return m

def test_execution_receipt_returns_real_result_to_llm_once():
    s, m = session.new_session(), model()
    result = handle('x: 1,2,3,4,5,6 y: 5,7,9,11,13,15 선형회귀 해줘', s, False, m)
    receipt = m.explain_tool.call_args.args[-1]
    assert receipt['status'] == 'success' and receipt['dataset']['rows'] == 6
    assert receipt['result']['metrics'][0]['RMSE'] < 1e-10
    assert result['tool_receipt']['llm_feedback'] == 'completed'
    assert len(s['runs']) == 1 and len(s['messages']) == 2
    prediction = handle('x가 10일 때 예측해줘', s, False, m)
    assert prediction['tool_receipt']['prediction']['value'] == pytest.approx(23)
    assert m.explain_tool.call_args.args[-1]['prediction']['value'] == pytest.approx(23)
    assert len(s['runs']) == 1

def test_llm_failure_keeps_successful_calculation():
    s, m = session.new_session(), model()
    m.explain_tool.side_effect = LLMError('not available')
    result = handle('x: 1,2,3,4,5,6 y: 5,7,9,11,13,15 선형회귀 해줘', s, True, m)
    assert result['tool_receipt']['status'] == 'success'
    assert result['tool_receipt']['llm_feedback'] == 'failed'
    assert len(s['runs']) == 1 and '완료' in result['reply']

@pytest.mark.parametrize('plan',[
    {'action':'exec'}, {'action':'linear','degree':9},
    {'action':'predict','values':{'x':float('nan')}},
    {'action':'predict','values':{'x':True}}, {'action':'linear','values':{'x':3}}
])
def test_tool_boundary_rejects_invalid_calls(plan):
    with pytest.raises(ValueError): validate(plan)

def test_rag_registration_retrieval_offline_and_isolation(monkeypatch):
    def blocked(*args, **kwargs): raise AssertionError('Network used')
    monkeypatch.setattr(socket, 'getaddrinfo', blocked)
    monkeypatch.setattr(socket.socket, 'connect', blocked)
    s, other = session.new_session(), session.new_session()
    m = model(); m.available.return_value = False
    handle('자료 등록: 실험선 A\n실험선 A의 전장은 42미터이고 시험 조건은 담수이다. 목표 속도는 3노트이다.', s, False, m)
    refs = retrieve('실험선 A 전장은 얼마야?', s)
    assert refs[0]['title'] == '실험선 A'
    result = handle('실험선 A 전장은 얼마야?', s, False, m)
    assert '42미터' in result['reply'] and '생성 답변은 아닙니다' in result['reply']
    assert not other.get('documents')
    handle('자료 삭제: 실험선 A', s, False, m)
    assert not s['documents']

def test_memory_survives_history_trimming_and_deletes():
    s = session.new_session()
    manage('기억해: 내 발표 주제는 선체 저항이야', s)
    s['messages'] = []
    actual = ConversationModel()
    actual._complete = Mock(return_value=('답변', {}))
    from conversation.web import NetworkPolicy
    actual.answer('내 발표 주제?', s, NetworkPolicy())
    assert '선체 저항' in actual._complete.call_args.args[0][0]['content']
    manage('기억 삭제', s)
    assert s['notes'] == []

def test_followup_does_not_repeat_web_search_and_receives_documents():
    s, m = session.new_session(), model()
    m.route.return_value[0]['action'] = 'search'
    researcher = Mock(side_effect=AssertionError('Unexpected search'))
    handle('정지한 배의 부력과 무게는 같은 거야?', s, True, m, researcher)
    assert s['active_knowledge'][0]['id'] == 'buoyancy'
    assert s['messages'][-1]['knowledge']
    researcher.assert_not_called()
    handle('방금 설명을 더 쉽게 설명해줘', s, True, m, researcher)
    researcher.assert_not_called()

def test_failed_prediction_records_error_without_success_message():
    s, m = session.new_session(), model()
    with pytest.raises(ValueError):
        handle('예측: wrong=2', s, False, m)
    assert s['tool_receipts'][-1]['status'] == 'error'
    assert not s['messages']
    m.explain_tool.assert_not_called()

def test_parallel_model_downloads_use_independent_temporary_files(tmp_path, monkeypatch):
    import io
    import hashlib
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from llm_lab import cloud_client as runtime
    payload = b'example model bytes for concurrency regression'
    gate = Barrier(2)
    monkeypatch.setenv('LLM_MODEL_DIR', str(tmp_path / 'models'))
    monkeypatch.setattr(runtime, 'MODEL_BYTES', len(payload))
    monkeypatch.setattr(runtime, 'MODEL_SHA256', hashlib.sha256(payload).hexdigest())
    def open_model(*args, **kwargs):
        gate.wait(timeout=10)
        return io.BytesIO(payload)
    monkeypatch.setattr(runtime, 'urlopen', open_model)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(runtime.ensure_model) for _ in range(2)]
        paths = [f.result(timeout=15) for f in futures]
    assert all(p.read_bytes() == payload for p in paths)
    assert not list(paths[0].parent.glob('*.part'))
