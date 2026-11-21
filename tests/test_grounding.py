from conversation.grounding import check_analysis, check_documents
from conversation.web import probe, NetworkPolicy
from unittest.mock import Mock
import requests

def test_observed_wrong_extrapolation_is_replaced():
    receipt={'prediction':{'model':'선형회귀','value':23,'outside_training_range':['x']}}
    text, issues=check_analysis('학습 데이터 범위 내에서 계산되었습니다.',receipt)
    assert issues and '외삽입니다' in text and '오차는 아직 알 수 없습니다' in text

def test_perfect_accuracy_claim_is_replaced():
    text, issues=check_analysis('정확도가 완벽한 것으로 보입니다.',{})
    assert issues and '보장하지 않습니다' in text

def test_buoyancy_contradiction_is_replaced():
    text, issues=check_documents('부력은 잠긴 체적에 따라 변화하지 않습니다.',[{'id':'buoyancy'}])
    assert issues and '체적에 비례' in text

def test_correct_explanation_is_preserved():
    text='학습 범위 밖이므로 외삽입니다.'
    assert check_analysis(text,{'prediction':{'outside_training_range':['x']}})==(text,[])

def test_probe_one_service_failure_not_total_outage(monkeypatch):
    monkeypatch.setattr('conversation.web.requests.get',Mock(side_effect=[requests.Timeout(),Mock(status_code=200)]))
    result=probe(NetworkPolicy(True))
    assert result['connected'] and not result['checks'][0]['ok'] and result['checks'][1]['ok']

def test_corrected_document_uses_actual_source_number():
    text, _=check_documents('잠긴 체적에 따라 변하지 않습니다.',[{'id':'ann'},{'id':'buoyancy'}])
    assert '[자료2]' in text and '[자료1]' not in text
