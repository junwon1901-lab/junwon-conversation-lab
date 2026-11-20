"""Validated in-process tool API. No arbitrary code, shell, or file operations."""
import hashlib
import json
import math
import re
import uuid
from llm_lab.agent import execute_plan
from llm_lab import session as memory
from llm_lab.client import LLMError

ACTIONS = frozenset(('describe', 'linear', 'polynomial', 'neural', 'compare', 'epoch_compare', 'last', 'errors', 'report', 'predict'))


def validate(plan):
    if not isinstance(plan, dict) or plan.get('action') not in ACTIONS:
        raise ValueError('허용되지 않은 분석 동작입니다.')
    degree = plan.get('degree', 2)
    if type(degree) is not int or degree not in (2, 3):
        raise ValueError('다항식 차수는 2 또는 3이어야 합니다.')
    values = plan.get('values', {})
    if not isinstance(values, dict) or len(values) > 8 or any(
        not isinstance(k, str) or type(v) not in (int, float) or not math.isfinite(v) for k, v in values.items()
    ):
        raise ValueError('예측 입력은 최대 8개의 유한한 숫자여야 합니다.')
    if plan['action'] != 'predict' and values:
        raise ValueError('예측 이외의 동작에는 숫자 입력을 전달할 수 없습니다.')
    return {'action': plan['action'], 'degree': degree, 'values': values, 'reply': ''}


def run_tool(text, state, plan, evidence, model, policy, progress):
    plan = validate(plan)
    data = state.get('dataset')
    receipt = {'id': uuid.uuid4().hex[:12], 'time': memory.now(), 'action': plan['action'],
               'arguments': {'degree': plan['degree'], 'values': plan['values']},
               'dataset': {'rows': len(data['rows']) if data else 1018,
                           'fingerprint': hashlib.sha256(json.dumps(data or 'bundled hull v4', sort_keys=True).encode()).hexdigest()[:12],
                           'source': '사용자 입력' if data else '선체 저항 공개 데이터'}}
    try:
        output = execute_plan(text, state, plan, evidence)
    except (ValueError, RuntimeError) as exc:
        receipt.update({'status': 'error', 'error': str(exc)})
        state.setdefault('tool_receipts', []).append(receipt)
        state['tool_receipts'] = state['tool_receipts'][-30:]
        memory.save(state)
        raise
    result = output.get('result')
    receipt.update({'status': 'success', 'run_id': result.get('id') if result else None,
                    'result': memory.summary(result) if result else {},
                    'prediction': output['evidence'].get('prediction'),
                    'python_reply': output['reply']})
    # Pass a compact execution result to the same LLM, then retain both outputs separately.
    message = state['messages'][-1]
    message['tool_receipt'] = receipt
    state.setdefault('tool_receipts', []).append(receipt)
    state['tool_receipts'] = state['tool_receipts'][-30:]
    if policy.enabled or model.available():
        progress(output['reply'] + '\n\nPython 계산이 끝났습니다. 이 결과를 LLM에 돌려주어 해석하고 있습니다.')
        try:
            explanation, meta = model.explain_tool(text, state, policy, receipt)
            explanation = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', explanation)
            message['interpretation'] = explanation
            receipt['llm_feedback'] = 'completed'
            receipt['llm_seconds'] = meta.get('seconds')
        except (LLMError, ValueError, OSError) as exc:
            receipt['llm_feedback'] = 'failed'
            message['interpretation'] = '계산은 완료했지만 LLM 해석을 생성하지 못했습니다. 위 Python 결과는 그대로 사용할 수 있습니다.'
    else:
        receipt['llm_feedback'] = 'unavailable_offline'
        message['interpretation'] = '인터넷 OFF이며 대화 모델이 준비되지 않아 Python 계산 결과만 표시합니다.'
    memory.save(state)
    output['tool_receipt'] = receipt
    return output
