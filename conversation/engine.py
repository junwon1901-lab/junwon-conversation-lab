"""Exact local commands first, then free conversation and bounded retrieval."""
import re
from llm_lab.input_data import command
from llm_lab.agent import execute_plan
from llm_lab import session as memory
from llm_lab.client import LLMError
from conversation.model import ConversationModel
from conversation.web import NetworkPolicy, search
from conversation import knowledge
from conversation.tools import run_tool


def record(text, reply, state, evidence, sources=None):
    # Don't allow a model response to silently load remote images in OFF mode.
    reply = re.sub(r'!\[[^\]]*\]\([^)]*\)', '', reply)
    count = len(sources or [])
    reply = re.sub(r'\[(\d+)\](?!\()', lambda m: m.group(0) if 1 <= int(m[1]) <= count else '', reply)
    reply = re.sub(r'\[자료(\d+)\]', lambda m: m.group(0) if 1 <= int(m[1]) <= len(state.get('active_knowledge', [])) else '', reply)
    message = {'role': 'assistant', 'content': reply, 'action': 'conversation',
               'sources': sources or [], 'evidence': evidence}
    state['messages'].extend([{'role': 'user', 'content': text}, message])
    state['messages'] = state['messages'][-200:]
    state['events'].append({**evidence, 'time': memory.now(), 'action': 'conversation'})
    state['events'] = state['events'][-100:]
    memory.save(state)
    return {'reply': reply, 'action': 'conversation', 'sources': sources or [], 'evidence': evidence}


def handle(text, state, enabled=False, model=None, researcher=search, progress=lambda _: None):
    if not isinstance(text, str) or not text.strip() or len(text) > 100_000:
        raise ValueError('입력은 1~100,000자로 작성하세요.')
    policy = NetworkPolicy(enabled)
    text = text.strip()
    managed = knowledge.manage(text, state)
    if managed is not None:
        return record(text, managed, state, {'model': '대화 자료·기억 관리', 'internet': enabled, 'network_used': False})
    model = model or ConversationModel()
    plan = command(text, state)
    if plan is not None:
        progress('입력한 데이터와 명령을 확인하고 Python으로 계산하고 있습니다.')
        evidence = {'model': '기존 Python 분석', 'seconds': 0, 'internet': enabled, 'network_used': False}
        if plan['action'] == 'chat':
            return execute_plan(text, state, plan, evidence)
        return run_tool(text, state, plan, evidence, model, policy, progress)
    if len(text) > 4000:
        raise ValueError('일반 대화는 한 번에 4,000자 이내로 나누어 주세요. 데이터 입력은 100,000자까지 가능합니다.')
    local_refs = knowledge.retrieve(text, state)
    followup = re.search(r'방금|그걸|그것|그 내용|그 자료|그 설명|이어서|더 쉽게|쉽게 다시|첫 번째|두 번째|세 번째|위 내용|위 자료|요약해', text)
    if not local_refs and followup:
        local_refs = state.get('last_knowledge', [])
    state['active_knowledge'] = local_refs
    state['last_knowledge'] = local_refs
    if not enabled and not model.available():
        if local_refs:
            reply = '대화 모델이 준비되지 않아 **저장 자료 원문**을 표시합니다. LLM 생성 답변은 아닙니다.\n\n' + '\n\n'.join(f'**{d["title"]}**\n\n{d["text"]}\n\n출처: {d["source"]} · 조각 {d["chunk"]}' for d in local_refs)
            return record(text, reply, state, {'model': '로컬 문서 검색', 'internet': False, 'network_used': False})
        return record(text, '지금은 인터넷 OFF이고 대화 모델이 아직 준비되지 않았습니다. **선형회귀 해줘**, **비선형회귀 해줘**, **신경망 학습해줘**와 직접 입력한 데이터 분석은 바로 사용할 수 있어요. ON에서 모델을 한 번 준비하면 OFF에서도 일반 대화를 이어갈 수 있습니다.', state,
                      {'model': '오프라인 안내', 'internet': False, 'network_used': False})
    progress('이전 대화와 현재 질문을 읽고 있습니다. 첫 대화는 모델 다운로드가 필요할 수 있습니다.')
    try:
        plan, routing = model.route(text, state, policy)
    except LLMError:
        # A failed classifier never executes a tool; ordinary dialogue can still continue.
        plan, routing = {'action': 'chat', 'query': '', 'degree': 2, 'values': {}}, {'route_fallback': True}
        if re.search(r'데이터|회귀|모델|신경망', text) and re.search(r'학습|돌려|적합|분석|실행', text) and not re.search(r'하지\s*마|하지\s*말|말고|설명', text):
            return record(text, '분석 요청을 안전하게 해석하지 못해 **실행하지 않았습니다**. `선형회귀 해줘`, `비선형회귀 해줘`, `신경망 학습해줘`로 다시 요청하세요. 실행 전 오차나 예측값을 만들어 답하지 않습니다.', state, routing)
    if enabled and re.search(r'(?:검색|찾아)\s*(?:해서|해서\s*알려|해\s*줘|줘|봐)|search\s+(?:for|the)', text, re.I):
        plan['action'] = 'search'
        plan['query'] = plan.get('query') or text[:180]
    if routing.get('needs_literal_prediction'):
        columns = state.get('dataset', {}).get('columns', [])[:-1] or ['Fn','Rn','Cp','Cb','Cw','L_B','B_T','LCB_L']
        return record(text, '숫자를 정확히 계산에 전달할 수 있도록 다음 형식으로 입력해 주세요: `예측: ' + ', '.join(c+'=값' for c in columns) + '`. 등록한 입력 열의 값을 모두 적으면 학습한 모델로 예측합니다.', state, routing)
    if plan['action'] not in ('chat', 'search'):
        progress('질문을 분석 요청으로 해석했습니다. Python으로 실제 계산하고 있습니다.')
        return run_tool(text, state, plan, {**routing, 'internet': enabled, 'network_used': False}, model, policy, progress)
    sources, search_error = [], None
    # Re-explain retrieved passages locally unless fresh information/search is requested.
    explicit_search = bool(re.search(r'검색|찾아|최신|오늘|현재|지금|search', text, re.I))
    if (followup or local_refs) and not explicit_search and plan['action'] == 'search':
        plan['action'] = 'chat'
    if plan['action'] == 'search' and enabled:
        query = plan.get('query') or text
        progress('인터넷에서 관련 자료를 찾고 있습니다: ' + query)
        try:
            sources = researcher(query, policy)
            if not sources:
                search_error = '검색 결과가 없습니다.'
            else:
                state['last_sources'] = sources
        except Exception as exc:
            # Search outages must not erase conversation or prevent local calculations.
            search_error = '검색 서비스에 연결하지 못했습니다.'
        state['search_status'] = {'ok': bool(sources), 'checked_at': memory.now(), 'query': query}
    # A follow-up may explain a previously retrieved passage without claiming a new search.
    followup = re.search(r'방금|그걸|그것|그 내용|그 자료|그 설명|이어서|더 쉽게|쉽게 다시|첫 번째|두 번째|세 번째|위 내용|위 자료|요약해', text)
    previous_sources = state.get('last_sources', []) if followup and not sources and not search_error else []
    if not followup and not sources:
        state.pop('last_sources', None)
    progress('검색 자료와 대화 맥락을 바탕으로 답변을 작성하고 있습니다.' if sources else '대화 맥락을 바탕으로 답변을 작성하고 있습니다.')
    reply, evidence = model.answer(text, state, policy, sources=sources or previous_sources, search_error=search_error)
    if search_error:
        reply = '**이번 검색을 완료하지 못했습니다. 최신 사실은 확인되지 않았습니다.**\n\n' + reply
    if evidence.get('truncated'):
        reply += '\n\n_(한 번의 응답 길이에 도달했습니다. “이어서 설명해줘”로 계속할 수 있습니다.)_'
    evidence.update({'internet': enabled, 'network_used': bool(sources) or bool(search_error),
                     'search_attempted': plan['action'] == 'search', 'search_error': search_error,
                     'reused_sources': bool(previous_sources), 'routing_seconds': routing.get('seconds', 0)})
    output = record(text, reply, state, evidence, sources or previous_sources)
    state['messages'][-1]['knowledge'] = local_refs
    memory.save(state)
    return output
