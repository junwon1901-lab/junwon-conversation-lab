"""Bounded session documents and local character TF-IDF retrieval (no network)."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer


@lru_cache(maxsize=1)
def builtin_documents():
    return json.loads(Path(__file__).with_name('knowledge.json').read_text(encoding='utf-8'))


def chunks(document):
    body = document['text']
    for start in range(0, len(body), 450):
        text = body[start:start+600]
        if text.strip():
            yield {**document, 'text': text, 'chunk': start // 450 + 1}


def retrieve(query, state, limit=3):
    docs = builtin_documents() + state.get('documents', [])
    passages = [part for doc in docs for part in chunks(doc)]
    if not passages:
        return []
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), max_features=30000)
    matrix = vectorizer.fit_transform([p['title'] + ' ' + p['text'] for p in passages])
    scores = (matrix @ vectorizer.transform([query]).T).toarray().ravel()
    chosen = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:limit]
    return [{**passages[i], 'score': round(float(scores[i]), 4)} for i in chosen if scores[i] >= .08]


def manage(text, state):
    if text.startswith('자료 등록:'):
        header, sep, body = text.partition('\n')
        title = header.split(':', 1)[1].strip()
        if not sep or not 1 <= len(title) <= 80 or not 20 <= len(body.strip()) <= 12000:
            raise ValueError('자료 등록: 제목 다음 줄에 본문 20~12,000자를 붙여 넣으세요. 제목은 80자 이내입니다.')
        docs = state.setdefault('documents', [])
        if len(docs) >= 10:
            raise ValueError('자료는 대화당 10개까지입니다. 자료 삭제: 제목으로 먼저 정리하세요.')
        if any(d['title'] == title for d in docs):
            raise ValueError('같은 제목이 있습니다. 자료 삭제: 제목 후 다시 등록하세요.')
        doc = {'title': title, 'text': body.strip(), 'source': '사용자가 이 대화에 등록한 자료 · 사실 검증 전',
               'id': hashlib.sha256((title + body).encode()).hexdigest()[:12]}
        docs.append(doc)
        return f'「{title}」을 {len(list(chunks(doc)))}개 조각으로 등록했습니다. 이제 내용에 관해 질문하세요. 이 대화에서만 참고하며 LLM 가중치를 바꾸지는 않습니다.'
    if text == '자료 목록':
        return '\n\n'.join(f'• {d["title"]} — {d["source"]}' for d in builtin_documents() + state.get('documents', []))
    if text.startswith('자료 삭제:'):
        title = text.split(':', 1)[1].strip()
        docs = state.get('documents', [])
        kept = [d for d in docs if d['title'] != title]
        state['documents'] = kept
        state.pop('last_knowledge', None)
        return f'「{title}」 자료를 삭제했습니다.' if len(kept) != len(docs) else '이 대화에 등록한 해당 제목의 자료가 없습니다.'
    if text.startswith('기억해:'):
        note = text.split(':', 1)[1].strip()
        if not 1 <= len(note) <= 300:
            raise ValueError('기억할 내용은 1~300자로 적으세요.')
        notes = state.setdefault('notes', [])
        if len(notes) >= 10:
            raise ValueError('기억은 10개까지입니다. 기억 삭제 후 다시 등록하세요.')
        notes.append(note)
        return '이 대화의 기억에 저장했습니다: ' + note
    if text == '기억 보여줘':
        return '\n\n'.join('• ' + n for n in state.get('notes', [])) or '별도로 저장한 기억이 없습니다. 기억해: 내용으로 등록하세요.'
    if text == '기억 삭제':
        state['notes'] = []
        return '별도로 저장한 기억을 삭제했습니다. 앞선 채팅까지 지우려면 새 대화 시작을 입력하세요.'
    return None
