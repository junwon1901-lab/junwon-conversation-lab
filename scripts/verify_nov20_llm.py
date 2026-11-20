"""Opt-in real Qwen integration check; downloads model only with --download."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conversation.engine import handle
from conversation.model import ConversationModel
from llm_lab import session

parser = argparse.ArgumentParser()
parser.add_argument('--download', action='store_true')
args = parser.parse_args()
m = ConversationModel()
if not args.download and not m.available():
    raise SystemExit('Model missing: prepare it first or pass --download.')
s = session.new_session()
queries = [
    'x: 1,2,3,4,5,6 y: 5,7,9,11,13,15',
    '이 데이터로 직선 모델을 학습시켜서 오차가 얼마나 되는지 알려줄래?',
    'x가 10일 때 예측해줘',
    '정지한 배의 부력과 무게는 같은 거야? 왜 그런지 두 문장으로 설명해줘.',
    '자료 등록: 실험선 A\n실험선 A의 전장은 42미터이고 시험 조건은 담수이다. 목표 속도는 3노트이다.',
    '실험선 A의 전장은 얼마이고 시험 조건은 뭐야?',
    '기억해: 내 발표 주제는 선체 저항이야',
    '내 발표 주제가 뭐라고 했지?',
]
results=[]
for query in queries:
    start=time.perf_counter()
    result=handle(query,s,args.download,model=m, progress=lambda message: print('PROGRESS: '+message, flush=True))
    row={'prompt':query,'action':result['action'],'reply':result['reply'],
         'interpretation':s['messages'][-1].get('interpretation'),
         'receipt':result.get('tool_receipt'),
         'knowledge_ids':[d['id'] for d in s.get('active_knowledge',[])],
         'seconds':round(time.perf_counter()-start,2)}
    results.append(row)
    print(json.dumps(row,ensure_ascii=False),flush=True)
    Path('docs/nov20_llm_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
