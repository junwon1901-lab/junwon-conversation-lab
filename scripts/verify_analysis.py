"""Reproduce numeric results without an LLM or Internet access."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from conversation.engine import handle
from llm_lab import analysis, session


def compact(run):
    return {
        'selected': run['selected'], 'selection_basis': run['selection_basis'],
        'models': [{k: m.get(k) for k in ('kind', 'name', 'equation', 'metrics', 'data', 'config')} for m in run['models']],
    }


state = session.new_session()
handle('x: 1,2,3,4,5,6 y: 5,7,9,11,13,15 선형회귀 해줘', state, False)
prediction = handle('x가 10일 때 예측해줘', state, False)['evidence']['prediction']
handle('모델 3가지 비교해줘', state, False)
result = {'checked_at': session.now(), 'custom_six_points': compact(state['runs'][-1]),
          'custom_prediction': prediction, 'public_hull_data': compact(analysis.run_analysis('compare'))}
target = ROOT / 'docs' / 'numeric_results.json'
target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(target)
