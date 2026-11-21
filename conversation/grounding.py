"""Explicit checks for observed contradictions; not a general truth verifier."""
import re


def check_analysis(text, receipt):
    prediction = receipt.get('prediction') or {}
    issues = []
    if prediction.get('outside_training_range') and re.search(r'범위\s*(?:내|안)|외삽이\s*아니', text):
        issues.append('학습 범위 밖의 입력을 범위 안으로 설명')
    if re.search(r'완벽|100\s*%|오차가?\s*전혀\s*없|외삽이\s*불가능', text):
        issues.append('평가 결과의 과도한 일반화')
    if not issues:
        return text, []
    if prediction:
        corrected = f"{prediction['model']}의 계산값은 {prediction['value']:.6g}입니다. "
        corrected += ('학습 범위 밖의 입력이므로 외삽입니다. 실제 정답이 없어 이 입력의 예측 오차는 아직 알 수 없습니다.' if prediction.get('outside_training_range') else '이 입력의 실제 정답이 없다면 예측 오차를 확정할 수 없습니다.')
    else:
        corrected = '이번 데이터에서 측정한 오차가 작더라도 새로운 데이터에서의 정확성을 보장하지 않습니다. R²는 정답률이 아니며, 화면에 표시된 평가 분할과 표본 수를 함께 확인해야 합니다.'
    return '생성 설명에서 계산 근거와 충돌하는 표현을 발견해 다음의 검증 규칙 기반 설명으로 대체했습니다.\n\n' + corrected, issues


def check_documents(text, documents):
    source_index = next((i for i, d in enumerate(documents, 1) if d.get('id') == 'buoyancy'), None)
    if source_index and (
        re.search(r'체적.{0,35}(?:변화하지|변하지|무관)', text) or
        re.search(r'부력.{0,12}(?:무게|중력)보다\s*(?:커야|크다|큽니다)', text)
    ):
        return ('생성 설명의 부력 관계를 참고 자료와 대조해 수정했습니다. 정지해서 떠 있는 배는 수직 가속도가 0이므로 부력과 무게가 같습니다. '
                '부력은 F_b=ρg∇로 잠긴 체적에 비례합니다. 부력이 무게보다 크고 다른 힘이 없다면 위로 가속하며, 정지 부유 평형과는 다릅니다. ' + f'[자료{source_index}]'), ['부력과 체적 또는 평형 관계 충돌']
    return text, []
