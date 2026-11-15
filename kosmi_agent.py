# -*- coding: utf-8 -*-
"""
kosmi_agent.py — 프롬프트·양식·에이전트 루프(참고 구현)·자동 점검

노트북에서는 같은 루프를 my_agent()로 직접 보여 준다. 이 파일의 run_agent는
강사 검증(run_all.py)에서 쓴다. show()는 출력의 종류(코드·LLM·도구·최종 산출물)를 색으로 구분한다.
"""
import json
import os
import re
import time

import kosmi_tools as kt

# ---------------------------------------------------------------------------
# 1. 모델 설정 — base_url, api_key, model 세 줄만 바꾸면 원내 서버(vLLM 등)로 옮겨진다
# ---------------------------------------------------------------------------
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODELS = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]   # 무료 등급: 각 15 RPM, 500 RPD


def make_client(api_key=None, base_url=GEMINI_BASE_URL):
    from openai import OpenAI
    return OpenAI(api_key=api_key or os.environ.get("GEMINI_API_KEY"), base_url=base_url, max_retries=0)


CALL_LOG = []   # (시각, 모델, 입력 토큰, 출력 토큰)


TIMEOUT = 45        # 한 번의 호출이 이 초를 넘기면 다음 모델로 (서버가 느린 날 대비)
LONG_TIMEOUT = 150  # 모든 모델이 느리면 마지막으로 이만큼 기다린다


def chat(client, messages, tools=None, models=None, temperature=0.2):
    """LLM 한 번 호출. 분당 한도(429)면 잠시 쉬고, 하루 한도나 응답 지연이면 다음 모델로 넘어간다.
    모든 모델이 느리면 첫 모델로 한 번 더, 더 길게 기다린다."""
    from openai import RateLimitError, APIStatusError, APITimeoutError
    models = models or MODELS
    last = None
    plan = [(m, TIMEOUT) for m in models] + [(models[0], LONG_TIMEOUT)]
    for i, (model, timeout) in enumerate(plan):
        if i == len(models):
            if not isinstance(last, APITimeoutError):
                break
            print(f"🐢 모든 모델이 느립니다. {model}로 최대 {LONG_TIMEOUT}초 기다립니다.")
        for attempt in range(4):
            try:
                kw = dict(model=model, messages=messages, temperature=temperature, timeout=timeout)
                if tools:
                    kw["tools"] = tools
                try:
                    r = client.chat.completions.create(**kw)
                except APITimeoutError as e:
                    last = e
                    if i < len(models):
                        print(f"🐢 {model} 응답이 {timeout}초를 넘김 → 다음 모델로 전환")
                    break
                u = getattr(r, "usage", None)
                CALL_LOG.append((time.time(), model, getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0)))
                return r
            except RateLimitError as e:
                last = e
                msg = str(e).lower()
                if "per day" in msg or "perday" in msg or "daily" in msg:
                    print(f"⚠️ {model} 하루 한도 소진 → 다음 모델로 전환")
                    break
                wait = 10 * (attempt + 1)
                print(f"⏳ 분당 한도(15회/분)에 걸림. {wait}초 쉬고 다시 시도합니다.")
                time.sleep(wait)
            except APIStatusError as e:
                last = e
                if e.status_code >= 500:
                    time.sleep(5)
                    continue
                raise
    raise RuntimeError(f"모든 모델 호출 실패: {last}")


# ---------------------------------------------------------------------------
# 2. 프롬프트
# ---------------------------------------------------------------------------
SYSTEM_COMMON = """당신은 대학병원 가정의학과 비만클리닉의 진료 보조 에이전트입니다.
의사가 검토하고 고칠 '기록 초안'을 씁니다. 진단과 처방은 의사가 결정합니다.

[데이터]
- 도구는 지금 진료 중인 환자에게 묶여 있어 인자 없이 부르면 됩니다.
- 오늘은 이 환자의 진료일입니다. 도구 결과에 없는 기록은 존재하지 않는다고 보세요.
- 약물: 삭센다(0.6→3.0mg), 큐시미아(3.75/23→15/92mg), 위고비(0.25→2.4mg).
- 비만 단계(대한비만학회 2022): BMI 25/30/35 이상 = 1/2/3단계.

[반드시 지킬 규칙]
1. 모든 사실 뒤에 출처를 붙입니다. 태그는 네 가지만 씁니다: [메모](오늘 의사 메모), [설문 YYYY-MM-DD], [기록 YYYY-MM-DD], [판정](assess_response 결과).
   도구 결과에 붙은 태그를 그대로 옮기고, 도구 이름이나 다른 말로 태그를 만들지 않습니다.
   입력에 의사 메모가 없으면 [메모] 태그를 쓰지 않습니다. 출처가 없는 문장("기록 없음" 등)에는 태그를 붙이지 않습니다.
2. 체중·BMI·변화율·날짜·용량 단계·판정은 도구 결과의 값만 씁니다. 직접 계산하거나 반올림을 바꾸지 마세요.
3. 근거가 없는 칸은 "기록 없음"으로 둡니다. 추측하지 않습니다.
4. 메모·문진표(설문)·기록이 서로 어긋나면 한쪽을 고르지 말고 둘 다 적고 "확인 필요"라고 씁니다.
5. 약물 반응·6개월 목표는 assess_response 결과를 그대로 씁니다. '판정 불가'면 사유를 쓰고, 체중이 줄었어도 '반응'으로 바꾸지 않습니다.
6. 지침에 근거한 사항만 권고하고, 새 약 이름이나 용량을 지어내지 않습니다.

[지침 요점 — 대한비만학회 2022]
- 유지 용량으로 3개월 투여해도 5% 이상 감량이 없으면 약을 바꾸거나 중단한다.
- 1차 목표는 6개월 안에 치료 전 체중의 5~10% 감량이다.
- BMI 35 이상, 또는 30 이상이면서 동반질환이 있는데 생활습관·약물치료로 감량이 부족하면 비만대사수술 상담을 고려한다.
- 코골이·수면 중 무호흡 목격·주간 졸림이 있으면 수면무호흡 평가(수면다원검사)를 고려한다.
- 고혈압 진단이 없는데 오늘 혈압이 140/90 이상이면 고혈압으로 쓰지 말고 '혈압 재측정'을 계획에 넣는다.
- 지방간이 있으면서 술을 자주·많이 마시면 절주 상담을 계획에 넣는다.
- 불안·우울 척도가 중등도(11점) 이상이면 기분 평가를 계획에 넣는다.

[작업 순서]
먼저 필요한 도구를 모두 불러 사실을 모은 뒤, 아래 양식대로 한국어로 씁니다. 양식 밖의 인사말·설명은 쓰지 않습니다."""

FORM_REVISIT = """[양식: 재진 전 요약 — 진료실 들어가기 전 30초 안에 읽는 분량, 15줄 안팎]
<꺾쇠 안의 설명은 실제 값으로 바꾸고 꺾쇠는 남기지 않는다>
[{환자} · 오늘({오늘}) 진료 전]

<성별>/<나이>세 · 등록 <등록일> (<n>주 전) · BMI <기준> → <최근>

체중   <기준>kg → <최근>kg (<날짜>), <변화율>%. 최저 <kg>
약물   <약> <용량> (<n>단계 중 <m>단계), <날짜>부터
       마지막 처방 <날짜> <일수>일분 → <소진일> 소진
반응   <assess_response의 약물 반응 판정과 근거>
6개월  <assess_response의 6개월 목표 판정>
혈압   <최근 2~3회>
문진표 <진료에 중요한 신호만: 수면무호흡·졸림·우울·불안 등>
초진   <이전 진료기록 '평가 및 계획' 중 아직 결과 기록이 없는 것>

오늘 확인할 것
<1~4개. 약물 판정에 따른 결정, 지침 요점에 해당하는 항목(수술 상담·수면 평가 등), 결과가 없는 계획 순으로. 위 사실에서 나온 것만>"""

FORM_INITIAL_BASE = """[양식: 초진기록 초안]
<꺾쇠 안의 설명은 실제 값으로 바꾸고 꺾쇠는 남기지 않는다>
[{환자} · 오늘({오늘}) 초진 · 입력: 의사 메모 + 문진표 + 기존 기록]

주호소: <>

현병력:
# <진단> (BMI <값>)
- <체중력, 감량 시도, 수면, 기분, 목표 체중 등 한 줄씩>

과거력: <진단 근거가 있는 질환, 수술력>
가족력: <>
복용약: <>
생활습관: 흡연 <> / 음주 <> / 운동 <>
신체계측: 키 <cm>, 체중 <kg>, BMI <값>, 혈압 <오늘 혈압>
검사: <검사 결과(검사일) 또는 기록 없음>

평가 및 계획:
<진단> (BMI <값>)
- <메모의 계획>
- <문진표·기록에서 나온 추가 평가 필요 사항>"""

# 대조 절차: 비교할 짝을 '출력 칸'으로 만들어 하나씩 쓰게 한다 (6번 실습에서 있고 없고를 비교)
CHECK_BLOCK = """

메모·문진표 대조 <짝마다 한 줄. 결과는 '일치' 또는 '불일치 → 확인 필요' 중 하나만.
 메모에 '없음'이라고 적혀 있으면 그것이 메모 내용이다(예: 메모 '기저질환 없음' vs 문진표 '빈혈' → 불일치).
 메모가 그 항목을 아예 다루지 않을 때만 '메모 언급 없음'>
- 기저질환: 메모 <메모에 적힌 그대로> / 문진표 '진단받은 질환' <> → <결과>
- 복용약: 메모 <메모에 적힌 그대로> / 문진표 '현재 복용 중인 약' <> → <결과>
- 음주·흡연: 메모 <메모에 적힌 그대로> / 문진표 <> → <결과>
- 체중 변화: 메모 <메모에 적힌 그대로> / 문진표 <> / 기록상 1년간 변화 <> → <결과>"""

FORM_INITIAL_NOCHECK = FORM_INITIAL_BASE
FORM_INITIAL = FORM_INITIAL_BASE + CHECK_BLOCK

FORM_CLOSING = """[양식: 치료 종료 기록 — 지침 기준 치료 요약]
<꺾쇠 안의 설명은 실제 값으로 바꾸고 꺾쇠는 남기지 않는다>
[{환자} · 오늘({오늘}) 치료 종료]

치료 기간   <등록일> ~ 오늘 (<n>주), 비만약 처방 방문 <n>회
사용 약물   <약·용량 단계 변화>, 총 <n>일분 (투약 교육용 1일분 제외)
            유지 용량 <도달/미도달>
체중        기준 <kg> → 종료 <kg> (<변화율>%), 최저 <kg>
지침 판정   약물 반응: <약마다 한 줄씩, assess_response 그대로(변화율·기준 포함). 판정 불가면 사유>
            6개월 목표: <assess_response 그대로>
동반 질환   <기록·문진표에 근거한 것만. 혈압 추이(처음 → 마지막) 포함. 문진표의 질환·복용약이 기록과 어긋나면 확인 필요>
종료 사유   <기록에 있으면 쓰고, 없으면 "사유 기록 없음">
향후 계획   <기록에 없으면 "기록 없음 → 의사 입력">"""

FORMS = {"revisit": FORM_REVISIT, "initial": FORM_INITIAL, "closing": FORM_CLOSING}


def build_messages(ctx, note=None, system=SYSTEM_COMMON, form=None):
    """시스템 프롬프트 + 장면 양식 + 사용자 요청(메모/대화 포함)"""
    form = (form or FORMS[ctx.scene]).format(환자=ctx.label, 오늘=kt._d(ctx.today))
    user = f"오늘({kt._d(ctx.today)}) 진료하는 환자 {ctx.label}의 {kt.SCENES[ctx.scene]}을 위 양식대로 작성해 주세요."
    if note:
        user += f"\n\n[오늘 의사 메모 — 출처 태그 [메모]]\n{note}"
    return [{"role": "system", "content": system + "\n\n" + form}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
# 3. 에이전트 루프 (참고 구현)
# ---------------------------------------------------------------------------
def run_agent(client, ctx, messages, tools=kt.TOOL_SPECS, max_steps=8, verbose=True, models=None):
    """LLM이 도구를 부르면 실행해 결과를 돌려주고, 도구 호출이 없으면 끝낸다."""
    trace = []
    for step in range(1, max_steps + 1):
        r = chat(client, messages, tools=tools, models=models)
        msg = r.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))
        if not msg.tool_calls:
            if not (msg.content or "").strip():     # 드물게 빈 응답이 온다 → 한 번 더 요청
                messages.append({"role": "user", "content": "도구 결과를 바탕으로 양식대로 작성해 주세요."})
                continue
            if verbose:
                print(f"✅ {step}번째 호출에서 답변 완료")
            return msg.content, trace
        for tc in msg.tool_calls:
            result = kt.call_tool(ctx, tc.function.name, tc.function.arguments)
            trace.append((step, tc.function.name, tc.function.arguments, result))
            if verbose:
                first = result.split("\n")[1] if "\n" in result else result
                print(f"🔧 [{step}] {tc.function.name}({tc.function.arguments or ''}) → {first[:70]}")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    return "(최대 단계 초과 — 답변을 끝내지 못함)", trace


# ---------------------------------------------------------------------------
# 4. 자동 점검 (평가 루브릭 A: 사실 추적성)
# ---------------------------------------------------------------------------
_DATE = re.compile(r"(20\d\d)[-./](\d{1,2})(?:[-./](\d{1,2}))?")
_NUM = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(kg|%|mg|cm|점|주|일분|일)")


def _dates(text):
    out = []
    for y, m, d in _DATE.findall(text):
        try:
            out.append((int(y), int(m), int(d) if d else None))
        except ValueError:
            pass
    return out


def check_output(ctx, output, trace, note=None):
    """출처 표시, 시점 준수, 수치 근거를 자동 점검한다. 통과/주의 목록을 돌려준다."""
    output = output or ""
    evidence = "\n".join(t[3] for t in trace) + "\n" + (note or "")
    body = output.split("메모·문진표 대조")[0]      # 대조 칸은 위에서 출처를 단 사실을 다시 견주는 곳이라 출처 점검에서 뺀다
    lines = [l for l in body.splitlines() if l.strip() and not l.strip().startswith("[") and ":" not in l[:3]]
    fact_lines = [l for l in lines if re.search(r"\d", l) and not re.match(r"^\s*(\d\.|오늘 확인할 것)", l)]
    tagged = [l for l in fact_lines if re.search(r"\[(메모|설문|대화|기록|판정)[^\]]*\]", l)]
    res = []
    rate = len(tagged) / max(1, len(fact_lines))
    msg = f"숫자가 있는 줄 {len(fact_lines)}개 중 {len(tagged)}개에 출처 태그 ({rate:.0%})"
    bogus = [t for t in ("메모", "대화") if f"[{t}" in output and not (note and t == "메모")]
    if bogus:
        msg += f" · 입력에 없는 출처 [{'], ['.join(bogus)}]"
    res.append(("출처 표시", rate >= 0.8 and not bogus, msg))
    t = ctx.today
    odd = []
    for y, m, d in _dates(output):
        s = f"{y}-{m:02d}" + (f"-{d:02d}" if d else "")
        if (y, m, d or 1) > (t.year, t.month, t.day if d else 1) and s not in evidence:   # 도구가 계산한 소진 예정일 등은 통과
            odd.append(s)
    odd = list(dict.fromkeys(odd))
    res.append(("날짜 근거", not odd, "오늘 이후의 지어낸 날짜 없음" if not odd else f"근거 없는 미래 날짜: {', '.join(odd)}"))
    unknown = []
    for v, unit in _NUM.findall(output):
        if v not in evidence and v.rstrip("0").rstrip(".") not in evidence:
            unknown.append(f"{v}{unit}")
    unknown = list(dict.fromkeys(unknown))
    res.append(("수치 근거", not unknown, "모든 수치가 도구 결과·입력에 있음" if not unknown else f"근거를 못 찾은 수치: {', '.join(unknown)}"))
    return res


def print_check(res):
    show("code", "\n".join(f"{'✅' if ok else '⚠️'} {name}: {msg}" for name, ok, msg in res), title="자동 점검")


# ---------------------------------------------------------------------------
# 5. 보기 좋게 출력하기, 실습 입력
# ---------------------------------------------------------------------------
# 출력 상자: 색과 머리말로 "누가 만든 결과인가"를 구분한다 (노트북에서는 큰 글씨 HTML, 그 밖에서는 텍스트)
BOX = {
    "code":  ("#F1F3F4", "#80868B", "🧑‍💻 코드 실행 결과 · LLM 사용 안 함"),
    "llm":   ("#E8F0FE", "#3B6FD8", "🤖 LLM"),
    "tool":  ("#FEF3E2", "#E8912D", "🔧 도구 결과 · LLM이 요청 → 코드가 실행해 LLM에게 돌려줌"),
    "final": ("#E6F4EA", "#34A853", "📄 최종 산출물 · LLM이 작성"),
}
FONT_PX = 16      # 상자 안 글씨 크기. 더 키우려면 ka.FONT_PX = 18


def _in_notebook():
    try:
        from IPython import get_ipython
        shell = get_ipython()           # Colab: google.colab._shell.Shell, Jupyter: ZMQInteractiveShell
        return shell is not None and type(shell).__name__ != "TerminalInteractiveShell"
    except Exception:
        return False


def show(kind, body, title=None):
    """kind: code(코드 실행) · llm(LLM 응답·요청) · tool(도구 결과) · final(최종 산출물)"""
    bg, border, label = BOX[kind]
    if kind in ("llm", "final"):
        body = str(body).replace("**", "")      # LLM이 붙인 마크다운 굵게 표시는 상자에서 뺀다
    head = label + (f"  —  {title}" if title else "")
    if not _in_notebook():
        print(f"──── {head} ────\n{body}\n")
        return
    import html
    from IPython.display import HTML, display
    display(HTML(
        f'<div style="background:{bg};border-left:6px solid {border};border-radius:6px;padding:10px 14px;margin:6px 0;color:#202124">'
        f'<div style="font-weight:700;font-size:{FONT_PX - 1}px;margin-bottom:6px">{html.escape(head)}</div>'
        f'<pre style="white-space:pre-wrap;word-break:break-word;font-size:{FONT_PX}px;line-height:1.55;margin:0;'
        f'background:transparent;color:#202124;font-family:\'Noto Sans Mono\',\'D2Coding\',monospace">{html.escape(str(body))}</pre></div>'))


_INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inputs")
NOTE_FILES = {"P2": "memo_P2.txt"}     # 초진 환자 P2의 오늘 의사 메모


def load_note(patient):
    f = NOTE_FILES.get(patient)
    return open(os.path.join(_INPUT_DIR, f), encoding="utf-8").read().strip() if f else None


def show_trace(trace, full=False):
    """도구 호출 기록. full=True면 도구 결과 전체를 보여준다."""
    for step, name, args, result in trace:
        lines = result.splitlines()
        body = result if full else "\n".join(lines[1:6]) + ("\n…" if len(lines) > 6 else "")
        show("tool", body, title=f"[{step}번째 호출] {name}")


def usage_summary():
    n = len(CALL_LOG)
    ti = sum(c[2] or 0 for c in CALL_LOG)
    to = sum(c[3] or 0 for c in CALL_LOG)
    by = {}
    for c in CALL_LOG:
        by[c[1]] = by.get(c[1], 0) + 1
    return f"이 세션에서 LLM 호출 {n}회 (모델별 {by}), 입력 {ti:,} 토큰, 출력 {to:,} 토큰. 무료 한도: 모델별 하루 500회."
