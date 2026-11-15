# -*- coding: utf-8 -*-
"""
kosmi_tools.py — 비만클리닉 doctor agent 실습용 데이터·도구 모듈 (KOSMI 2026 튜토리얼)

핵심 설계
  * 환자 한 명을 연 Context를 먼저 만들고, 도구는 그 Context에 묶인다.
    → LLM은 지금 진료 중인 환자의 기록만 볼 수 있다(다른 환자를 조회할 방법이 없다).
  * 계산(체중 변화율, 용량 단계, 지침 판정)은 전부 도구가 한다. LLM은 고르고, 읽고, 쓴다.
  * 도구 결과의 각 사실에는 출처 태그([기록 날짜], [설문 날짜])가 이미 붙어 있다.

데이터는 모두 합성 데이터다(실제 환자 아님). 실습 환자 3명, 진료일은 모두 오늘(TODAY). data/ 폴더의 표 7개:
  patients        patient_id, sex, birth_year
  diagnoses       patient_id, date, code, name
  prescriptions   patient_id, date, drug, generic, dose, days
  vitals          patient_id, date, weight_kg, height_cm, bmi, sbp, dbp   (측정 1회 = 1행)
  questionnaire   patient_id, date, no, question, answer                  (체중관리 문진표 21문항)
  notes           patient_id, date, section, text                         (초진기록 자유서술)
  labs            patient_id, date, test, value, unit
"""
import json
import os
from datetime import timedelta

import pandas as pd

# ---------------------------------------------------------------------------
# 0. 데이터 적재
# ---------------------------------------------------------------------------
DATA_DIR = os.environ.get("KOSMI_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
TABLES = ["patients", "diagnoses", "prescriptions", "vitals", "questionnaire", "notes", "labs"]
_DB = None


def load_data(data_dir=None):
    """합성 데이터 표 7개를 dict로 읽는다. 여러 번 불러도 한 번만 읽는다."""
    global _DB, DATA_DIR
    if data_dir:
        DATA_DIR, _DB = data_dir, None
    if _DB is None:
        db = {}
        for t in TABLES:
            df = pd.read_csv(os.path.join(DATA_DIR, f"{t}.csv"), dtype=str, encoding="utf-8-sig")
            if "date" in df:
                df["date"] = pd.to_datetime(df["date"])
            db[t] = df
        for c in ("weight_kg", "height_cm", "bmi", "sbp", "dbp"):
            db["vitals"][c] = pd.to_numeric(db["vitals"][c])
        db["prescriptions"]["days"] = db["prescriptions"]["days"].astype(int)
        _DB = db
    return _DB


# 오늘 = 실습 환자 3명의 진료일 (강의 날)
TODAY = pd.Timestamp("2026-11-25")

# 실습 장면
SCENES = {
    "revisit": "재진 요약",
    "initial": "초진기록 초안",
    "closing": "치료 종료 기록",
}
# 실습 환자 3명: 환자마다 오늘 할 일(장면)이 정해져 있다
PATIENTS = {
    "P1": ("revisit", "삭센다 2.4mg 4개월째 재진. 진료 전 요약"),
    "P2": ("initial", "처음 온 환자. 의사 메모 + 문진표로 초진기록"),
    "P3": ("closing", "위고비 1.0mg에서 중단. 치료 종료 기록"),
}

# ---------------------------------------------------------------------------
# 1. 약물 정보
# ---------------------------------------------------------------------------
DRUG_KO = {"Saxenda": "삭센다", "Qsymia": "큐시미아", "Wegovy": "위고비"}
STEPS = {
    "Saxenda": ["0.6mg", "1.2mg", "1.8mg", "2.4mg", "3.0mg"],
    "Qsymia": ["3.75/23mg", "7.5/46mg", "11.25/69mg", "15/92mg"],
    "Wegovy": ["0.25mg", "0.5mg", "1.0mg", "1.7mg", "2.4mg"],
}
# 유지 용량 기준 (실습 정의): 이 단계 이상이면 유지 용량으로 본다
MAINT_STEP = {"Saxenda": 1, "Qsymia": 1, "Wegovy": 3}


def _step_of(drug, dose):
    """용량 문자열 → 증량 단계 번호(0부터)"""
    if drug == "Saxenda":   # 삭센다는 0.6mg 단위라 숫자로 비교
        return [0.6, 1.2, 1.8, 2.4, 3.0].index(round(float(dose.replace("mg", "")), 1))
    return STEPS[drug].index(dose)


def bmi_class(bmi):
    """대한비만학회 2022 기준 비만 단계"""
    if bmi is None:
        return "알 수 없음"
    if bmi >= 35:
        return "비만 3단계, Obesity class III"
    if bmi >= 30:
        return "비만 2단계, Obesity class II"
    if bmi >= 25:
        return "비만 1단계, Obesity class I"
    if bmi >= 23:
        return "비만 전단계, Overweight"
    return "정상 범위"


def _d(x):
    return pd.Timestamp(x).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# 2. Context: 진료 중인 환자 한 명의 기록을 연다
# ---------------------------------------------------------------------------
class Context:
    """
    환자 한 명의 기록 묶음. 오늘(TODAY) 할 일(scene)은 환자마다 정해져 있다.
      revisit (재진 요약)   : 오늘 진료 전에 지난 기록을 요약
      initial (초진기록)    : 처음 온 환자. 오늘 낸 문진표·오늘 잰 활력징후 + 의사 메모로 초진기록 작성
      closing (종료 기록)   : 오늘이 마지막 진료. 오늘 기록까지 포함해 치료 전체를 정리
    """

    def __init__(self, patient):
        db = load_data()
        if patient not in PATIENTS:
            raise ValueError(f"실습 환자는 {list(PATIENTS)} 중 하나")
        self.pid = self.label = patient
        self.scene = PATIENTS[patient][0]
        self.today = TODAY
        info = db["patients"][db["patients"]["patient_id"] == patient]
        self.sex = {"F": "여", "M": "남"}[info["sex"].iloc[0]]
        self.age = self.today.year - int(info["birth_year"].iloc[0])
        # 표마다 이 환자의 행만 남긴다
        self.t = {name: db[name][db[name]["patient_id"] == patient].sort_values("date", kind="stable")
                  for name in TABLES[1:]}

    def header(self):
        return f"[{self.label} · {SCENES[self.scene]} · 오늘 {_d(self.today)}]"

    # ---- 내부 표 ----
    def _rx(self):
        rx = self.t["prescriptions"]
        rx = rx[rx["drug"].isin(STEPS)].copy()
        rx["step"] = [_step_of(d, s) for d, s in zip(rx["drug"], rx["dose"])]
        return rx

    def _weights(self):
        w = self.t["vitals"].dropna(subset=["weight_kg"])
        return w.groupby("date")["weight_kg"].mean().reset_index().rename(columns={"weight_kg": "kg"})

    def _height(self):
        h = self.t["vitals"].dropna(subset=["height_cm"])
        return float(h["height_cm"].iloc[-1]) if not h.empty else None

    def registration_date(self):
        """등록일: 첫 진료기록·첫 처방 중 이른 날. 없으면 첫 진단일. 초진 환자는 오늘."""
        if self.scene == "initial":
            return self.today
        firsts = [self.t[n]["date"].min() for n in ("notes", "prescriptions") if not self.t[n].empty]
        if firsts:
            return min(firsts)
        dx = self.t["diagnoses"]
        return dx["date"].min() if not dx.empty else None


def _nearest(w, target, tol):
    """target ±tol일 안에서 가장 가까운 체중 (같은 거리면 이른 날)"""
    c = w[(w["date"] - target).abs() <= timedelta(days=tol)].copy()
    if c.empty:
        return None
    c["dd"] = (c["date"] - target).abs()
    r = c.sort_values(["dd", "date"]).iloc[0]
    return r["date"], float(r["kg"])


# ---------------------------------------------------------------------------
# 3. 도구 (모두 문자열을 돌려준다 — LLM이 그대로 읽는다)
# ---------------------------------------------------------------------------
def get_patient_summary(ctx):
    """기본 정보: 성별·나이·키, 등록일, 진단 목록, 방문 수"""
    reg = ctx.registration_date()
    lines = [ctx.header(), f"성별/나이: {ctx.sex}/{ctx.age}세"]
    h = ctx._height()
    lines.append(f"키: {h:.1f}cm [기록]" if h else "키: 기록 없음")
    if reg is not None:
        weeks = (ctx.today - reg).days // 7
        lines.append(f"비만클리닉 등록일: {_d(reg)}" + (" (오늘 첫 진료)" if ctx.scene == "initial" else f" (오늘까지 {weeks}주)"))
    dx = ctx.t["diagnoses"]
    if dx.empty:
        lines.append("진단: 기록 없음")
    else:
        g = dx.groupby(["code", "name"]).agg(first=("date", "min"), last=("date", "max"), n=("date", "count")).reset_index()
        lines.append("진단:")
        for _, r in g.sort_values("first").iterrows():
            lines.append(f"  - {r['code']} {r['name']} (첫 기록 {_d(r['first'])}, 마지막 {_d(r['last'])}, {r['n']}회) [기록]")
    visits = set().union(*[set(ctx.t[n]["date"]) for n in ("diagnoses", "prescriptions", "vitals", "notes")])
    lines.append(f"방문 날짜 수: {len(visits)}일 (그중 비만약 처방 방문 {ctx.t['prescriptions']['date'].nunique()}일)")
    if ctx.scene == "initial":
        lines.append("※ 초진: 오늘 진단·처방·진료기록은 지금 작성 중이라 아직 없음. 오늘 낸 문진표와 오늘 잰 활력징후는 있음.")
    return "\n".join(lines)


def get_weight_trend(ctx):
    """체중 추이: 기준 체중, 최근 체중, 변화율, 최저, BMI와 비만 단계, 측정 목록"""
    w = ctx._weights()
    if w.empty:
        return ctx.header() + "\n체중 기록 없음"
    h = ctx._height()
    bmi = lambda kg: round(kg / (h / 100) ** 2, 1) if h else None
    lines = [ctx.header()]
    reg = ctx.registration_date()
    base = _nearest(w, reg, 14) if reg is not None else None
    last = w.iloc[-1]
    if ctx.scene == "initial":
        today = w[w["date"] == ctx.today]
        if not today.empty:
            kg = float(today["kg"].iloc[0])
            lines.append(f"오늘 체중: {kg:.1f}kg, BMI {bmi(kg)} ({bmi_class(bmi(kg))}) [기록 {_d(ctx.today)}]")
        prev = w[w["date"] < ctx.today]
        yr = _nearest(prev, ctx.today - timedelta(days=365), 60)
        if yr:
            lines.append(f"약 1년 전 체중: {yr[1]:.1f}kg [기록 {_d(yr[0])}]")
            if not today.empty:
                lines.append(f"  → 1년간 변화: {float(today['kg'].iloc[0]) - yr[1]:+.1f}kg (기록 기준)")
        else:
            lines.append("약 1년 전 체중: 기록 없음")
    else:
        if base:
            ch = 100 * (float(last["kg"]) - base[1]) / base[1]
            lines.append(f"기준 체중(등록일 ±14일): {base[1]:.1f}kg, BMI {bmi(base[1])} [기록 {_d(base[0])}]")
            lines.append(f"최근 체중: {float(last['kg']):.1f}kg, BMI {bmi(float(last['kg']))} ({bmi_class(bmi(float(last['kg'])))}) [기록 {_d(last['date'])}]")
            lines.append(f"기준 대비 변화: {float(last['kg']) - base[1]:+.1f}kg ({ch:+.1f}%)")
            after = w[w["date"] >= base[0]]
            mn = after.loc[after["kg"].idxmin()]
            lines.append(f"등록 후 최저: {float(mn['kg']):.1f}kg [기록 {_d(mn['date'])}]")
        else:
            lines.append("기준 체중: 등록일 ±14일 안에 체중 기록 없음 → 변화율 계산 불가")
            lines.append(f"최근 체중: {float(last['kg']):.1f}kg [기록 {_d(last['date'])}]")
    show = w.tail(12)
    lines.append(f"측정 목록 (최근 {len(show)}회 / 전체 {len(w)}회):")
    lines.append("  " + ", ".join(f"{_d(r['date'])} {r['kg']:.1f}" for _, r in show.iterrows()))
    return "\n".join(lines)


def get_medication_timeline(ctx):
    """비만약 처방 타임라인: 약별 용량 단계, 현재 단계·다음 단계, 마지막 처방 소진일"""
    rx = ctx._rx()
    if rx.empty:
        msg = "비만약 처방 기록 없음"
        if ctx.scene == "initial":
            msg += " (초진: 오늘 처방은 아직 없음 — 메모의 처방 계획을 사용)"
        return ctx.header() + "\n" + msg
    lines = [ctx.header()]
    edu = rx[rx["days"] <= 1]          # 주사제 첫 방문의 원내 투약 교육용 1일 처방
    rx = rx[rx["days"] > 1]
    for drug in dict.fromkeys(rx["drug"]):
        rd = rx[rx["drug"] == drug]
        steps = STEPS[drug]
        lines.append(f"■ {DRUG_KO[drug]} ({len(steps)}단계: {' → '.join(steps)})")
        for _, r in rd.iterrows():
            end = r["date"] + timedelta(days=int(r["days"]))
            lines.append(f"  {_d(r['date'])} {steps[r['step']]} {r['days']}일분 (소진 {_d(end)}) [기록 {_d(r['date'])}]")
        top = int(rd["step"].max())
        cur = rd.iloc[-1]
        first_top = rd[rd["step"] == top].iloc[0]["date"]
        lines.append(f"  최고 도달 용량: {steps[top]} ({len(steps)}단계 중 {top + 1}단계), {_d(first_top)}부터")
        lines.append("  유지 용량 도달: " + ("예" if top >= MAINT_STEP[drug] else f"아니오 (유지 용량 기준 {steps[MAINT_STEP[drug]]} 이상)"))
        nxt = steps[int(cur["step"]) + 1] if int(cur["step"]) + 1 < len(steps) else "없음(최고 용량)"
        lines.append(f"  마지막 처방 용량: {steps[int(cur['step'])]}, 다음 단계: {nxt}")
        lines.append(f"  처방일수 합계: {int(rd['days'].sum())}일")
    last = rx.iloc[-1]
    end = last["date"] + timedelta(days=int(last["days"]))
    gap = (end - ctx.today).days
    status = f"오늘 기준 {gap}일분 남음" if gap > 0 else ("오늘 소진" if gap == 0 else f"{-gap}일 전에 소진")
    lines.append(f"마지막 처방: {_d(last['date'])} {DRUG_KO[last['drug']]} {STEPS[last['drug']][int(last['step'])]} {last['days']}일분 → 소진 {_d(end)} ({status})")
    if rx["drug"].nunique() > 1:
        lines.append("약 전환 있음: " + " → ".join(DRUG_KO[x] for x in dict.fromkeys(rx["drug"])))
        lines.append(f"전체 비만약 처방일수 합계: {int(rx['days'].sum())}일 (모든 약 합산)")
    if not edu.empty:
        lines.append(f"(주사제 투약 교육용 1일 처방 {len(edu)}건은 위 계산에서 제외)")
    return "\n".join(lines)


def get_vitals(ctx, n=6):
    """최근 혈압 측정 (같은 날 여러 번이면 모두)"""
    bp = ctx.t["vitals"].dropna(subset=["sbp"])
    if bp.empty:
        return ctx.header() + "\n혈압 기록 없음"
    n = int(n or 6)
    lines = [ctx.header(), f"혈압 (최근 {min(n, len(bp))}회 / 전체 {len(bp)}회):"]
    for _, r in bp.tail(n).iterrows():
        lines.append(f"  {int(r['sbp'])}/{int(r['dbp'])} mmHg [기록 {_d(r['date'])}]")
    hi = bp[(bp["sbp"] >= 140) | (bp["dbp"] >= 90)]
    lines.append(f"140/90 이상: 전체 {len(bp)}회 중 {len(hi)}회")
    return "\n".join(lines)


def _score_label(no, v):
    v = int(v)
    if no == "Q19":   # 주간 졸림 (ESS)
        return "정상" if v <= 10 else "경도 주간졸림" if v <= 12 else "중등도 주간졸림" if v <= 15 else "심한 주간졸림"
    return "정상" if v <= 7 else "경계(mild)" if v <= 10 else "중등도(moderate)" if v <= 14 else "심함(severe)"   # HADS


def get_questionnaire(ctx):
    """체중관리 문진표(가장 최근 것): 체중력·질환·복용약·가족력·흡연·음주·수면, 졸림·불안·우울 점수와 해석"""
    q = ctx.t["questionnaire"]
    if q.empty:
        return ctx.header() + "\n문진표 기록 없음"
    last = q["date"].max()
    q = q[q["date"] == last]
    tag = f"[설문 {_d(last)}]"
    lines = [ctx.header(), f"체중관리 문진표 작성일: {_d(last)} (가장 최근)"]
    for _, r in q.iterrows():
        ans = r["answer"]
        if r["no"] in ("Q19", "Q20", "Q21"):
            ans = f"{ans}점 → {_score_label(r['no'], ans)}"
        lines.append(f"{r['no']} {r['question']}: {ans} {tag}")
    return "\n".join(lines)


def get_previous_notes(ctx):
    """이전 진료기록(초진기록)의 자유서술: 주호소·현병력·과거력·가족력·복용약·평가 및 계획"""
    d = ctx.t["notes"]
    if d.empty:
        return ctx.header() + "\n이전 진료기록 없음" + (" (초진: 오늘 기록은 지금 작성 중)" if ctx.scene == "initial" else "")
    lines = [ctx.header()]
    for date, g in d.groupby("date"):
        lines.append(f"■ 초진기록 {_d(date)} [기록 {_d(date)}]")
        for _, r in g.iterrows():
            lines.append(f"  {r['section']}: " + str(r["text"]).replace("\n", "\n    "))
    return "\n".join(lines)


def get_labs(ctx):
    """혈액검사 결과 (검사일별)"""
    d = ctx.t["labs"]
    if d.empty:
        return ctx.header() + "\n검사 결과 기록 없음"
    lines = [ctx.header()]
    for date, g in d.groupby("date"):
        lines.append(f"■ 검사일 {_d(date)} [기록 {_d(date)}]")
        lines.append("  " + ", ".join(f"{r['test']} {r['value']} {r['unit']}" for _, r in g.iterrows()))
    return "\n".join(lines)


# ---- 지침 판정 (judge.py와 같은 정의) ----
def assess_response(ctx):
    """대한비만학회 2022 기반 실습 정의로 약물 반응(유지 용량 3개월 시점 5%)과 6개월 목표를 판정"""
    if ctx.scene == "initial":
        return ctx.header() + "\n초진: 치료 시작 전이므로 판정 대상 아님"
    w = ctx._weights()
    rx = ctx._rx()
    rx = rx[rx["days"] > 1]
    end = ctx.today
    reg = ctx.registration_date()
    base = _nearest(w, reg, 14) if (reg is not None and not w.empty) else None
    lines = [ctx.header(), "판정 정의: 유지 용량 첫 처방일 + 84일 ±28일 체중이 기준 대비 -5% 이하면 '반응', 아니면 '무반응'.",
             "          6개월 목표 = 등록일 + 182일 ±28일 체중: 5% 미만 미달, 5~10% 달성, 10% 초과 초과 달성."]
    # 6개월 목표
    if base is None:
        lines.append("6개월 목표: 판정 불가 (등록일 ±14일 체중 없음)")
    elif reg + timedelta(days=182 - 28) > end:
        lines.append(f"6개월 목표: 판정 불가 (등록 후 {(end - reg).days}일, 182일 미경과)")
    else:
        m6 = _nearest(w, reg + timedelta(days=182), 28)
        if m6 is None:
            lines.append("6개월 목표: 판정 불가 (182일 ±28일 안에 체중 없음)")
        else:
            ch = 100 * (m6[1] - base[1]) / base[1]
            res = "미달" if ch > -5 else ("달성" if ch >= -10 else "초과 달성")
            lines.append(f"6개월 목표: {res} ({ch:+.1f}%, {base[1]:.1f} → {m6[1]:.1f}kg) [기록 {_d(m6[0])}]")
    if rx.empty:
        lines.append("약물 반응: 약물 없음")
        return "\n".join(lines)
    # 약물 반응: 약마다 따로
    for i, drug in enumerate(dict.fromkeys(rx["drug"])):
        rd = rx[rx["drug"] == drug]
        first = rd["date"].min()
        b = base if i == 0 else _nearest(w, first, 14)   # 두 번째 약의 기준 = 그 약 첫 처방일 ±14일 체중
        top = int(rd["step"].max())
        head = f"약물 반응 [{DRUG_KO[drug]}]:"
        if b is None:
            lines.append(f"{head} 판정 불가 (기준 체중 없음)")
            continue
        if top < MAINT_STEP[drug]:
            last = rd.iloc[-1]
            ongoing = last["date"] + timedelta(days=int(last["days"])) > end
            lines.append(f"{head} 판정 불가 ({'아직 증량 중' if ongoing else '유지 용량 전 중단'}; 최고 {STEPS[drug][top]}, 유지 기준 {STEPS[drug][MAINT_STEP[drug]]} 이상)")
            if not w.empty:
                ch = 100 * (float(w.iloc[-1]["kg"]) - b[1]) / b[1]
                lines.append(f"  참고: 최근 체중 변화 {ch:+.1f}% — 정의상 '반응'으로 쓰지 않음")
            continue
        md = rd[rd["step"] == top].iloc[0]["date"]
        if md + timedelta(days=84 - 28) > end:
            lines.append(f"{head} 판정 불가 (유지 용량 {STEPS[drug][top]} 도달 {_d(md)}, 84일 미경과)")
            continue
        m3 = _nearest(w, md + timedelta(days=84), 28)
        if m3 is None:
            lw = w["date"].max() if not w.empty else None
            why = "유지 용량 3개월 전 체중 추적 끊김" if (lw is None or lw < md + timedelta(days=84 - 28)) else "84일 ±28일 안에 체중 없음"
            lines.append(f"{head} 판정 불가 ({why})")
            continue
        ch = 100 * (m3[1] - b[1]) / b[1]
        res = "반응" if ch <= -5 else "무반응 → 지침상 약 변경·중단 권고 대상"
        lines.append(f"{head} {res} ({ch:+.1f}%, 기준 {b[1]:.1f}kg [기록 {_d(b[0])}] → {m3[1]:.1f}kg [기록 {_d(m3[0])}]; 유지 용량 {STEPS[drug][top]} 도달 {_d(md)} + 84일 시점)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. LLM에 넘길 도구 명세 (OpenAI 호환 function calling 형식)
#    LLM은 이 '이름'과 '설명'만 보고 어떤 도구를 부를지 고른다.
# ---------------------------------------------------------------------------
def _schema(name, desc, props=None):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props or {}, "required": []}}}


TOOL_SPECS = [   # 도구는 모두 '지금 연 환자'에 묶여 있어 인자로 환자를 받지 않는다
    _schema("get_patient_summary", "환자 기본 정보: 성별·나이·키, 비만클리닉 등록일, 진단 목록, 방문 수."),
    _schema("get_weight_trend", "체중 추이: 기준 체중, 최근 체중, 변화율(%), 최저 체중, BMI와 비만 단계, 측정 목록. 체중·BMI·변화율은 반드시 이 도구 값만 쓴다."),
    _schema("get_medication_timeline", "비만약(삭센다·큐시미아·위고비) 처방 타임라인: 날짜별 용량·처방일수·소진일, 최고 도달 용량, 현재·다음 증량 단계, 약 전환."),
    _schema("get_vitals", "최근 혈압 측정 목록과 140/90 이상 횟수.",
            {"n": {"type": "integer", "description": "보여줄 최근 측정 횟수 (기본 6)"}}),
    _schema("get_questionnaire", "환자가 작성한 체중관리 문진표: 체중력·감량 시도·진단받은 질환·복용약·수술·가족력·흡연·음주·코골이·수면 중 무호흡·수면, 졸림·불안·우울 점수와 해석."),
    _schema("get_previous_notes", "이전 진료기록(초진기록)의 주호소·현병력·과거력·가족력·복용약·평가 및 계획."),
    _schema("get_labs", "혈액검사 결과(지질, 공복혈당, HbA1c, 간기능, TSH)를 검사일별로."),
    _schema("assess_response", "대한비만학회 2022 기반 실습 정의로 약물 반응(유지 용량 3개월 시점 5% 감량)과 6개월 목표 달성을 판정. 판정·판정 불가 사유는 반드시 이 도구 결과를 쓴다."),
]

TOOL_FUNCS = {
    "get_patient_summary": get_patient_summary,
    "get_weight_trend": get_weight_trend,
    "get_medication_timeline": get_medication_timeline,
    "get_vitals": get_vitals,
    "get_questionnaire": get_questionnaire,
    "get_previous_notes": get_previous_notes,
    "get_labs": get_labs,
    "assess_response": assess_response,
}


def call_tool(ctx, name, args_json):
    """LLM이 고른 도구를 실행한다. 인자 오류도 문자열로 돌려줘서 LLM이 고칠 수 있게 한다."""
    try:
        args = json.loads(args_json or "{}") or {}
        if name not in TOOL_FUNCS:
            return f"오류: '{name}'라는 도구는 없음. 사용 가능: {list(TOOL_FUNCS)}"
        return TOOL_FUNCS[name](ctx, **args)
    except Exception as e:
        return f"오류: {type(e).__name__}: {e}"
