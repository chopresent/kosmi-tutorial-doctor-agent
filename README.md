# 비만클리닉 doctor agent 실습 (대한의료정보학회 2026 튜토리얼)

의사가 환자를 열면 **재진 전 요약 · 초진기록 초안 · 치료 종료 기록**을 써 주는 LLM 에이전트를 만들어 보는 실습 자료입니다.
**모든 환자 데이터는 합성 데이터입니다.** 실제 환자 정보는 이 코드나 무료 API에 넣지 마세요.

## 바로 시작
1. [Google AI Studio](https://aistudio.google.com/apikey)에서 API 키를 만듭니다 (무료, 결제 정보 없음).
2. Colab에서 `notebooks/실습.ipynb`를 엽니다 → `파일 → Drive에 사본 저장`.
3. 왼쪽 🔑(보안 비밀)에 `GEMINI_API_KEY`를 등록하고 위에서부터 한 셀씩 실행합니다.

## 구성
| 경로 | 내용 |
|---|---|
| `kosmi_tools.py` | 데이터 적재, 환자 열기(`Context`), 도구 8개, 도구 명세 |
| `kosmi_agent.py` | 모델 호출(한도 처리), 시스템 프롬프트·양식 3종, 에이전트 루프, 자동 점검, 출력 상자(show) |
| `data/` | 합성 환자 3명의 표 7개 (아래) |
| `inputs/` | 초진 환자 P2의 의사 메모 |
| `notebooks/실습.ipynb` | 실습 노트북 (섹션 0~8 + 9. 시간이 남으면: 선택 블록 7개) |
| `notebooks/실습_결과포함.ipynb` | 실행 결과가 저장된 읽기용 노트북 (API가 안 될 때) |
| `replay/recorded.json` | 녹화 재생용 실제 응답 (API가 안 될 때 `ka.REPLAY = True`) |
| `figures/` | 에이전트 구조도, 도구 요청서 왕복, 에이전트 루프 그림 (슬라이드용 PNG) |

## 데이터 (`data/`, 모두 합성, 진료일은 모두 2026-11-25)
| 파일 | 열 |
|---|---|
| `patients.csv` | patient_id, sex, birth_year |
| `diagnoses.csv` | patient_id, date, code, name (KCD) |
| `prescriptions.csv` | patient_id, date, drug, generic, dose, days |
| `vitals.csv` | patient_id, date, weight_kg, height_cm, bmi, sbp, dbp (측정 1회 = 1행) |
| `questionnaire.csv` | patient_id, date, no, question, answer (체중관리 문진표 21문항) |
| `notes.csv` | patient_id, date, section, text (초진기록 자유서술) |
| `labs.csv` | patient_id, date, test, value, unit |

## API가 안 될 때
노트북 1번 섹션의 `ka.REPLAY = True` 셀을 켜면, 강의 전에 실제 Gemini로 녹화한 응답으로 끝까지 따라갈 수 있습니다(도구 실행·점검은 실제로 돎). 켜지 않아도 API 호출이 실패하면 녹화가 있는 셀은 자동으로 녹화로 대신합니다. 녹화는 원래 코드·질문 그대로일 때만 맞습니다.

## 모델
기본은 Gemini 3.5 Flash-Lite(무료 등급: 분당 15회, 하루 500회), 하루 한도가 끝나면 3.1 Flash-Lite로 넘어갑니다.
OpenAI 호환 API라서 원내 GPU 서버(vLLM 등)로 옮길 때는 `base_url`, `api_key`, 모델 이름만 바꿉니다.

```python
client = OpenAI(base_url="http://내부서버:8000/v1", api_key="...")
ka.MODELS = ["내부-모델-이름"]
```

## 오늘(2026-11-25)의 환자
| 환자 | 할 일 | 포인트 |
|---|---|---|
| P1 (남, 27세) | 재진 전 요약 | 삭센다 무반응 -4.0%, 수면무호흡 신호, 3단계 비만 |
| P2 (여, 40세) | 초진기록 초안 | 의사 메모 ↔ 문진표·기록 불일치 3곳(기저질환·음주·체중) |
| P3 (남, 35세) | 치료 종료 기록 | -5.7%인데 유지 용량 전 중단 → 약물 반응 판정 불가 |

판정 정의는 대한비만학회 2022 진료지침을 바탕으로 한 실습용 정의입니다(`assess_response` 참고).
