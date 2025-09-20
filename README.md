# RAG 평가 시스템 (RAG Evaluation System)

이 프로젝트는 RAG(Retrieval-Augmented Generation) 기반 챗봇의 성능을 4가지 핵심 지표로 평가하는 종합적인 평가 시스템입니다.

## 📋 평가 지표

### 1. 정확도 (Accuracy) - `rag_evaluation_system_accuracy.py`
- **평가 기준**: 답변의 정확성과 사실성
- **하위 지표**:
  - `doc_match` (가중치 0.3): 검색된 문서가 정답 문서와 일치하는지
  - `answer_from_retrieved` (가중치 0.5): 답변이 검색된 문서 기반인지
  - `no_hallucination` (가중치 0.2): 허위 정보가 없는지

### 2. 관련성 (Relevance) - `rag_evaluation_system_relevance.py`
- **평가 기준**: 질문과 답변의 관련성
- **하위 지표**:
  - `aligns_with_intent` (가중치 0.5): 답변이 질문 의도와 일치하는지
  - `topicality` (가중치 0.5): 답변이 주제에 적합한지

### 3. 가독성 (Readability) - `rag_evaluation_system_readability.py`
- **평가 기준**: 답변의 명확성과 간결성
- **하위 지표**:
  - `concise` (가중치 0.5): 핵심 내용이 명확하게 제시되는지
  - `no_redundancy` (가중치 0.5): 불필요한 반복이 없는지

### 4. 개인정보 보호 (PII Protection) - `rag_evaluation_system_pii.py`
- **평가 기준**: 개인정보 노출 방지
- **하위 지표**:
  - `no_unnecessary_pii` (가중치 0.6): 불필요한 개인정보 포함 여부
  - `proper_masking` (가중치 0.4): 개인정보 마스킹 처리 여부

## 🚀 설치 및 설정

### 1. 필요 라이브러리 설치
```bash
pip install requests json logging
```

### 2. Ollama 설치 및 모델 다운로드
```bash
# Ollama 설치 (Ubuntu/Debian)
curl -fsSL https://ollama.ai/install.sh | sh

# 모델 다운로드 (Gemma 3 12B)
ollama pull gemma3:12b

# Ollama 서버 시작
ollama serve
```

### 3. 데이터셋 준비
각 평가 시스템은 다음 형태의 JSON 데이터셋을 필요로 합니다:

```json
[
  {
    "question_id": "q001",
    "question": "대출 신청 방법을 알려주세요",
    "doc_ids": ["doc_001", "doc_002"],
    "required_facts": [
      {"label": "신청방법", "text": "온라인 신청", "must": true},
      {"label": "필요서류", "text": "소득증빙서류", "must": false}
    ],
    "forbidden_claims": [
      {"text": "무조건 승인"}
    ],
    "intent_summary": "대출 신청 절차 문의",
    "on_topic_keywords": ["대출", "신청", "절차"],
    "off_topic_indicators": ["투자", "보험"],
    "concise_hint": {"max_core_chars": 400},
    "redundancy_indicators": ["반복되는 내용"]
  }
]
```

## ⚙️ 설정 및 실행

### 중요: 데이터셋 경로 설정

각 파일의 `main()` 함수에서 데이터셋 경로를 수정해야 합니다:

```python
# 각 파일에서 수정 필요
test_dataset_path = "/your/actual/path/to/gold_dataset.json"
```

### 중요: RAG 모델 연동 구현

현재 `call_rag_model` 메서드는 임시 구현으로 되어 있습니다. 실제 RAG 시스템과 연동하려면 다음 메서드를 수정해야 합니다:

```python
def call_rag_model(self, question: str) -> tuple[str, List[str]]:
    """
    TODO: 실제 RAG 시스템과 연동
    - RAG 모델 API 호출
    - 답변과 검색된 문서 ID 반환
    """
    # 현재는 Ollama를 사용한 임시 구현
    # 실제 구현 시에는:
    # 1. RAG 시스템 API 호출
    # 2. 답변과 retrieved_doc_ids 반환
    pass
```

### 실행 방법

각 평가 지표별로 독립적으로 실행할 수 있습니다:

```bash
# 정확도 평가
python rag_evaluation_system_accuracy.py

# 관련성 평가
python rag_evaluation_system_relevance.py

# 가독성 평가
python rag_evaluation_system_readability.py

# 개인정보 보호 평가
python rag_evaluation_system_pii.py
```

## 📊 출력 결과

각 평가 실행 후 다음 파일들이 생성됩니다:

- `rag_evaluation_results_accuracy.json`: 정확도 평가 결과
- `rag_evaluation_results_relevance.json`: 관련성 평가 결과
- `rag_evaluation_results_readability.json`: 가독성 평가 결과
- `rag_evaluation_results_pii.json`: 개인정보 보호 평가 결과

### 결과 파일 구조
```json
[
  {
    "question_id": "q001",
    "doc_match": 1,
    "answer_from_retrieved": 1,
    "no_hallucination": 1,
    "weighted_accuracy": 1.000,
    "why": {
      "doc_match": "사전계산됨",
      "answer_from_retrieved": "필수사실포함",
      "no_hallucination": "허위정보없음"
    },
    "original_question": "대출 신청 방법을 알려주세요",
    "rag_answer": "온라인으로 신청 가능합니다...",
    "retrieved_doc_ids": ["doc_001"]
  }
]
```

## 🔧 주요 기능

### 1. 자동화된 평가 프로세스
- 질문별 자동 RAG 모델 호출
- 구조화된 평가 프롬프트 생성
- JSON 형태의 평가 결과 파싱

### 2. 오류 처리
- API 호출 실패 시 기본값 반환
- 파싱 오류 시 안전한 fallback 처리
- 상세한 로깅으로 디버깅 지원

### 3. 진행률 추적
- 실시간 진행률 표시
- 개별 질문 평가 상태 로깅
- API 호출 간격 조절 (1초)

## 📈 성능 요약

각 평가 실행 후 콘솔에 다음과 같은 요약 정보가 출력됩니다:

```
=== 정확도 평가 결과 요약 ===
총 질문 수: 100
평균 weighted_accuracy: 0.850
doc_match 통과: 85/100 (85.0%)
answer_from_retrieved 통과: 78/100 (78.0%)
no_hallucination 통과: 92/100 (92.0%)
```

## 🛠️ 커스터마이징

### 모델 변경
```python
self.model_name = "llama3:8b"  # 다른 Ollama 모델로 변경
```

### 평가 기준 가중치 조정
각 파일의 프롬프트에서 가중치를 수정할 수 있습니다:

```python
# 정확도 예시
weighted_accuracy = 0.3 * doc_match + 0.5 * answer_from_retrieved + 0.2 * no_hallucination
```

### API 호출 간격 조정
```python
time.sleep(2)  # 2초로 변경
```

## 🚨 주의사항

1. **데이터셋 경로**: 반드시 실제 데이터셋 파일 경로로 수정하세요
2. **RAG 모델 연동**: `call_rag_model` 메서드를 실제 RAG 시스템에 맞게 구현하세요
3. **Ollama 서버**: 평가 실행 전 Ollama 서버가 실행 중인지 확인하세요
4. **메모리 사용량**: 대용량 데이터셋 처리 시 충분한 메모리를 확보하세요

## 📝 TODO

- [ ] `call_rag_model` 메서드 실제 RAG 시스템 연동 구현
- [ ] 데이터셋 경로 환경변수로 설정 가능하게 변경



