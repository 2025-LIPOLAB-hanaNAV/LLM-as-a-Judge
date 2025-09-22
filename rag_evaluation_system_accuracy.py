import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# .env 파일에서 환경변수 로드
load_dotenv()
def _normalize_doc_key(value: str | None) -> str:
    """Normalize document identifiers/names for comparison."""
    if not value:
        return ""
    return value.strip().lower()

class RAGEvaluationSystem:
    def __init__(self, test_dataset_path: str, ollama_url: str = "http://localhost:11435"):
        """
        RAG 기반 챗봇 평가 시스템 초기화
        
        Args:
            test_dataset_path: 테스트 데이터셋 JSON 파일 경로
            ollama_url: Ollama 서버 URL
        """
        self.test_dataset_path = test_dataset_path
        self.ollama_url = os.getenv("OLLAMA_URL", ollama_url)
        self.model_name = os.getenv("OLLAMA_MODEL", "gemma3:12b")
        self.ragflow_api_key = os.getenv("RAGFLOW_API_KEY")
        self.ragflow_base_url = os.getenv("RAGFLOW_BASE_URL", "http://localhost:9380")
        self.ragflow_assistant_name = os.getenv("RAGFLOW_ASSISTANT_NAME")
        self.ragflow_assistant_id = os.getenv("RAGFLOW_ASSISTANT_ID")
        self._ragflow_assistant = None
        self._ragflow_session = None
        self.test_data = self.load_test_dataset()
        
    def load_test_dataset(self) -> List[Dict[str, str]]:
        """테스트 데이터셋 로드"""
        try:
            with open(self.test_dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"테스트 데이터셋 로드 완료: {len(data)}개 질문")
            return data
        except Exception as e:
            logger.error(f"데이터셋 로드 실패: {e}")
            return []
    
    def _get_ragflow_assistant(self):
        """Ensure RAGFlow assistant handle is available."""
        if self._ragflow_assistant is not None:
            return self._ragflow_assistant

        if not self.ragflow_api_key or not self.ragflow_base_url:
            logger.error("RAGFlow 환경변수(RAGFLOW_API_KEY, RAGFLOW_BASE_URL) 설정 필요")
            return None

        try:
            from ragflow_sdk import RAGFlow
        except ImportError as exc:
            logger.error(f"ragflow_sdk 불러오기 실패: {exc}")
            return None

        try:
            client = RAGFlow(api_key=self.ragflow_api_key, base_url=self.ragflow_base_url)
            assistants = client.list_chats()
            if not assistants:
                logger.error("사용 가능한 RAGFlow 챗 어시스턴트를 찾지 못했습니다.")
                return None

            selected = None
            if self.ragflow_assistant_id:
                for assistant in assistants:
                    if getattr(assistant, "id", None) == self.ragflow_assistant_id:
                        selected = assistant
                        break
                if selected is None:
                    logger.error("지정한 RAGFlow 어시스턴트 ID를 찾을 수 없습니다.")
                    return None
            elif self.ragflow_assistant_name:
                for assistant in assistants:
                    if getattr(assistant, "name", None) == self.ragflow_assistant_name:
                        selected = assistant
                        break
                if selected is None:
                    logger.error("지정한 RAGFlow 어시스턴트 이름을 찾을 수 없습니다.")
                    return None
            else:
                selected = assistants[0]

            self._ragflow_assistant = selected
            logger.info(
                "RAGFlow 어시스턴트 연결: %s",
                getattr(selected, "name", None) or getattr(selected, "id", "unknown"),
            )
            return self._ragflow_assistant
        except Exception as exc:
            logger.error(f"RAGFlow 어시스턴트 초기화 실패: {exc}")
            return None

    def _get_ragflow_session(self):
        """Ensure a reusable RAGFlow session is available."""
        if self._ragflow_session is not None and getattr(self._ragflow_session, "id", None):
            return self._ragflow_session

        assistant = self._get_ragflow_assistant()
        if assistant is None:
            return None

        try:
            session_name = f"eval-{int(time.time())}"
            session = assistant.create_session(name=session_name)
            self._ragflow_session = session
            logger.info(
                "RAGFlow 세션 생성: %s",
                getattr(session, "id", None) or session_name,
            )
            return session
        except Exception as exc:
            logger.error(f"RAGFlow 세션 생성 실패: {exc}")
            return None

    def _cleanup_ragflow_session(self):
        """Delete the temporary RAGFlow session when evaluation ends."""
        if not self._ragflow_session or not self._ragflow_assistant:
            self._ragflow_session = None
            return

        session_id = getattr(self._ragflow_session, "id", None)
        try:
            if session_id:
                self._ragflow_assistant.delete_sessions([session_id])
            else:
                self._ragflow_assistant.delete_sessions(None)
            logger.info("RAGFlow 세션 정리 완료: %s", session_id)
        except Exception as exc:
            logger.warning(f"RAGFlow 세션 삭제 실패: {exc}")
        finally:
            self._ragflow_session = None

    def call_rag_model(self, question: str) -> tuple[str, List[Dict[str, str]]]:
        """RAGFlow를 사용해 답변과 참조 문서를 조회."""
        session = self._get_ragflow_session()
        if session is None:
            return "", []

        try:
            raw_response = session._ask_chat(question=question, stream=False)
        except Exception as exc:
            logger.error(f"RAGFlow 세션 호출 실패: {exc}")
            return "", []

        try:
            data = raw_response.json()
        except Exception as exc:
            logger.error(f"RAGFlow 응답 파싱 실패: {exc}")
            logger.debug("raw response text: %s", raw_response.text if hasattr(raw_response, "text") else raw_response)
            return "", []

        answer = ""
        retrieved_sources: List[Dict[str, str]] = []

        try:
            payload = data.get("data", {}) if isinstance(data, dict) else {}
            answer = (payload.get("answer") or "").strip()

            reference = payload.get("reference") or {}
            chunks = reference.get("chunks") or []
            for chunk in chunks:
                doc_id = chunk.get("document_id") or chunk.get("document_name")
                doc_name = chunk.get("document_name") or chunk.get("document_id")
                if doc_id:
                    entry = {"id": doc_id, "name": doc_name}
                    if entry not in retrieved_sources:
                        retrieved_sources.append(entry)
        except Exception as exc:
            logger.error(f"RAGFlow 응답 해석 실패: {exc}")

        if not answer:
            logger.warning("RAGFlow 응답이 비어 있습니다. payload=%s", data)

        return answer, retrieved_sources
        
    
    def call_ollama_model(self, prompt: str) -> str:
        """Ollama 모델 호출"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Ollama 모델 호출 실패: {e}")
            return ""
    
    
    def create_evaluation_prompt(
        self,
        question_id: str,
        question: str,
        answer: str,
        retrieved_sources: List[Dict[str, str]],
        gold_data: Dict[str, Any],
    ) -> str:
        """평가 프롬프트 생성"""
        doc_ids = gold_data.get("doc_ids", [])
        normalized_doc_ids = {_normalize_doc_key(doc_id) for doc_id in doc_ids if doc_id}
        required_facts = gold_data.get("required_facts", [])
        forbidden_claims = gold_data.get("forbidden_claims", [])

        # doc_match 계산을 명시적으로 수행
        doc_match = 0
        matched_sources: List[Dict[str, str]] = []
        for src in retrieved_sources:
            src_id = _normalize_doc_key(src.get("id"))
            src_name = _normalize_doc_key(src.get("name"))
            if src_id in normalized_doc_ids or src_name in normalized_doc_ids:
                doc_match = 1
                matched_sources.append(src)

        if doc_match:
            logger.info(
                "doc_match achieved with: %s",
                ", ".join(f"{src.get('name')} ({src.get('id')})" if src.get('name') else src.get('id') for src in matched_sources),
            )

        retrieved_display = [
            f"{src.get('name')} ({src.get('id')})" if src.get('name') and src.get('id') else (src.get('name') or src.get('id'))
            for src in retrieved_sources
        ]

        prompt = f"""SYSTEM
당신은 엄격한 은행 QA 평가자입니다. 오직 하나의 JSON 객체만 반환하세요. 설명 문장(Prose)은 금지합니다.

USER
[DATA]
question_id: {question_id}
question: {question}

[GOLD]
doc_ids: {doc_ids}
required_facts: {required_facts}        // 각 항목: {{"label","text"["aliases"|"regex"],"must":true|false}}
forbidden_claims: {forbidden_claims}    // 각 항목: {{"text"["aliases"|"regex"]}}

[SYSTEM_OUTPUT]
answer: {answer}
retrieved_sources: {retrieved_display}

[PRECALCULATED]
doc_match: {doc_match} (retrieved_sources의 id 또는 name이 doc_ids와 일치하면 1, 아니면 0)

[RULES]
평가 지표는 "정확도" 하나이며, 3개의 이진(0/1) 하위 기준을 가중합으로 집계합니다.

1. doc_match: {doc_match} (이미 계산됨)
2. answer_from_retrieved (0/1, weight 0.5):
    - answer에서 사실적 진술만 고려(군더더기 무시).
    - gold.required_facts 중 **must=true**인 항목의 text(또는 aliases/regex)가 모두 포함되어야 함.
3. no_hallucination (0/1, weight 0.2):
    - answer에 forbidden_claims의 주장(또는 그 동의어/정규식)이 나타나면 0.
    - 그렇지 않으면 1.

[AGGREGATION]
weighted_accuracy = 0.3 * {doc_match} + 0.5 * answer_from_retrieved + 0.2 * no_hallucination

[CONSTRAINTS]

- 각 하위 기준에서 확신이 없으면 0을 선택합니다.
- 유효한 값만 출력하세요. 추가 텍스트 금지.
- "why" 사유는 한국어 20자 이하(간단명료).

[OUTPUT JSON SCHEMA]
{{
"question_id": "{question_id}",
"doc_match": {doc_match},
"answer_from_retrieved": 0 | 1,
"no_hallucination": 0 | 1,
"weighted_accuracy": 0.000,
"why": {{
"doc_match": "사전계산됨",
"answer_from_retrieved": "짧은 이유(≤20자)",
"no_hallucination": "짧은 이유(≤20자)"
}}
}}
"""
        return prompt
    
    def parse_evaluation_result(self, result: str) -> Dict[str, Any]:
        """평가 결과 파싱"""
        try:
            # JSON 부분만 추출
            start_idx = result.find('{')
            end_idx = result.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                logger.warning("JSON 형식을 찾을 수 없음")
                return {}
            
            json_str = result[start_idx:end_idx]
            return json.loads(json_str)
            
        except Exception as e:
            logger.error(f"평가 결과 파싱 실패: {e}")
            return {}
    
    def evaluate_single_question(self, question_data: Dict[str, str]) -> Dict[str, Any]:
        """단일 질문 평가"""
        question_id = question_data["question_id"]
        question = question_data["question"]
        
        logger.info(f"평가 시작: {question_id}")
        
        # 1. RAG 모델 호출
        answer, retrieved_sources = self.call_rag_model(question)
        logger.info(f"RAG 답변: {answer[:100]}..." if answer else "RAG 답변 없음")
        logger.info(f"검색된 문서: {retrieved_sources}")

        # 2. 평가 프롬프트 생성
        gold_data = next((item for item in self.test_data if item.get("question_id") == question_id), {})

        prompt = self.create_evaluation_prompt(question_id, question, answer, retrieved_sources, gold_data)

        
        # 3. Ollama 모델 호출
        evaluation_result = self.call_ollama_model(prompt)
        logger.info(f"평가 결과: {evaluation_result[:100]}..." if evaluation_result else "평가 결과 없음")
        
        # 4. 결과 파싱
        parsed_result = self.parse_evaluation_result(evaluation_result)
        
        # 5. weighted_accuracy 계산 (파싱 실패 시 기본값)
        if not parsed_result:
            parsed_result = {
                "question_id": question_id,
                "doc_match": 0,
                "answer_from_retrieved": 0,
                "no_hallucination": 0,
                "weighted_accuracy": 0.0,
                "why": {
                    "doc_match": "파싱실패",
                    "answer_from_retrieved": "파싱실패",
                    "no_hallucination": "파싱실패"
                }
            }
        else:
            # weighted_accuracy 재계산
            doc_match = parsed_result.get("doc_match", 0)
            answer_from_retrieved = parsed_result.get("answer_from_retrieved", 0)
            no_hallucination = parsed_result.get("no_hallucination", 0)
            parsed_result["weighted_accuracy"] = 0.3 * doc_match + 0.5 * answer_from_retrieved + 0.2 * no_hallucination
        
        # 원본 데이터 추가
        parsed_result["original_question"] = question
        parsed_result["rag_answer"] = answer
        parsed_result["retrieved_sources"] = retrieved_sources
        parsed_result["gold_doc_ids"] = gold_data.get("doc_ids", [])

        return parsed_result
    
    def evaluate_all_questions(self) -> List[Dict[str, Any]]:
        """모든 질문 평가"""
        results = []

        try:
            for i, question_data in enumerate(self.test_data):
                logger.info(f"진행률: {i+1}/{len(self.test_data)}")

                try:
                    result = self.evaluate_single_question(question_data)
                    results.append(result)

                    # API 호출 간격 조절
                    time.sleep(1)

                except Exception as e:
                    logger.error(f"질문 {question_data['question_id']} 평가 실패: {e}")
                    gold_data = next((item for item in self.test_data if item.get("question_id") == question_data["question_id"]), {})
                    # 실패한 경우에도 기본 구조 유지
                    results.append({
                        "question_id": question_data["question_id"],
                        "doc_match": 0,
                        "answer_from_retrieved": 0,
                        "no_hallucination": 0,
                        "weighted_accuracy": 0.0,
                        "why": {
                            "doc_match": "평가실패",
                            "answer_from_retrieved": "평가실패",
                            "no_hallucination": "평가실패"
                        },
                        "original_question": question_data["question"],
                        "rag_answer": "",
                        "retrieved_sources": [],
                        "gold_doc_ids": gold_data.get("doc_ids", []),
                        "error": str(e)
                    })
        finally:
            self._cleanup_ragflow_session()

        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str = "evaluation_results.json"):
        """결과 저장"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"결과 저장 완료: {output_path}")
        except Exception as e:
            logger.error(f"결과 저장 실패: {e}")
    
    def print_summary(self, results: List[Dict[str, Any]]):
        """평가 결과 요약 출력"""
        if not results:
            print("평가 결과가 없습니다.")
            return
        
        total_questions = len(results)
        avg_weighted_accuracy = sum(r.get("weighted_accuracy", 0) for r in results) / total_questions
        
        doc_match_count = sum(1 for r in results if r.get("doc_match", 0) == 1)
        answer_from_retrieved_count = sum(1 for r in results if r.get("answer_from_retrieved", 0) == 1)
        no_hallucination_count = sum(1 for r in results if r.get("no_hallucination", 0) == 1)
        
        print("\n=== 정확도 평가 결과 요약 ===")
        print(f"총 질문 수: {total_questions}")
        print(f"평균 weighted_accuracy: {avg_weighted_accuracy:.3f}")
        print(f"doc_match 통과: {doc_match_count}/{total_questions} ({doc_match_count/total_questions*100:.1f}%)")
        print(f"answer_from_retrieved 통과: {answer_from_retrieved_count}/{total_questions} ({answer_from_retrieved_count/total_questions*100:.1f}%)")
        print(f"no_hallucination 통과: {no_hallucination_count}/{total_questions} ({no_hallucination_count/total_questions*100:.1f}%)")
        
        print("\n=== 개별 결과 ===")
        for result in results:
            print(f"{result['question_id']}: weighted_accuracy={result.get('weighted_accuracy', 0):.3f} "
                  f"(doc_match={result.get('doc_match', 0)}, "
                  f"answer_from_retrieved={result.get('answer_from_retrieved', 0)}, "
                  f"no_hallucination={result.get('no_hallucination', 0)})")
            sources = result.get("retrieved_sources") or []
            if sources:
                sources_display = ", ".join(
                    f"{src.get('name')} ({src.get('id')})" if src.get('name') and src.get('id') else (src.get('name') or src.get('id'))
                    for src in sources
                )
                print(f"  ↳ retrieved: {sources_display}")
            gold_docs = result.get("gold_doc_ids") or []
            if gold_docs:
                print(f"  ↳ gold: {', '.join(gold_docs)}")

def main():
    """메인 실행 함수"""
    # 설정
    test_dataset_path = "./gold_dataset.json"  # TODO: 위치 경로 변경하기
    ollama_url = "http://localhost:11435"
    
    # 평가 시스템 초기화
    evaluator = RAGEvaluationSystem(test_dataset_path, ollama_url)
    
    # 모든 질문 평가
    print("RAG 기반 챗봇 평가 시작...")
    results = evaluator.evaluate_all_questions()
    
    # 결과 저장
    evaluator.save_results(results, "rag_evaluation_results_accuracy.json")
    
    # 요약 출력
    evaluator.print_summary(results)
    
    print("\n평가 완료!")

if __name__ == "__main__":
    main()
