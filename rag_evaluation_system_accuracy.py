import json
import requests
import time
from typing import Dict, List, Any
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEvaluationSystem:
    def __init__(self, test_dataset_path: str, ollama_url: str = "http://localhost:11435"):
        """
        RAG 기반 챗봇 평가 시스템 초기화
        
        Args:
            test_dataset_path: 테스트 데이터셋 JSON 파일 경로
            ollama_url: Ollama 서버 URL
        """
        self.test_dataset_path = test_dataset_path
        self.ollama_url = ollama_url
        self.model_name = "gemma3:12b"
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
    
    def call_rag_model(self, question: str) -> tuple[str, List[str]]:
        """
        TODO: RAG 모델 호출
        TODO: 실제 구현 시 RAG 모델을 호출하여 답변과 retrieved_doc_ids를 받아옴
        """
        # 임시로 Ollama를 사용하여 질문에 대한 답변 생성
        rag_prompt = f"""당신은 은행 고객상담원입니다. 다음 질문에 대해 간단하고 정확한 답변을 제공해주세요.
                질문: {question}
                답변:"""
        try:
            # RAG 답변 생성을 위한 별도 옵션 설정
            url = f"{self.ollama_url}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": rag_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "num_predict": 200  # 짧은 답변으로 제한
                }
            }
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            result = response.json()
            answer = result.get("response", "").strip()
            
            # TODO: 실제 RAG 시스템에서는 검색된 문서 ID들을 반환해야 함
            # 현재는 임시로 빈 리스트 반환
            retrieved_doc_ids = []
            
            return answer, retrieved_doc_ids
        except Exception as e:
            logger.error(f"RAG 모델 호출 실패: {e}")
            return "", []
        
    
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
    
    
    def create_evaluation_prompt(self, question_id: str, question: str, answer: str, retrieved_doc_ids: List[str]) -> str:
        """평가 프롬프트 생성"""
        # 골드 데이터셋에서 해당 질문의 정보 찾기
        gold_data = None
        for item in self.test_data:
            if item["question_id"] == question_id:
                gold_data = item
                break
        

        # 골드 데이터에서 정확도 평가 관련 정보 추출
        doc_ids = gold_data.get("doc_ids", [])
        required_facts = gold_data.get("required_facts", [])
        forbidden_claims = gold_data.get("forbidden_claims", [])
        
        # doc_match 계산을 명시적으로 수행
        doc_match = 1 if any(doc in doc_ids for doc in retrieved_doc_ids) else 0
        
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
retrieved_doc_ids: {retrieved_doc_ids}

[PRECALCULATED]
doc_match: {doc_match} (retrieved_doc_ids와 doc_ids의 교집합이 있으면 1, 없으면 0)

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
        answer, retrieved_doc_ids = self.call_rag_model(question)
        logger.info(f"RAG 답변: {answer[:100]}..." if answer else "RAG 답변 없음")
        logger.info(f"검색된 문서 ID: {retrieved_doc_ids}")
        
        # 2. 평가 프롬프트 생성
        prompt = self.create_evaluation_prompt(question_id, question, answer, retrieved_doc_ids)

        
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
        parsed_result["retrieved_doc_ids"] = retrieved_doc_ids
        
        return parsed_result
    
    def evaluate_all_questions(self) -> List[Dict[str, Any]]:
        """모든 질문 평가"""
        results = []
        
        for i, question_data in enumerate(self.test_data):
            logger.info(f"진행률: {i+1}/{len(self.test_data)}")
            
            try:
                result = self.evaluate_single_question(question_data)
                results.append(result)
                
                # API 호출 간격 조절
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"질문 {question_data['question_id']} 평가 실패: {e}")
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
                    "retrieved_doc_ids": [],
                    "error": str(e)
                })
        
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

def main():
    """메인 실행 함수"""
    # 설정
    test_dataset_path = "/home/dev7/project/gold_dataset.json"  # TODO: 위치 경로 변경하기
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
