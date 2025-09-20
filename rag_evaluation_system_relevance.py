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
    
    def call_rag_model(self, question: str) -> str:
        """
        TODO: RAG 모델 호출 
        TODO: 실제 구현 시 RAG 모델을 호출하여 답변을 받아옴
        """

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
            return result.get("response", "").strip()
        except Exception as e:
            logger.error(f"RAG 모델 호출 실패: {e}")
            return ""
        """
    
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
    
    
    def create_evaluation_prompt(self, question_id: str, question: str, answer: str) -> str:
        """평가 프롬프트 생성"""
        # 골드 데이터셋에서 해당 질문의 정보 찾기
        gold_data = None
        for item in self.test_data:
            if item["question_id"] == question_id:
                gold_data = item
                break
        
        # 골드 데이터에서 정보 추출
        intent_summary = gold_data.get("intent_summary", "")
        on_topic_keywords = gold_data.get("on_topic_keywords", [])
        off_topic_indicators = gold_data.get("off_topic_indicators", [])
        
        prompt = f"""SYSTEM
당신은 엄격한 은행 QA 평가자입니다. 오직 하나의 JSON 객체만 반환하세요. 설명 문장(Prose)은 금지합니다.

USER
[DATA]
question_id: {question_id}
question: {question}

[GOLD]
intent_summary: {intent_summary}                 
on_topic_keywords: {on_topic_keywords}
off_topic_indicators: {off_topic_indicators}

[SYSTEM_OUTPUT]
answer: {answer}

[RULES]
평가 지표는 "관련성" 하나이며, 2개의 이진(0/1) 하위 기준을 가중합으로 집계합니다.
가중치: aligns_with_intent=0.5, topicality=0.5

1. aligns_with_intent (0/1, weight 0.5):
    - 1로 판단하는 조건(모두 충족):
    a) answer의 주된 내용(체감 60% 이상)이 intent_summary와 의미상 부합한다.
    b) intent_summary의 핵심 행위/결론과 모순되는 주장을 핵심으로 내세우지 않는다.
    - 위 조건을 충족하지 못하면 0.
2. topicality (0/1, weight 0.5):
    - 1로 판단하는 조건(모두 충족):
    a) on_topic_keywords 중 하나 이상(동의어/요약 허용)이 답변에서 명시적으로 다뤄진다.
    b) off_topic_indicators에 해당하는 주제가 답변의 핵심을 이루지 않는다
    (부수적 언급은 허용하되, 핵심 비중이 되면 0).
    - 위 조건을 충족하지 못하면 0.

[AGGREGATION]
weighted_relevance = 0.5 * aligns_with_intent + 0.5 * topicality

[CONSTRAINTS]

- 확신이 없으면 0을 선택합니다.
- 유효한 JSON만 출력하세요. 추가 텍스트 금지.
- "why" 사유는 한국어 20자 이하(간단명료).

[OUTPUT JSON SCHEMA]
{{
"question_id": "{{question_id}}",
"aligns_with_intent": 0 | 1,
"topicality": 0 | 1,
"weighted_relevance": 0.000,
"why": {{
"aligns_with_intent": "짧은 이유(≤20자)",
"topicality": "짧은 이유(≤20자)"
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
        answer = self.call_rag_model(question)
        logger.info(f"RAG 답변: {answer[:100]}..." if answer else "RAG 답변 없음")
        
        # 2. 평가 프롬프트 생성
        prompt = self.create_evaluation_prompt(question_id, question, answer)
        
        # 3. Ollama 모델 호출
        evaluation_result = self.call_ollama_model(prompt)
        logger.info(f"평가 결과: {evaluation_result[:100]}..." if evaluation_result else "평가 결과 없음")
        
        # 4. 결과 파싱
        parsed_result = self.parse_evaluation_result(evaluation_result)
        
        # 5. weighted_relevance 계산 (파싱 실패 시 기본값)
        if not parsed_result:
            parsed_result = {
                "question_id": question_id,
                "aligns_with_intent": 0,
                "topicality": 0,
                "weighted_relevance": 0.0,
                "why": {
                    "aligns_with_intent": "파싱실패",
                    "topicality": "파싱실패"
                }
            }
        else:
            # weighted_relevance 재계산
            aligns_with_intent = parsed_result.get("aligns_with_intent", 0)
            topicality = parsed_result.get("topicality", 0)
            parsed_result["weighted_relevance"] = 0.5 * aligns_with_intent + 0.5 * topicality
        
        # 원본 데이터 추가
        parsed_result["original_question"] = question
        parsed_result["rag_answer"] = answer
        
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
                    "aligns_with_intent": 0,
                    "topicality": 0,
                    "weighted_relevance": 0.0,
                    "why": {
                        "aligns_with_intent": "평가실패",
                        "topicality": "평가실패"
                    },
                    "original_question": question_data["question"],
                    "rag_answer": "",
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
        avg_weighted_relevance = sum(r.get("weighted_relevance", 0) for r in results) / total_questions
        
        aligns_with_intent_count = sum(1 for r in results if r.get("aligns_with_intent", 0) == 1)
        topicality_count = sum(1 for r in results if r.get("topicality", 0) == 1)
        
        print("\n=== 평가 결과 요약 ===")
        print(f"총 질문 수: {total_questions}")
        print(f"평균 weighted_relevance: {avg_weighted_relevance:.3f}")
        print(f"aligns_with_intent 통과: {aligns_with_intent_count}/{total_questions} ({aligns_with_intent_count/total_questions*100:.1f}%)")
        print(f"topicality 통과: {topicality_count}/{total_questions} ({topicality_count/total_questions*100:.1f}%)")
        
        print("\n=== 개별 결과 ===")
        for result in results:
            print(f"{result['question_id']}: weighted_relevance={result.get('weighted_relevance', 0):.3f} "
                  f"(intent={result.get('aligns_with_intent', 0)}, "
                  f"topic={result.get('topicality', 0)})")

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
    evaluator.save_results(results, "rag_evaluation_results_relevance.json")
    
    # 요약 출력
    evaluator.print_summary(results)
    
    print("\n평가 완료!")

if __name__ == "__main__":
    main()
