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
        
        # 골드 데이터에서 가독성 관련 정보 추출
        concise_hint = gold_data.get("concise_hint", {})
        redundancy_indicators = gold_data.get("redundancy_indicators", [])
        
        prompt = f"""SYSTEM

당신은 엄격한 은행 QA 평가자입니다. 오직 하나의 JSON 객체만 반환하세요. 설명 문장(Prose)은 금지합니다.

USER
[DATA]
question_id: {{question_id}}
question: {{question}}

[GOLD]   // 가독성 판정용 힌트(옵션)
concise_hint: {{concise_hint}}                   
redundancy_indicators: {{redundancy_indicators}} 

[SYSTEM_OUTPUT]
answer: {{answer}}

[RULES]
평가 지표는 "가독성" 하나이며, 2개의 이진(0/1) 하위 기준을 가중합으로 집계합니다.

1. concise (0/1, weight 0.5):
    - 1로 판단하는 조건(모두 충족):
    a) 답변 초반(첫 2~3문장 또는 처음 {concise_hint.get('max_core_chars', 400)}자) 안에 핵심 결론이 명확히 제시됨.
    b) 불필요한 서론/장황한 배경설명이 핵심보다 길지 않음.
    - 위 조건을 충족하지 못하면 0.
2. no_redundancy (0/1, weight 0.5):
    - 동일/유사한 내용이나 수치·정책 문구를 실질적 추가 정보 없이 반복하지 않으면 1, 반복이 눈에 띄면 0.
    - redundancy_indicators가 주어졌다면 이를 특히 주의해 판단.

[AGGREGATION]
weighted_readability = 0.5 * concise + 0.5 * no_redundancy

[CONSTRAINTS]

- 각 기준에서 확신이 없으면 0을 선택합니다.
- 유효한 JSON만 출력하세요. 추가 텍스트 금지.
- "why" 사유는 한국어 20자 이하(간단명료).

[OUTPUT JSON SCHEMA]
{{
"question_id": "{question_id}",
"concise": 0 | 1,
"no_redundancy": 0 | 1,
"weighted_readability": 0.000,
"why": {{
"concise": "짧은 이유(≤20자)",
"no_redundancy": "짧은 이유(≤20자)"
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
        
        # 5. weighted_readability 계산 (파싱 실패 시 기본값)
        if not parsed_result:
            parsed_result = {
                "question_id": question_id,
                "concise": 0,
                "no_redundancy": 0,
                "weighted_readability": 0.0,
                "why": {
                    "concise": "파싱실패",
                    "no_redundancy": "파싱실패"
                }
            }
        else:
            # weighted_readability 재계산
            concise = parsed_result.get("concise", 0)
            no_redundancy = parsed_result.get("no_redundancy", 0)
            parsed_result["weighted_readability"] = 0.5 * concise + 0.5 * no_redundancy
        
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
                    "concise": 0,
                    "no_redundancy": 0,
                    "weighted_readability": 0.0,
                    "why": {
                        "concise": "평가실패",
                        "no_redundancy": "평가실패"
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
        avg_weighted_readability = sum(r.get("weighted_readability", 0) for r in results) / total_questions
        
        concise_count = sum(1 for r in results if r.get("concise", 0) == 1)
        no_redundancy_count = sum(1 for r in results if r.get("no_redundancy", 0) == 1)
        
        print("\n=== 평가 결과 요약 ===")
        print(f"총 질문 수: {total_questions}")
        print(f"평균 weighted_readability: {avg_weighted_readability:.3f}")
        print(f"concise 통과: {concise_count}/{total_questions} ({concise_count/total_questions*100:.1f}%)")
        print(f"no_redundancy 통과: {no_redundancy_count}/{total_questions} ({no_redundancy_count/total_questions*100:.1f}%)")
        
        print("\n=== 개별 결과 ===")
        for result in results:
            print(f"{result['question_id']}: weighted_readability={result.get('weighted_readability', 0):.3f} "
                  f"(concise={result.get('concise', 0)}, "
                  f"no_redundancy={result.get('no_redundancy', 0)})")

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
    evaluator.save_results(results, "rag_evaluation_results_readability.json")
    
    # 요약 출력
    evaluator.print_summary(results)
    
    print("\n평가 완료!")

if __name__ == "__main__":
    main()
