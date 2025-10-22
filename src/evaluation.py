"""
RAG Evaluation Framework for Fitness Exercise Recommendations
Implements comprehensive evaluation metrics and LLM-as-Judge assessment
"""

import json
import statistics
from typing import List, Dict, Any, Tuple, Optional
from .rag_agent import FitnessRAG
import requests
from datetime import datetime
import os


class RAGEvaluator:
    """Comprehensive evaluation framework for RAG exercise recommendations"""
    
    def __init__(self, rag_agent: FitnessRAG, ollama_url: str = "http://localhost:11434", show_explanations: bool = True):
        self.rag = rag_agent
        self.ollama_url = ollama_url
        self.show_explanations = show_explanations
        self.test_queries_file = "data/test_queries.json"
        
        # Load test queries from file if it exists, otherwise use default queries
        self.test_queries = self._load_test_queries()
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
    
    # =========================================================================
    # TEST QUERY LOADING AND PREPARATION
    # =========================================================================
        
    def _load_test_queries(self) -> List[Dict]:
        """Load test queries from generated file or use defaults"""
        try:
            from pathlib import Path
            if Path(self.test_queries_file).exists():
                import json
                with open(self.test_queries_file, 'r') as f:
                    data = json.load(f)
                
                # Convert to expected format
                test_queries = []
                for item in data['queries'][:15]:  # Use first 15 for detailed evaluation
                    query = item['query']
                    test_queries.append({
                        "query": query,
                        "relevant_keywords": self._extract_keywords(query),
                        "expected_body_parts": self._extract_body_parts(query),
                        "expected_equipment": self._extract_equipment(query)
                    })
                
                print(f"✓ Loaded {len(test_queries)} queries from {self.test_queries_file}")
                return test_queries
            else:
                print(f"⚠️ No test queries file found at {self.test_queries_file}")
                print("   Run 'python scripts/generate_test_queries.py' first")
                return self._get_default_queries()
                
        except Exception as e:
            print(f"⚠️ Error loading test queries: {e}")
            return self._get_default_queries()
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract relevant keywords from query"""
        keywords = []
        query_lower = query.lower()
        
        # Body part keywords
        body_parts = ['chest', 'back', 'shoulder', 'arm', 'leg', 'abs', 'core', 'bicep', 'tricep']
        keywords.extend([bp for bp in body_parts if bp in query_lower])
        
        # Equipment keywords  
        equipment = ['dumbbell', 'barbell', 'kettlebell', 'cable', 'bodyweight', 'machine']
        keywords.extend([eq for eq in equipment if eq in query_lower])
        
        # Exercise types
        exercises = ['press', 'curl', 'row', 'squat', 'lunge', 'fly', 'raise']
        keywords.extend([ex for ex in exercises if ex in query_lower])
        
        return list(set(keywords)) if keywords else ['exercise', 'workout']
    
    def _extract_body_parts(self, query: str) -> List[str]:
        """Extract expected body parts from query"""
        query_lower = query.lower()
        body_parts = []
        
        mapping = {
            'chest': ['chest'], 'back': ['back'], 'shoulder': ['shoulders'],
            'arm': ['arms', 'upper arms'], 'leg': ['legs', 'upper legs', 'lower legs'],
            'abs': ['abs', 'core'], 'core': ['abs', 'core', 'waist'],
            'bicep': ['upper arms'], 'tricep': ['upper arms']
        }
        
        for keyword, parts in mapping.items():
            if keyword in query_lower:
                body_parts.extend(parts)
        
        return list(set(body_parts)) if body_parts else ['chest', 'back', 'legs']
    
    def _extract_equipment(self, query: str) -> List[str]:
        """Extract expected equipment from query"""
        query_lower = query.lower()
        equipment = []
        
        mapping = {
            'dumbbell': ['dumbbell'], 'barbell': ['barbell'], 'kettlebell': ['kettlebell'],
            'cable': ['cable'], 'bodyweight': ['body weight'], 'machine': ['machine', 'lever']
        }
        
        for keyword, equip in mapping.items():
            if keyword in query_lower:
                equipment.extend(equip)
        
        return list(set(equipment)) if equipment else ['dumbbell', 'barbell', 'body weight']
    
    def _get_default_queries(self) -> List[Dict]:
        """Default test queries if file not found"""
        return [
            # Body part focused queries
            {
                "query": "chest exercises with dumbbells",
                "relevant_keywords": ["chest", "dumbbell", "pectoral"],
                "expected_body_parts": ["chest"],
                "expected_equipment": ["dumbbell"]
            },
            {
                "query": "back exercises for strength training", 
                "relevant_keywords": ["back", "strength", "pull", "row", "lat"],
                "expected_body_parts": ["back"],
                "expected_equipment": ["barbell", "dumbbell", "cable"]
            },
            {
                "query": "shoulder workout with cables",
                "relevant_keywords": ["shoulder", "deltoid", "cable"],
                "expected_body_parts": ["shoulders"],
                "expected_equipment": ["cable"]
            },
            {
                "query": "leg exercises for beginners",
                "relevant_keywords": ["leg", "quad", "hamstring", "squat", "lunge"],
                "expected_body_parts": ["legs", "upper legs", "lower legs"],
                "expected_equipment": ["body weight", "dumbbell"]
            },
            {
                "query": "abs and core strengthening",
                "relevant_keywords": ["abs", "core", "plank", "crunch"],
                "expected_body_parts": ["abs", "core", "waist"],
                "expected_equipment": ["body weight"]
            }
        ]
    
    # =========================================================================
    # RETRIEVAL METRICS CALCULATION
    # =========================================================================
    
    def calculate_precision_at_k(self, retrieved_docs: List[Dict], relevant_keywords: List[str], k: int = 5) -> float:
        """
        Calculate precision@k - how many of the top-k retrieved docs contain relevant keywords
        
        Args:
            retrieved_docs: List of retrieved documents with metadata and content
            relevant_keywords: List of keywords that should appear in relevant docs
            k: Number of top documents to evaluate
            
        Returns:
            float: Precision@k score (0.0 to 1.0)
        """
        if not retrieved_docs or k <= 0:
            return 0.0
        
        top_k_docs = retrieved_docs[:k]
        relevant_count = 0
        
        for doc in top_k_docs:
            # Check if any relevant keyword appears in doc content or metadata
            content_text = (doc['content'] + ' ' + 
                          doc['metadata'].get('name', '') + ' ' +
                          doc['metadata'].get('target_muscle', '') + ' ' +
                          doc['metadata'].get('body_part', '')).lower()
            
            if any(keyword.lower() in content_text for keyword in relevant_keywords):
                relevant_count += 1
        
        return relevant_count / len(top_k_docs)
    
    def calculate_hit_rate(self, retrieved_docs: List[Dict], relevant_keywords: List[str]) -> float:
        """
        Calculate hit rate - did we find at least one relevant document?
        
        Args:
            retrieved_docs: List of retrieved documents
            relevant_keywords: List of keywords that should appear in relevant docs
            
        Returns:
            float: Hit rate (0.0 or 1.0)
        """
        if not retrieved_docs:
            return 0.0
        
        for doc in retrieved_docs:
            content_text = (doc['content'] + ' ' + 
                          doc['metadata'].get('name', '') + ' ' +
                          doc['metadata'].get('target_muscle', '') + ' ' +
                          doc['metadata'].get('body_part', '')).lower()
            
            if any(keyword.lower() in content_text for keyword in relevant_keywords):
                return 1.0
        
        return 0.0
    
    def check_metadata_accuracy(self, retrieved_docs: List[Dict], test_case: Dict) -> Dict[str, float]:
        """
        Check if retrieved documents match expected metadata filters
        
        Args:
            retrieved_docs: List of retrieved documents
            test_case: Test case with expected body parts and equipment
            
        Returns:
            Dict with accuracy scores for different metadata aspects
        """
        if not retrieved_docs:
            return {"body_part_accuracy": 0.0, "equipment_accuracy": 0.0}
        
        # Check body part accuracy
        expected_body_parts = test_case.get("expected_body_parts", [])
        body_part_matches = 0
        
        for doc in retrieved_docs[:5]:  # Check top 5
            doc_body_part = doc['metadata'].get('body_part', '').lower()
            if any(expected.lower() in doc_body_part or doc_body_part in expected.lower() 
                   for expected in expected_body_parts):
                body_part_matches += 1
        
        body_part_accuracy = body_part_matches / min(5, len(retrieved_docs))
        
        # Check equipment accuracy  
        expected_equipment = test_case.get("expected_equipment", [])
        equipment_matches = 0
        
        for doc in retrieved_docs[:5]:  # Check top 5
            doc_equipment = doc['metadata'].get('equipment', '').lower()
            if any(expected.lower() in doc_equipment or doc_equipment in expected.lower()
                   for expected in expected_equipment):
                equipment_matches += 1
        
        equipment_accuracy = equipment_matches / min(5, len(retrieved_docs))
        
        return {
            "body_part_accuracy": body_part_accuracy,
            "equipment_accuracy": equipment_accuracy
        }
    
    # =========================================================================
    # LLM-AS-JUDGE EVALUATION
    # =========================================================================
    
    def evaluate_response_quality(self, query: str, response: str) -> Dict[str, Any]:
        """
        Use Ollama (Llama 3.2) as judge to evaluate response quality with detailed feedback
        
        Args:
            query: Original user query
            response: RAG system response
            
        Returns:
            Dict with scores (1-10 scale) and detailed explanations
        """
        eval_prompt = f"""You are an expert evaluator for an exercise recommendation system. Evaluate how well this answer responds to the user's query.

QUERY: {query}

GENERATED ANSWER:
{response}

Evaluate the following aspects on a scale of 1-10:

1. RETRIEVAL RELEVANCE (1-10): Do the exercises mentioned in the answer match what the query asked for?
   - Check if muscle groups, exercise types, equipment, difficulty levels align with the query
   - Are the recommended exercises actually relevant to the user's request?
   - Does the answer address the right type of exercises?

2. ANSWER ACCURACY (1-10): Is the generated answer factually correct about the exercises mentioned?
   - Are the muscle groups, equipment requirements, rep ranges, difficulty levels accurate?
   - No obvious errors or contradictions in exercise descriptions?
   - Does the information seem reliable and consistent?

3. ANSWER COMPLETENESS (1-10): Does the answer fully address the query?
   - Does it answer all parts of the question?
   - Provides sufficient detail and context for decision-making?
   - Addresses the user's intent and fitness goals appropriately?

4. CITATION QUALITY (1-10): Are the exercise recommendations well-presented?
   - Are exercise names clearly stated?
   - Does it include helpful metadata (muscle groups, equipment, difficulty, reps/sets)?
   - Are the recommendations well-organized and easy to understand?
   - Good balance of breadth vs. depth in recommendations?

Respond in this EXACT format:
RETRIEVAL_RELEVANCE: [score]/10
REASON: [one sentence explaining the score]

ANSWER_ACCURACY: [score]/10  
REASON: [one sentence explaining the score]

ANSWER_COMPLETENESS: [score]/10
REASON: [one sentence explaining the score]

CITATION_QUALITY: [score]/10
REASON: [one sentence explaining the score]

OVERALL: [average score]/10
"""
        
        try:
            # Check if Ollama is running first
            test_response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if test_response.status_code != 200:
                raise ConnectionError(f"Ollama not accessible at {self.ollama_url}")
            
            # Call Ollama API
            response_data = {
                "model": "llama3.2",
                "prompt": eval_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1  # Low temperature for consistent evaluation
                }
            }
            
            llm_response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=response_data,
                timeout=45
            )
            
            if llm_response.status_code != 200:
                raise ConnectionError(f"Ollama API error: {llm_response.status_code}")
            
            evaluation_text = llm_response.json()["response"]
            
            # Parse the evaluation with robust parsing
            scores = self._parse_evaluation(evaluation_text)
            
            # Calculate overall if not provided
            if scores.get('overall') is None:
                valid_scores = [v for v in [
                    scores.get('retrieval_relevance'),
                    scores.get('answer_accuracy'), 
                    scores.get('answer_completeness'),
                    scores.get('citation_quality')
                ] if v is not None]
                
                if valid_scores:
                    scores['overall'] = sum(valid_scores) / len(valid_scores)
            
            # Extract reasons from evaluation text
            reasons = self._extract_reasons(evaluation_text)
            
            # Combine scores with reasons
            result = {
                'retrieval_relevance_score': scores.get('retrieval_relevance', 0),
                'answer_accuracy_score': scores.get('answer_accuracy', 0),
                'answer_completeness_score': scores.get('answer_completeness', 0),
                'citation_quality_score': scores.get('citation_quality', 0),
                'overall_score': scores.get('overall', 0),
                'retrieval_relevance_reason': reasons.get('retrieval_relevance', 'No explanation provided'),
                'answer_accuracy_reason': reasons.get('answer_accuracy', 'No explanation provided'),
                'answer_completeness_reason': reasons.get('answer_completeness', 'No explanation provided'),
                'citation_quality_reason': reasons.get('citation_quality', 'No explanation provided'),
                'overall_reason': reasons.get('overall', 'No explanation provided'),
                'full_evaluation': evaluation_text
            }
            
            return result
            
        except ConnectionError as e:
            raise e
        except Exception as e:
            print(f"⚠️ LLM evaluation parsing failed: {e}")
            return {
                'retrieval_relevance_score': 0,
                'answer_accuracy_score': 0,
                'answer_completeness_score': 0,
                'citation_quality_score': 0,
                'overall_score': 0,
                'retrieval_relevance_reason': 'Evaluation failed',
                'answer_accuracy_reason': 'Evaluation failed',
                'answer_completeness_reason': 'Evaluation failed',
                'citation_quality_reason': 'Evaluation failed',
                'overall_reason': 'Evaluation failed',
                'full_evaluation': str(e)
            }
    
    def _parse_evaluation(self, evaluation: str) -> Dict[str, float]:
        """Robust score parsing that handles typos"""
        import re
        scores = {}
        
        patterns = {
            'retrieval_relevance': r'(?:RETRIEVAL_RELEVANCE|RETIEVAL_RELEVANCE):\s*(\d+(?:\.\d+)?)',
            'answer_accuracy': r'ANSWER_ACCURACY:\s*(\d+(?:\.\d+)?)',
            'answer_completeness': r'ANSWER_COMPLETENESS:\s*(\d+(?:\.\d+)?)',
            'citation_quality': r'CITATION_QUALITY:\s*(\d+(?:\.\d+)?)',
            'overall': r'OVERALL:\s*(\d+(?:\.\d+)?)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, evaluation, re.IGNORECASE)
            if match:
                scores[key] = float(match.group(1))
            else:
                scores[key] = None
        
        return scores
    
    def _extract_reasons(self, evaluation: str) -> Dict[str, str]:
        """Extract reasoning text from evaluation"""
        import re
        reasons = {}
        
        # Extract reasons that follow the REASON: pattern
        reason_patterns = {
            'retrieval_relevance': r'RETRIEVAL_RELEVANCE:.*?\nREASON:\s*([^\n]+)',
            'answer_accuracy': r'ANSWER_ACCURACY:.*?\nREASON:\s*([^\n]+)',
            'answer_completeness': r'ANSWER_COMPLETENESS:.*?\nREASON:\s*([^\n]+)',
            'citation_quality': r'CITATION_QUALITY:.*?\nREASON:\s*([^\n]+)',
            'overall': r'OVERALL:.*?(?:\nREASON:\s*([^\n]+)|$)'
        }
        
        for key, pattern in reason_patterns.items():
            match = re.search(pattern, evaluation, re.IGNORECASE | re.DOTALL)
            if match and match.group(1):
                reasons[key] = match.group(1).strip()
            else:
                reasons[key] = f"Assessment for {key.replace('_', ' ')}"
        
        return reasons
    
    # =========================================================================
    # EVALUATION DISPLAY AND FORMATTING
    # =========================================================================
    
    def _display_evaluation_result(self, evaluation: Dict[str, Any]):
        """Display beautiful evaluation result with progress bars like the judge.md example"""
        if not evaluation:
            return
            
        # Extract scores
        relevance = evaluation.get('retrieval_relevance_score', 0)
        accuracy = evaluation.get('answer_accuracy_score', 0)
        completeness = evaluation.get('answer_completeness_score', 0)
        citation = evaluation.get('citation_quality_score', 0)
        overall = evaluation.get('overall_score', 0)
        
        if overall == 0:
            overall = statistics.mean([relevance, accuracy, completeness, citation])
        
        print("\n" + "="*80)
        print("EVALUATION (LLM-as-a-judge with Llama 3.2)")
        print("="*80)
        print()
        print("Scores:")
        print(f"  Retrieval Relevance:   {self._create_progress_bar(relevance)} {relevance:.1f}/10")
        print(f"  Answer Accuracy:       {self._create_progress_bar(accuracy)} {accuracy:.1f}/10")
        print(f"  Answer Completeness:   {self._create_progress_bar(completeness)} {completeness:.1f}/10")
        print(f"  Citation Quality:      {self._create_progress_bar(citation)} {citation:.1f}/10")
        print(f"  ──────────────────────────────────────────────────")
        print(f"  Overall Score:         {self._create_progress_bar(overall)} {overall:.1f}/10")
        print()
        
        if self.show_explanations:
            print("Detailed Feedback:")
            print(f"RETRIEVAL_RELEVANCE: {relevance:.0f}/10")
            print(f"REASON: {evaluation.get('retrieval_relevance_reason', 'No explanation provided')}")
            print()
            print(f"ANSWER_ACCURACY: {accuracy:.0f}/10")
            print(f"REASON: {evaluation.get('answer_accuracy_reason', 'No explanation provided')}")
            print()
            print(f"ANSWER_COMPLETENESS: {completeness:.0f}/10")
            print(f"REASON: {evaluation.get('answer_completeness_reason', 'No explanation provided')}")
            print()
            print(f"CITATION_QUALITY: {citation:.0f}/10")
            print(f"REASON: {evaluation.get('citation_quality_reason', 'No explanation provided')}")
            print()
            print(f"OVERALL: {overall:.1f}/10")
            print(f"REASON: {evaluation.get('overall_reason', 'Comprehensive assessment of all evaluation criteria.')}")
        
        print("="*80)
    
    def _create_progress_bar(self, score: float, max_score: float = 10.0) -> str:
        """Create a visual progress bar for scores"""
        filled_blocks = int((score / max_score) * 10)
        empty_blocks = 10 - filled_blocks
        return "█" * filled_blocks + "░" * empty_blocks
    
    # =========================================================================
    # MAIN EVALUATION ORCHESTRATION
    # =========================================================================
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run comprehensive evaluation on all test queries
        
        Returns:
            Dict with evaluation results and summary statistics
        """
        print("🔍 Starting RAG Evaluation...")
        print(f"Testing {len(self.test_queries)} diverse queries")
        print("="*60)
        
        results = []
        
        for i, test_case in enumerate(self.test_queries, 1):
            query = test_case["query"]
            print(f"\n[{i}/{len(self.test_queries)}] Testing: '{query}'")
            
            # Execute RAG query (silent mode - no retrieval lists)
            rag_result = self.rag.query(query, verbose=False, silent=True)
            retrieved_docs = rag_result['raw_documents']
            response = rag_result['recommendations']
            
            # Show the actual response being evaluated (instead of just the list)
            print("📝 RAG Response:")
            print("-" * 40)
            print(response)
            print("-" * 40)
            
            # Calculate retrieval metrics
            precision_5 = self.calculate_precision_at_k(retrieved_docs, test_case["relevant_keywords"], k=5)
            precision_10 = self.calculate_precision_at_k(retrieved_docs, test_case["relevant_keywords"], k=10)
            hit_rate = self.calculate_hit_rate(retrieved_docs, test_case["relevant_keywords"])
            
            # Calculate metadata accuracy
            metadata_accuracy = self.check_metadata_accuracy(retrieved_docs, test_case)
            
            # Evaluate response quality with LLM
            print("  🤖 Getting LLM quality scores...")
            llm_evaluation = None
            try:
                llm_evaluation = self.evaluate_response_quality(query, response)
                relevance = llm_evaluation.get('retrieval_relevance_score', 0)
                accuracy = llm_evaluation.get('answer_accuracy_score', 0)
                completeness = llm_evaluation.get('answer_completeness_score', 0)
                citation = llm_evaluation.get('citation_quality_score', 0)
                overall = llm_evaluation.get('overall_score', 0)
                
                if overall == 0:
                    overall = statistics.mean([relevance, accuracy, completeness, citation])
                
                # Display beautiful formatted evaluation
                self._display_evaluation_result(llm_evaluation)
                
            except ConnectionError as e:
                print(f"\n{str(e)}")
                print("\n⚠️ Skipping LLM evaluation - running retrieval metrics only...")
                llm_evaluation = None
            except Exception as e:
                print(f"⚠️ LLM evaluation failed: {e}")
                llm_evaluation = None
            
            if not llm_evaluation:
                print(f"  ⚠️ LLM evaluation not available (Ollama not running)")
            
            # Store results
            result = {
                "query": query,
                "response": response,  # Store the actual RAG response
                "retrieved_count": len(retrieved_docs),
                "precision_at_5": precision_5,
                "precision_at_10": precision_10,
                "hit_rate": hit_rate,
                "body_part_accuracy": metadata_accuracy["body_part_accuracy"],
                "equipment_accuracy": metadata_accuracy["equipment_accuracy"],
                "llm_evaluation": llm_evaluation,  # Store full LLM evaluation
                "timestamp": datetime.now().isoformat()
            }
            
            results.append(result)
        
        # Calculate summary statistics
        summary = self._calculate_summary_stats(results)
        
        # Save detailed results to logs
        log_filename = self.save_detailed_results(results, summary)
        
        # Print final report
        self._print_evaluation_report(results, summary)
        
        # Print detailed LLM judge summary if available
        if summary.get('avg_overall_score') is not None:
            self._print_llm_judge_summary(summary)
        
        print(f"\n✅ Detailed evaluation results saved to: {log_filename}")
        
        return {
            "individual_results": results,
            "summary": summary,
            "test_queries_count": len(self.test_queries),
            "log_file": log_filename
        }
    
    # =========================================================================
    # SUMMARY STATISTICS AND REPORTING
    # =========================================================================
    
    def _calculate_summary_stats(self, results: List[Dict]) -> Dict[str, float]:
        """Calculate summary statistics across all test queries"""
        retrieval_metrics = [
            "precision_at_5", "precision_at_10", "hit_rate",
            "body_part_accuracy", "equipment_accuracy"
        ]
        
        summary = {}
        
        # Calculate retrieval metrics (always available)
        for metric in retrieval_metrics:
            values = [r[metric] for r in results]
            summary[f"avg_{metric}"] = statistics.mean(values)
            summary[f"min_{metric}"] = min(values)
            summary[f"max_{metric}"] = max(values)
            summary[f"std_{metric}"] = statistics.stdev(values) if len(values) > 1 else 0.0
        
        # Calculate LLM metrics (only if LLM evaluation was successful)
        llm_results = [r for r in results if r.get('llm_evaluation') is not None]
        
        if llm_results:
            llm_metrics = ['retrieval_relevance_score', 'answer_accuracy_score', 
                          'answer_completeness_score', 'citation_quality_score', 'overall_score']
            
            for metric in llm_metrics:
                values = []
                for r in llm_results:
                    eval_data = r['llm_evaluation']
                    if metric == 'overall_score':
                        # Calculate overall if not provided
                        overall = eval_data.get('overall_score', 0)
                        if overall == 0:
                            scores = [eval_data.get('retrieval_relevance_score', 0),
                                    eval_data.get('answer_accuracy_score', 0),
                                    eval_data.get('answer_completeness_score', 0),
                                    eval_data.get('citation_quality_score', 0)]
                            overall = statistics.mean([s for s in scores if s > 0])
                        values.append(overall)
                    else:
                        values.append(eval_data.get(metric, 0))
                
                if values and any(v > 0 for v in values):
                    summary[f"avg_{metric}"] = statistics.mean(values)
                    summary[f"min_{metric}"] = min(values)
                    summary[f"max_{metric}"] = max(values)
                    summary[f"std_{metric}"] = statistics.stdev(values) if len(values) > 1 else 0.0
                else:
                    summary[f"avg_{metric}"] = None
                    summary[f"min_{metric}"] = None
                    summary[f"max_{metric}"] = None
                    summary[f"std_{metric}"] = None
        else:
            # No LLM evaluations available
            llm_metrics = ['retrieval_relevance_score', 'answer_accuracy_score', 
                          'answer_completeness_score', 'citation_quality_score', 'overall_score']
            for metric in llm_metrics:
                summary[f"avg_{metric}"] = None
                summary[f"min_{metric}"] = None
                summary[f"max_{metric}"] = None
                summary[f"std_{metric}"] = None
        
        return summary
    
    def _print_evaluation_report(self, results: List[Dict], summary: Dict[str, float]):
        """Print comprehensive evaluation report"""
        print("\n" + "="*80)
        print("📊 RAG EVALUATION REPORT")
        print("="*80)
        
        print(f"\n🎯 OVERALL PERFORMANCE:")
        print(f"  • Average Precision@5:     {summary['avg_precision_at_5']:.3f}")
        print(f"  • Average Precision@10:    {summary['avg_precision_at_10']:.3f}")
        print(f"  • Average Hit Rate:        {summary['avg_hit_rate']:.3f}")
        print(f"  • Body Part Accuracy:      {summary['avg_body_part_accuracy']:.3f}")
        print(f"  • Equipment Accuracy:      {summary['avg_equipment_accuracy']:.3f}")
        
        if summary['avg_overall_score'] is not None:
            print(f"\n🤖 LLM JUDGE SCORES (1-10 scale):")
            print(f"  • Average Retrieval Relevance:   {summary['avg_retrieval_relevance_score']:.1f}/10")
            print(f"  • Average Answer Accuracy:       {summary['avg_answer_accuracy_score']:.1f}/10")
            print(f"  • Average Answer Completeness:   {summary['avg_answer_completeness_score']:.1f}/10")
            print(f"  • Average Citation Quality:      {summary['avg_citation_quality_score']:.1f}/10")
            print(f"  • Overall LLM Score:             {summary['avg_overall_score']:.1f}/10")
        else:
            print(f"\n🤖 LLM JUDGE SCORES:")
            print(f"  • LLM Evaluation:          ❌ Not Available (Ollama not running)")
            print(f"  • To enable LLM scoring:   Run 'make ollama-setup' first")
        
        # Find best and worst performing queries (based on available metrics)
        llm_results = [r for r in results if r.get('llm_evaluation') is not None]
        if llm_results:
            def get_overall_score(result):
                eval_data = result['llm_evaluation']
                overall = eval_data.get('overall_score', 0)
                if overall == 0:
                    scores = [eval_data.get('retrieval_relevance_score', 0),
                            eval_data.get('answer_accuracy_score', 0),
                            eval_data.get('answer_completeness_score', 0),
                            eval_data.get('citation_quality_score', 0)]
                    overall = statistics.mean([s for s in scores if s > 0])
                return overall
            
            best_query = max(llm_results, key=get_overall_score)
            worst_query = min(llm_results, key=get_overall_score)
            
            print(f"\n🏆 BEST PERFORMING QUERY:")
            print(f"  • Query: '{best_query['query']}'")
            print(f"  • LLM Score: {get_overall_score(best_query):.1f}/10")
            print(f"  • Precision@5: {best_query['precision_at_5']:.3f}")
            
            print(f"\n❌ WORST PERFORMING QUERY:")
            print(f"  • Query: '{worst_query['query']}'")
            print(f"  • LLM Score: {get_overall_score(worst_query):.1f}/10")
            print(f"  • Precision@5: {worst_query['precision_at_5']:.3f}")
        else:
            best_query = max(results, key=lambda x: x['precision_at_5'])
            worst_query = min(results, key=lambda x: x['precision_at_5'])
            
            print(f"\n🏆 BEST PERFORMING QUERY (by Precision@5):")
            print(f"  • Query: '{best_query['query']}'")
            print(f"  • Precision@5: {best_query['precision_at_5']:.3f}")
            print(f"  • Hit Rate: {best_query['hit_rate']:.3f}")
            
            print(f"\n❌ WORST PERFORMING QUERY (by Precision@5):")
            print(f"  • Query: '{worst_query['query']}'")
            print(f"  • Precision@5: {worst_query['precision_at_5']:.3f}")
            print(f"  • Hit Rate: {worst_query['hit_rate']:.3f}")
        
        # Add comprehensive results table
        self._print_results_table(results)
        
        print(f"\n📈 RECOMMENDATIONS:")
        if summary['avg_precision_at_5'] < 0.6:
            print("  • Consider improving semantic search relevance")
        if summary['avg_body_part_accuracy'] < 0.7:
            print("  • Review body part filtering accuracy")
        if summary['avg_overall_score'] is None:
            print("  • Setup Ollama for LLM-based quality evaluation: make ollama-setup")
        elif summary['avg_retrieval_relevance_score'] < 7.0:
            print("  • Improve query-response relevance matching")
        if summary['avg_answer_completeness_score'] is not None and summary['avg_answer_completeness_score'] < 7.0:
            print("  • Add more comprehensive exercise details")
        
        print("="*80)
    
    def _print_results_table(self, results: List[Dict]):
        """Print comprehensive results table with all queries and scores"""
        print(f"\n📊 DETAILED RESULTS TABLE")
        print("="*130)
        
        # Header
        header = f"{'#':<3} {'Query':<35} {'P@5':<6} {'Hit':<5} {'Rel':<5} {'Acc':<5} {'Comp':<6} {'Cite':<6} {'Overall':<8}"
        print(header)
        print("-" * 130)
        
        # Data rows
        for i, result in enumerate(results, 1):
            query = result['query'][:32] + "..." if len(result['query']) > 32 else result['query']
            p_at_5 = f"{result['precision_at_5']:.2f}"
            hit_rate = f"{result['hit_rate']:.2f}"
            
            # LLM scores (if available)
            llm_eval = result.get('llm_evaluation')
            if llm_eval:
                relevance = f"{llm_eval.get('retrieval_relevance_score', 0):.1f}"
                accuracy = f"{llm_eval.get('answer_accuracy_score', 0):.1f}"
                completeness = f"{llm_eval.get('answer_completeness_score', 0):.1f}"
                citation = f"{llm_eval.get('citation_quality_score', 0):.1f}"
                overall = llm_eval.get('overall_score', 0)
                if overall == 0:
                    overall = statistics.mean([
                        llm_eval.get('retrieval_relevance_score', 0),
                        llm_eval.get('answer_accuracy_score', 0),
                        llm_eval.get('answer_completeness_score', 0),
                        llm_eval.get('citation_quality_score', 0)
                    ])
                overall_str = f"{overall:.1f}"
            else:
                relevance = accuracy = completeness = citation = overall_str = "N/A"
            
            row = f"{i:<3} {query:<35} {p_at_5:<6} {hit_rate:<5} {relevance:<5} {accuracy:<5} {completeness:<6} {citation:<6} {overall_str:<8}"
            print(row)
        
        print("-" * 130)
        print("Legend: P@5=Precision@5, Hit=Hit Rate, Rel=Retrieval Relevance, Acc=Answer Accuracy,")
        print("        Comp=Answer Completeness, Cite=Citation Quality, Overall=LLM Overall Score")
        print("="*130)
    
    def _print_llm_judge_summary(self, summary: Dict[str, float]):
        """Print beautiful LLM judge summary with progress bars"""
        print("\n" + "="*80)
        print("EVALUATION (LLM-as-a-judge with Llama 3.2)")
        print("="*80)
        print()
        print("Scores:")
        
        # Extract scores with safe defaults
        relevance = summary.get('avg_retrieval_relevance_score', 0)
        accuracy = summary.get('avg_answer_accuracy_score', 0)
        completeness = summary.get('avg_answer_completeness_score', 0)
        citation = summary.get('avg_citation_quality_score', 0)
        overall = summary.get('avg_overall_score', 0)
        
        # Create progress bars and display
        print(f"  Retrieval Relevance:   {self._create_progress_bar(relevance)} {relevance:.1f}/10")
        print(f"  Answer Accuracy:       {self._create_progress_bar(accuracy)} {accuracy:.1f}/10")
        print(f"  Answer Completeness:   {self._create_progress_bar(completeness)} {completeness:.1f}/10")
        print(f"  Citation Quality:      {self._create_progress_bar(citation)} {citation:.1f}/10")
        print(f"  ──────────────────────────────────────────────────")
        print(f"  Overall Score:         {self._create_progress_bar(overall)} {overall:.1f}/10")
        print()
        
        if self.show_explanations:
            print("Detailed Feedback:")
            # Note: We can't show specific reasons here since this is an aggregate summary
            # The individual query evaluations will show detailed reasons
            print(f"RETRIEVAL_RELEVANCE: {relevance:.0f}/10")
            print(f"REASON: Average performance across all {len(self.test_queries)} test queries for retrieving relevant exercises.")
            print()
            print(f"ANSWER_ACCURACY: {accuracy:.0f}/10")
            print(f"REASON: Average factual accuracy of exercise information, equipment, and body part targeting.")
            print()
            print(f"ANSWER_COMPLETENESS: {completeness:.0f}/10")
            print(f"REASON: Average completeness of exercise descriptions, instructions, and safety information.")
            print()
            print(f"CITATION_QUALITY: {citation:.0f}/10")
            print(f"REASON: Average quality of formatting, organization, and metadata presentation.")
            print()
            print(f"OVERALL: {overall:.1f}/10")
            print(f"REASON: Comprehensive average across all evaluation criteria for the RAG system.")
        
        print("="*80)
    
    def save_detailed_results(self, results: List[Dict], summary: Dict[str, float]) -> str:
        """Save detailed evaluation results to logs folder with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"logs/rag_evaluation_{timestamp}.json"
        
        # Prepare detailed log data
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "evaluation_config": {
                "show_explanations": self.show_explanations,
                "ollama_url": self.ollama_url,
                "total_queries": len(self.test_queries)
            },
            "summary_statistics": summary,
            "individual_evaluations": []
        }
        
        # Add individual evaluations with formatted output
        for result in results:
            eval_entry = {
                "query": result["query"],
                "response": result["response"],
                "retrieval_metrics": {
                    "precision_at_5": result["precision_at_5"],
                    "precision_at_10": result["precision_at_10"], 
                    "hit_rate": result["hit_rate"],
                    "body_part_accuracy": result["body_part_accuracy"],
                    "equipment_accuracy": result["equipment_accuracy"],
                    "retrieved_count": result["retrieved_count"]
                },
                "llm_evaluation": result.get("llm_evaluation"),
                "timestamp": result["timestamp"]
            }
            
            log_data["individual_evaluations"].append(eval_entry)
        
        # Save to JSON file
        with open(log_filename, "w") as f:
            json.dump(log_data, f, indent=2)
        
        return log_filename


# =========================================================================
# MAIN EXECUTION
# =========================================================================


def main():
    """Run RAG evaluation with optional command line arguments"""
    import sys
    
    # Parse command line arguments
    show_explanations = True
    if "--no-explanations" in sys.argv:
        show_explanations = False
        print("🔇 Running evaluation with explanations hidden")
    else:
        print("📝 Running evaluation with detailed explanations")
    
    try:
        # Initialize RAG agent
        print("🤖 Initializing RAG agent...")
        rag = FitnessRAG(persist_dir="chroma_db", top_k=10)
        
        # Initialize evaluator
        evaluator = RAGEvaluator(rag, show_explanations=show_explanations)
        
        # Run evaluation
        results = evaluator.run_evaluation()
        
        # Save detailed log file (already done in run_evaluation)
        
        # Also save a summary file for backwards compatibility
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"logs/rag_evaluation_summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump({
                "summary": results["summary"],
                "total_queries": results["test_queries_count"],
                "log_file": results["log_file"],
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        # Also create a latest summary in logs for convenience
        latest_summary = "logs/rag_evaluation_latest.json"
        with open(latest_summary, "w") as f:
            json.dump({
                "summary": results["summary"],
                "total_queries": results["test_queries_count"],
                "log_file": results["log_file"],
                "summary_file": summary_file,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"📄 Summary results also saved to: {summary_file}")
        print(f"📄 Latest summary (for convenience): {latest_summary}")
        
    except Exception as e:
        print(f"Error running evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()