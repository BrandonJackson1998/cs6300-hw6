"""
RAG Evaluation Framework for Fitness Exercise Recommendations
Implements comprehensive evaluation metrics and LLM-as-Judge assessment
"""

import json
import statistics
from typing import List, Dict, Any, Tuple, Optional
from .rag_agent import FitnessRAG
import requests
import time
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
    
    def evaluate_response_quality(self, query: str, response: str) -> Dict[str, Any]:
        """
        Use Ollama (Llama 3.2) as judge to evaluate response quality with detailed feedback
        
        Args:
            query: Original user query
            response: RAG system response
            
        Returns:
            Dict with scores (1-10 scale) and detailed explanations
        """
        evaluation_prompt = f"""
You are an expert fitness trainer evaluating exercise recommendations. Please rate the following response on a scale of 1-10 for each criterion and provide detailed explanations.

USER QUERY: {query}

SYSTEM RESPONSE: {response}

Please evaluate and respond with a JSON object containing scores and detailed reasoning:

1. RETRIEVAL_RELEVANCE (1-10): How well does the response address the specific query?
   - 10: Perfect match, exactly what was asked for
   - 7-9: Good match with minor irrelevant content  
   - 4-6: Partially relevant but missing key aspects
   - 1-3: Mostly irrelevant to the query

2. ANSWER_ACCURACY (1-10): How accurate and trustworthy is the fitness information?
   - 10: All exercise information is accurate and safe
   - 7-9: Mostly accurate with minor issues
   - 4-6: Some accuracy concerns
   - 1-3: Significant accuracy problems

3. ANSWER_COMPLETENESS (1-10): How complete and helpful is the response?
   - 10: Comprehensive, includes all necessary details
   - 7-9: Good detail level, minor gaps
   - 4-6: Basic information but lacks depth
   - 1-3: Insufficient information

4. CITATION_QUALITY (1-10): How well organized and formatted is the response?
   - 10: Excellent structure, clear formatting, easy to follow
   - 7-9: Good organization with minor formatting issues
   - 4-6: Basic organization, some clarity issues
   - 1-3: Poor structure, hard to understand

Respond with JSON only in this exact format:
{{
  "retrieval_relevance": {{"score": X, "reason": "detailed explanation"}},
  "answer_accuracy": {{"score": X, "reason": "detailed explanation"}},
  "answer_completeness": {{"score": X, "reason": "detailed explanation"}},
  "citation_quality": {{"score": X, "reason": "detailed explanation"}},
  "overall": {{"score": X, "reason": "overall assessment"}}
}}
"""
        
        try:
            # Check if Ollama is running first
            test_response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if test_response.status_code != 200:
                raise ConnectionError(f"Ollama not accessible at {self.ollama_url}")
            
            # Call Ollama API
            response_data = {
                "model": "llama3.2",
                "prompt": evaluation_prompt,
                "stream": False
            }
            
            llm_response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=response_data,
                timeout=45
            )
            
            if llm_response.status_code == 200:
                result = llm_response.json()
                evaluation_text = result.get('response', '').strip()
                
                # Try to parse JSON from response
                try:
                    # Look for JSON in the response
                    import re
                    json_match = re.search(r'\{.*\}', evaluation_text, re.DOTALL)
                    if json_match:
                        scores = json.loads(json_match.group())
                        
                        # Extract scores and reasons
                        evaluation = {}
                        for metric in ['retrieval_relevance', 'answer_accuracy', 'answer_completeness', 'citation_quality', 'overall']:
                            if metric in scores:
                                evaluation[f"{metric}_score"] = float(scores[metric].get("score", 5))
                                evaluation[f"{metric}_reason"] = scores[metric].get("reason", "No explanation provided")
                            else:
                                evaluation[f"{metric}_score"] = 5.0
                                evaluation[f"{metric}_reason"] = "Evaluation not provided"
                        
                        return evaluation
                except json.JSONDecodeError as e:
                    print(f"JSON parse error: {e}")
                    print(f"Raw response: {evaluation_text[:500]}...")
            
            raise Exception(f"Failed to get valid response from Ollama API")
            
        except (requests.exceptions.ConnectionError, ConnectionError):
            raise ConnectionError(
                f"❌ Ollama is not running!\n"
                f"   Please start Ollama first:\n"
                f"   • Run: make ollama-setup\n"
                f"   • Or: ollama serve\n"
                f"   • Then: make evaluate"
            )
            
        except Exception as e:
            raise Exception(f"LLM evaluation failed: {str(e)}")
    
    def _create_progress_bar(self, score: float, max_score: float = 10.0) -> str:
        """Create a visual progress bar for scores"""
        filled_blocks = int((score / max_score) * 10)
        empty_blocks = 10 - filled_blocks
        return "█" * filled_blocks + "░" * empty_blocks
    
    def _format_detailed_evaluation(self, evaluation: Dict[str, Any]) -> str:
        """Format LLM evaluation with progress bars and explanations"""
        if not evaluation:
            return "LLM evaluation not available"
        
        # Extract scores
        relevance = evaluation.get('retrieval_relevance_score', 0)
        accuracy = evaluation.get('answer_accuracy_score', 0) 
        completeness = evaluation.get('answer_completeness_score', 0)
        citation = evaluation.get('citation_quality_score', 0)
        overall = evaluation.get('overall_score', 0)
        
        # Calculate average if overall not provided
        if overall == 0:
            overall = statistics.mean([relevance, accuracy, completeness, citation])
        
        output = []
        output.append("="*80)
        output.append("EVALUATION (LLM-as-a-judge with Llama 3.2)")
        output.append("="*80)
        output.append("")
        output.append("Scores:")
        output.append(f"  Retrieval Relevance:   {self._create_progress_bar(relevance)} {relevance:.1f}/10")
        output.append(f"  Answer Accuracy:       {self._create_progress_bar(accuracy)} {accuracy:.1f}/10")
        output.append(f"  Answer Completeness:   {self._create_progress_bar(completeness)} {completeness:.1f}/10")
        output.append(f"  Citation Quality:      {self._create_progress_bar(citation)} {citation:.1f}/10")
        output.append(f"  ──────────────────────────────────────────────────")
        output.append(f"  Overall Score:         {self._create_progress_bar(overall)} {overall:.1f}/10")
        output.append("")
        
        if self.show_explanations:
            output.append("Detailed Feedback:")
            output.append(f"RETIEVAL_RELEVANCE: {relevance:.0f}/10")
            output.append(f"REASON: {evaluation.get('retrieval_relevance_reason', 'No explanation provided')}")
            output.append("")
            output.append(f"ANSWER_ACCURACY: {accuracy:.0f}/10")
            output.append(f"REASON: {evaluation.get('answer_accuracy_reason', 'No explanation provided')}")
            output.append("")
            output.append(f"ANSWER_COMPLETENESS: {completeness:.0f}/10")
            output.append(f"REASON: {evaluation.get('answer_completeness_reason', 'No explanation provided')}")
            output.append("")
            output.append(f"CITATION_QUALITY: {citation:.0f}/10")
            output.append(f"REASON: {evaluation.get('citation_quality_reason', 'No explanation provided')}")
            output.append("")
            output.append(f"OVERALL: {overall:.1f}/10")
            output.append(f"REASON: {evaluation.get('overall_reason', 'No explanation provided')}")
        
        output.append("="*80)
        return "\n".join(output)
    
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
                
                print(f"  📊 P@5: {precision_5:.2f} | Hit Rate: {hit_rate:.2f} | LLM Overall: {overall:.1f}/10")
                
            except ConnectionError as e:
                print(f"\n{str(e)}")
                print("\n⚠️ Skipping LLM evaluation - running retrieval metrics only...")
                llm_evaluation = None
            except Exception as e:
                print(f"⚠️ LLM evaluation failed: {e}")
                llm_evaluation = None
            
            if not llm_evaluation:
                print(f"  📊 P@5: {precision_5:.2f} | Hit Rate: {hit_rate:.2f} | LLM: N/A (Ollama not running)")
            
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
        
        print(f"\n✅ Detailed evaluation results saved to: {log_filename}")
        
        return {
            "individual_results": results,
            "summary": summary,
            "test_queries_count": len(self.test_queries),
            "log_file": log_filename
        }
    
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
            
            # Add formatted evaluation if LLM evaluation exists
            if result.get("llm_evaluation"):
                eval_entry["formatted_evaluation"] = self._format_detailed_evaluation(result["llm_evaluation"])
            
            log_data["individual_evaluations"].append(eval_entry)
        
        # Save to JSON file
        with open(log_filename, "w") as f:
            json.dump(log_data, f, indent=2)
        
        return log_filename


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
        
        print(f"📄 Summary results also saved to: {summary_file}")
        print(f"📄 Latest summary (for convenience): {latest_summary}")
        
    except Exception as e:
        
        # Also create a latest summary in root for convenience
        latest_summary = "rag_evaluation_results.json"
        with open(latest_summary, "w") as f:
            json.dump({
                "summary": results["summary"],
                "total_queries": results["test_queries_count"],
                "log_file": results["log_file"],
                "summary_file": summary_file,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
    except Exception as e:
        print(f"Error running evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()