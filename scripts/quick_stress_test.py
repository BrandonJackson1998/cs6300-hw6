#!/usr/bin/env python3
"""
Quick RAG Stress Test - Fast evaluation with essential metrics
"""

import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.rag_agent import FitnessRAG
from src.tools import exercise_lookup
import statistics


def quick_stress_test():
    """Run a quick stress test with diverse queries"""
    
    # Essential test queries covering key scenarios
    test_queries = [
        # Basic queries
        "chest exercises",
        "back workout",
        "leg training",
        "shoulder exercises",
        "abs workout",
        
        # Equipment specific
        "dumbbell exercises",
        "barbell workout",
        "bodyweight exercises",
        "cable training",
        "kettlebell routine",
        
        # Boolean logic
        "chest or shoulder exercises",
        "back and bicep workout",
        "legs or glutes training",
        
        # Quality focused
        "best chest exercises",
        "highest rated back workout",
        "exercises with detailed instructions",
        
        # Complex queries
        "chest exercises with dumbbells for muscle building",
        "beginner back workout at home",
        "advanced leg training with barbells",
        
        # Edge cases
        "exercise",
        "workout",
        "fitness routine",
        "training plan",
        "",  # Empty query
    ]
    
    print("🚀 Quick RAG Stress Test")
    print("="*50)
    print(f"Testing {len(test_queries)} diverse queries...")
    
    # Initialize RAG
    try:
        rag = FitnessRAG(persist_dir="chroma_db", top_k=10)
        print("✓ RAG agent initialized")
    except Exception as e:
        print(f"❌ Failed to initialize RAG: {e}")
        return
    
    # Test RAG agent directly
    print("\n1️⃣ Testing RAG Agent...")
    rag_times = []
    rag_success = 0
    
    for i, query in enumerate(test_queries, 1):
        print(f"  [{i:2d}/{len(test_queries)}] '{query[:30]}{'...' if len(query) > 30 else ''}'", end="")
        
        start_time = time.time()
        try:
            result = rag.query(query, verbose=False)
            end_time = time.time()
            query_time = end_time - start_time
            rag_times.append(query_time)
            rag_success += 1
            print(f" ✓ {query_time:.3f}s ({result['retrieved_count']} docs)")
        except Exception as e:
            end_time = time.time()
            query_time = end_time - start_time
            print(f" ❌ {query_time:.3f}s - {str(e)[:50]}")
    
    # Test tool wrapper
    print("\n2️⃣ Testing Tool Wrapper...")
    tool_times = []
    tool_success = 0
    
    # Test subset for speed
    tool_test_queries = test_queries[:10]
    
    for i, query in enumerate(tool_test_queries, 1):
        print(f"  [{i:2d}/{len(tool_test_queries)}] '{query[:30]}{'...' if len(query) > 30 else ''}'", end="")
        
        start_time = time.time()
        try:
            result = exercise_lookup(query)
            end_time = time.time()
            query_time = end_time - start_time
            tool_times.append(query_time)
            tool_success += 1
            response_length = len(result)
            print(f" ✓ {query_time:.3f}s ({response_length} chars)")
        except Exception as e:
            end_time = time.time()
            query_time = end_time - start_time
            print(f" ❌ {query_time:.3f}s - {str(e)[:50]}")
    
    # Performance summary
    print("\n📊 Performance Summary")
    print("="*50)
    
    if rag_times:
        print(f"🎯 RAG Agent ({len(test_queries)} queries):")
        print(f"  • Success Rate:     {rag_success}/{len(test_queries)} ({rag_success/len(test_queries)*100:.1f}%)")
        print(f"  • Avg Time:         {statistics.mean(rag_times):.3f}s")
        print(f"  • Min Time:         {min(rag_times):.3f}s")
        print(f"  • Max Time:         {max(rag_times):.3f}s")
        print(f"  • Total Time:       {sum(rag_times):.3f}s")
    
    if tool_times:
        print(f"\n🔧 Tool Wrapper ({len(tool_test_queries)} queries):")
        print(f"  • Success Rate:     {tool_success}/{len(tool_test_queries)} ({tool_success/len(tool_test_queries)*100:.1f}%)")
        print(f"  • Avg Time:         {statistics.mean(tool_times):.3f}s")
        print(f"  • Min Time:         {min(tool_times):.3f}s")
        print(f"  • Max Time:         {max(tool_times):.3f}s")
        print(f"  • Total Time:       {sum(tool_times):.3f}s")
    
    # Speed comparison
    if rag_times and tool_times:
        rag_avg = statistics.mean(rag_times)
        tool_avg = statistics.mean(tool_times)
        if rag_avg > tool_avg:
            speedup = rag_avg / tool_avg
            print(f"\n⚡ Tool wrapper is {speedup:.1f}x faster than direct RAG")
        else:
            slowdown = tool_avg / rag_avg
            print(f"\n⚡ Direct RAG is {slowdown:.1f}x faster than tool wrapper")
    
    # Performance recommendations
    print("\n💡 Recommendations:")
    if rag_times:
        avg_time = statistics.mean(rag_times)
        if avg_time > 2.0:
            print("  • Query times >2s - consider optimizing vector search")
        elif avg_time > 1.0:
            print("  • Query times >1s - performance is acceptable but could be improved")
        else:
            print("  • Query times <1s - excellent performance!")
        
        if rag_success < len(test_queries):
            failed = len(test_queries) - rag_success
            print(f"  • {failed} failed queries - check error handling")
    
    print("\n✅ Quick stress test complete!")


if __name__ == "__main__":
    quick_stress_test()