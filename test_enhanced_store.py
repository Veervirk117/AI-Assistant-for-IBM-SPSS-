#!/usr/bin/env python3
"""
Test script for Enhanced SPSS Vector Store
"""

from enhanced_spss_vector_store import EnhancedSPSSVectorStore

def main():
    """Test the enhanced vector store"""
    print("🧪 Testing Enhanced SPSS Vector Store")
    print("=" * 50)
    
    try:
        # Initialize the store
        store = EnhancedSPSSVectorStore()
        
        # Test basic functionality
        print("Testing search functionality...")
        results = store.search_spss_function("t-test")
        print(f"Found {len(results)} relevant documents for 't-test'")
        
        # Test performance metrics
        print("\nPerformance Metrics:")
        metrics = store.get_performance_metrics()
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        # Test a specific search
        print("\nTesting specific search...")
        results = store.search_spss_function("regression analysis")
        print(f"Found {len(results)} documents for 'regression analysis'")
        
        if results:
            print(f"First result metadata: {results[0].metadata}")
            print(f"First result preview: {results[0].page_content[:200]}...")
        
        print("\n✅ Enhanced vector store is working!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 