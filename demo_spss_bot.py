#!/usr/bin/env python3
"""
Demo script for the SPSS Bot
This script demonstrates the key features of the SPSS statistical analysis assistant.
"""

from spss_bot import SPSSBot
from spss_knowledge_base import get_spss_categories, get_spss_functions

def demo_spss_bot():
    """Demonstrate the SPSS bot capabilities"""
    print("🧪 SPSS Bot Demo - Testing Core Functionality")
    print("=" * 60)
    
    # Initialize the bot
    bot = SPSSBot()
    
    # Demo 1: Show available categories
    print("\n📂 Demo 1: Available SPSS Categories")
    print("-" * 40)
    categories = bot.get_available_categories()
    for i, category in enumerate(categories, 1):
        print(f"  {i:2d}. {category}")
    
    # Demo 2: Show available functions
    print("\n📚 Demo 2: Available SPSS Functions")
    print("-" * 40)
    functions = bot.get_available_functions()
    for i, func in enumerate(functions, 1):
        print(f"  {i:2d}. {func}")
    
    # Demo 3: Test specific function query
    print("\n🔍 Demo 3: Testing Function-Specific Query")
    print("-" * 40)
    test_question = "How do I perform a t-test in SPSS?"
    print(f"Question: {test_question}")
    
    try:
        response = bot.get_response(test_question)
        print(f"\nResponse Preview (first 200 chars):")
        print(response[:200] + "..." if len(response) > 200 else response)
    except Exception as e:
        print(f"Error: {e}")
    
    # Demo 4: Test workflow query
    print("\n🔄 Demo 4: Testing Workflow Query")
    print("-" * 40)
    test_question = "How do I analyze survey data from start to finish?"
    print(f"Question: {test_question}")
    
    try:
        response = bot.get_response(test_question)
        print(f"\nResponse Preview (first 200 chars):")
        print(response[:200] + "..." if len(response) > 200 else response)
    except Exception as e:
        print(f"Error: {e}")
    
    # Demo 5: Test category query
    print("\n📊 Demo 5: Testing Category Query")
    print("-" * 40)
    test_question = "What can I do with descriptive statistics?"
    print(f"Question: {test_question}")
    
    try:
        response = bot.get_response(test_question)
        print(f"\nResponse Preview (first 200 chars):")
        print(response[:200] + "..." if len(response) > 200 else response)
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n✅ Demo completed! The SPSS bot is ready to use.")
    print("\nTo start using the bot interactively, run: python spss_bot.py")

def test_vector_store():
    """Test the vector store functionality"""
    print("\n🧪 Testing Vector Store Functionality")
    print("=" * 60)
    
    try:
        from spss_vector_store import spss_vector_store
        
        # Test search functionality
        print("Testing search functionality...")
        results = spss_vector_store.search_spss_function("t-test")
        print(f"Found {len(results)} relevant documents for 't-test'")
        
        # Test function details
        print("Testing function details...")
        details = spss_vector_store.get_function_details("Independent Samples T-Test")
        if details:
            print(f"Found details for: {details.metadata.get('function')}")
            print(f"Category: {details.metadata.get('category')}")
            print(f"Difficulty: {details.metadata.get('difficulty')}")
        else:
            print("Function details not found")
        
        # Test category overview
        print("Testing category overview...")
        category_docs = spss_vector_store.get_category_overview("Descriptive Statistics")
        print(f"Found {len(category_docs)} documents in Descriptive Statistics category")
        
        print("✅ Vector store tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting SPSS Bot Demo...")
    
    # Test vector store first
    test_vector_store()
    
    # Run main demo
    demo_spss_bot()
    
    print("\n🎉 All demos completed! Your SPSS bot is ready to help with statistical analysis.") 