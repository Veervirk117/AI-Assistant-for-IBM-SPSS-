#!/usr/bin/env python3
"""
Test script for Improved SPSS Bot
Demonstrates better context understanding and conversational responses
"""

from spss_bot import SPSSBot

def test_improved_functionality():
    """Test the improved bot capabilities"""
    print("🧪 Testing Improved SPSS Bot Functionality")
    print("=" * 60)
    
    bot = SPSSBot()
    
    # Test 1: Ambiguous/Casual Queries
    print("\n🔍 Test 1: Ambiguous/Casual Queries")
    print("-" * 40)
    
    test_queries = [
        "testing 123",
        "hi",
        "what is",
        "explain"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        try:
            response = bot.get_response(query)
            print(f"Response: {response[:200]}...")
        except Exception as e:
            print(f"Error: {e}")
    
    # Test 2: Concept Explanation (without steps)
    print("\n📚 Test 2: Concept Explanation")
    print("-" * 40)
    
    concept_queries = [
        "What is ANOVA?",
        "Explain factor analysis",
        "What does regression mean?"
    ]
    
    for query in concept_queries:
        print(f"\nQuery: '{query}'")
        try:
            response = bot.get_response(query)
            print(f"Response: {response[:300]}...")
        except Exception as e:
            print(f"Error: {e}")
    
    # Test 3: Explicit Step Requests
    print("\n📋 Test 3: Explicit Step Requests")
    print("-" * 40)
    
    step_queries = [
        "Show me step by step how to run a t-test",
        "Walk me through the ANOVA procedure",
        "Give me the steps for regression analysis"
    ]
    
    for query in step_queries:
        print(f"\nQuery: '{query}'")
        try:
            response = bot.get_response(query)
            print(f"Response: {response[:300]}...")
        except Exception as e:
            print(f"Error: {e}")
    
    # Test 4: Context-Aware Responses
    print("\n🧠 Test 4: Context-Aware Responses")
    print("-" * 40)
    
    context_queries = [
        "I have survey data with Likert scales",
        "How do I analyze categorical variables?",
        "What's the best way to handle missing data?"
    ]
    
    for query in context_queries:
        print(f"\nQuery: '{query}'")
        try:
            response = bot.get_response(query)
            print(f"Response: {response[:300]}...")
        except Exception as e:
            print(f"Error: {e}")
    
    print("\n✅ Improved bot testing completed!")

if __name__ == "__main__":
    test_improved_functionality() 