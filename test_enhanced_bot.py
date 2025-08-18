#!/usr/bin/env python3
"""
Test script to compare old vs enhanced SPSS bot responses
"""

from spss_bot import SPSSBot

def test_enhanced_capabilities():
    """Test the enhanced capabilities of the SPSS bot"""
    print("🧪 Testing Enhanced SPSS Bot Capabilities")
    print("=" * 60)
    
    bot = SPSSBot()
    
    # Test 1: Basic search comparison
    print("\n🔍 Test 1: Basic Search - 't-test'")
    print("-" * 40)
    try:
        response = bot.get_response("How do I perform a t-test in SPSS?")
        print(f"Response Length: {len(response)} characters")
        print(f"Response Preview: {response[:300]}...")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Enhanced dataset recommendations
    print("\n📊 Test 2: Dataset Recommendations")
    print("-" * 40)
    try:
        recommendations = bot.get_dataset_recommendations("survey data with Likert scale responses")
        print(f"Found {len(recommendations)} recommendations")
        if recommendations:
            print(f"First recommendation: {recommendations[0].metadata.get('section_header', 'Unknown')}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: Statistical insights
    print("\n💡 Test 3: Statistical Insights")
    print("-" * 40)
    try:
        insights = bot.get_statistical_insights("regression analysis assumptions")
        print("Insights extracted:")
        for key, value in insights.items():
            if value:
                print(f"  {key}: {len(value)} items")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 4: Performance metrics
    print("\n📈 Test 4: Performance Metrics")
    print("-" * 40)
    try:
        metrics = bot.get_performance_metrics()
        print("Performance Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 5: Advanced query
    print("\n🚀 Test 5: Advanced Query")
    print("-" * 40)
    try:
        response = bot.get_response("What SPSS procedures should I use for analyzing survey data with multiple response questions?")
        print(f"Response Length: {len(response)} characters")
        print(f"Response Preview: {response[:300]}...")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n✅ Enhanced bot testing completed!")

if __name__ == "__main__":
    test_enhanced_capabilities() 