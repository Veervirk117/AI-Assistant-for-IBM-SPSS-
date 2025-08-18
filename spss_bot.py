from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from enhanced_spss_vector_store import enhanced_spss_vector_store
from spss_knowledge_base import get_spss_categories, get_spss_functions
import re

class SPSSBot:
    def __init__(self):
        self.model = OllamaLLM(model="llama3.2")
        self.vector_store = enhanced_spss_vector_store
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Setup different prompt templates for different types of queries"""
        
        # Context-aware explanation prompt
        self.context_prompt = ChatPromptTemplate.from_template("""
You are a helpful SPSS assistant. First understand what the user is asking, then provide appropriate help.

User Question: {question}

Relevant SPSS Information:
{spss_knowledge}

Provide a helpful response that:
1. **Understands the context** - What are they really asking?
2. **Explains the concept** - What is this SPSS procedure/function?
3. **Asks for clarification** - Do they want steps, examples, or just understanding?
4. **Offers next steps** - Suggest what they might want to know next

Be conversational and helpful, not robotic. If their question is unclear, ask for clarification.
""")

        # Step-by-step focused prompt
        self.steps_prompt = ChatPromptTemplate.from_template("""
You are an SPSS instructor. The user specifically wants step-by-step instructions.

Function: {function_name}

Function Details:
{function_details}

Provide ONLY the step-by-step instructions in a clear, numbered format. Keep it concise and actionable.
""")

        # Casual conversation prompt
        self.casual_prompt = ChatPromptTemplate.from_template("""
You are a friendly SPSS assistant. The user is having a casual conversation or asking unclear questions.

User Message: {question}

Respond naturally and warmly. If they ask about SPSS, offer help. If their question is unclear, ask for clarification.
If they're just testing or being casual, be friendly and supportive.
""")

        # Default help prompt
        self.help_prompt = ChatPromptTemplate.from_template("""
You are a helpful SPSS assistant. The user needs guidance on what they can ask.

Available SPSS Functions: {functions}
Available Categories: {categories}

Give them a friendly overview of what you can help with and suggest some example questions they might ask.
""")
    
    def analyze_query(self, question):
        """Analyze user query to determine the type of assistance needed"""
        question_lower = question.lower()
        
        # Check for casual/ambiguous queries
        if self._is_ambiguous_or_casual(question):
            return "casual", None
        
        # Check for specific step-by-step requests
        if self._explicitly_wants_steps(question):
            return "steps", None
        
        # Check for concept explanation requests
        if self._wants_concept_explanation(question):
            return "concept", None
        
        # Check for specific function requests
        spss_functions = get_spss_functions()
        for function in spss_functions:
            if function.lower() in question_lower:
                return "function_specific", function
        
        # Check for workflow/planning requests
        workflow_keywords = ["workflow", "plan", "process", "from start to finish"]
        if any(keyword in question_lower for keyword in workflow_keywords):
            return "workflow", None
        
        # Check for category-specific requests
        categories = get_spss_categories()
        for category in categories:
            if category.lower() in question_lower:
                return "category_overview", category
        
        # Default to context-aware explanation
        return "context_aware", None
    
    def _is_ambiguous_or_casual(self, question):
        """Check if query is ambiguous or casual"""
        question_lower = question.lower()
        
        # Very short or unclear queries
        if len(question.strip()) < 5:
            return True
        
        # Testing queries
        test_patterns = ["test", "testing", "123", "hello", "hi", "hey", "what's up"]
        if any(pattern in question_lower for pattern in test_patterns):
            return True
        
        # Unclear statistical terms without context
        unclear_patterns = ["what is", "explain", "tell me about", "how does"]
        if any(pattern in question_lower for pattern in unclear_patterns) and len(question.split()) < 6:
            return True
        
        return False
    
    def _explicitly_wants_steps(self, question):
        """Check if user explicitly wants step-by-step instructions"""
        step_keywords = ["step by step", "step-by-step", "steps", "how to do", "procedure", "process", "walk me through"]
        return any(keyword in question.lower() for keyword in step_keywords)
    
    def _wants_concept_explanation(self, question):
        """Check if user wants concept explanation"""
        concept_patterns = ["what is", "explain", "tell me about", "describe", "define", "meaning of"]
        return any(pattern in question.lower() for pattern in concept_patterns)
    
    def get_response(self, question):
        """Get appropriate response based on query type"""
        query_type, specific_item = self.analyze_query(question)
        
        if query_type == "casual":
            return self._get_casual_response(question)
        elif query_type == "steps":
            return self._get_steps_response(question)
        elif query_type == "concept":
            return self._get_concept_response(question)
        elif query_type == "function_specific":
            return self._get_function_response(question, specific_item)
        elif query_type == "workflow":
            return self._get_workflow_response(question)
        elif query_type == "category_overview":
            return self._get_category_response(question, specific_item)
        else:
            return self._get_context_aware_response(question)
    
    def _get_casual_response(self, question):
        """Handle casual conversation and ambiguous queries"""
        if "test" in question.lower():
            return "Hey! I see you're testing something. What would you like to test in SPSS? I can help you with:\n• Statistical procedures\n• Data analysis techniques\n• SPSS functions\n\nWhat are you working on?"
        
        if len(question.strip()) < 5:
            return "Hi there! I'm here to help with SPSS. What would you like to know about? You can ask me about:\n• Statistical procedures (like t-tests, ANOVA, regression)\n• SPSS functions and commands\n• Data analysis workflows\n\nWhat's on your mind?"
        
        # Generic casual response
        chain = self.casual_prompt | self.model
        result = chain.invoke({"question": question})
        return result
    
    def _get_concept_response(self, question):
        """Explain concepts and ask for clarification"""
        # Search for relevant SPSS knowledge
        relevant_docs = self.vector_store.search_spss_function(question)
        
        if not relevant_docs:
            return f"I'd be happy to explain that! However, I need a bit more context about what you're asking. Could you be more specific? For example:\n• 'What is ANOVA?'\n• 'Explain factor analysis'\n• 'What does regression mean?'\n\nWhat SPSS concept would you like me to explain?"
        
        # Use the most relevant document
        doc = relevant_docs[0]
        
        # Create simple concept explanation
        concept_explanation = f"""
**{question}** - Here's what I found:

{doc.page_content[:300]}...

**Would you like me to show you the step-by-step instructions for this?**
"""
        
        return concept_explanation
    
    def _get_steps_response(self, question):
        """Get step-by-step instructions for any SPSS procedure"""
        # Search for relevant SPSS knowledge
        relevant_docs = self.vector_store.search_spss_function(question)
        
        if not relevant_docs:
            return "I'd be happy to help with step-by-step SPSS instructions! Could you be more specific about what procedure you need help with? For example: 'How to run a t-test step by step' or 'Step-by-step instructions for creating a histogram'"
        
        # Use the most relevant document
        doc = relevant_docs[0]
        
        chain = self.steps_prompt | self.model
        result = chain.invoke({
            "function_name": doc.metadata.get('function', 'Unknown'),
            "function_details": doc.page_content
        })
        
        return result
    
    def _get_function_response(self, question, function_name):
        """Get focused response for a specific SPSS function"""
        function_details = self.vector_store.get_function_details(function_name)
        
        if not function_details:
            return f"I can help you with SPSS functions! What specifically would you like to know about '{function_name}'? You can ask for step-by-step instructions or a general explanation."
        
        # Check if user wants steps
        if any(word in question.lower() for word in ["step", "how to", "procedure"]):
            return self._get_steps_response(question)
        
        # Give simple concept explanation
        concept_explanation = f"""
**{function_name}** - Here's what this SPSS function does:

{function_details.page_content[:300]}...

**Would you like me to show you the step-by-step instructions?**
"""
        
        return concept_explanation
    
    def _get_workflow_response(self, question):
        """Get workflow planning response"""
        workflow_suggestions = self.vector_store.get_workflow_suggestions(question)
        
        if not workflow_suggestions:
            return "I can help you plan SPSS workflows! What specific analysis are you trying to accomplish? For example: 'Workflow for survey analysis' or 'Process for regression analysis'"
        
        # Format workflow suggestions by category
        formatted_suggestions = {}
        for category, docs in workflow_suggestions.items():
            formatted_suggestions[category] = [doc.metadata.get('function') for doc in docs]
        
        context = "\n".join([f"{cat}: {', '.join(funcs)}" for cat, funcs in formatted_suggestions.items()])
        
        chain = self.context_prompt | self.model
        result = chain.invoke({
            "question": question,
            "spss_knowledge": context
        })
        
        return result
    
    def _get_category_response(self, question, category):
        """Get response for category overview requests"""
        category_docs = self.vector_store.get_category_overview(category)
        
        if not category_docs:
            return f"I can help you explore SPSS categories! What would you like to know about '{category}' specifically?"
        
        # Format category information
        context = f"Category: {category}\n\nFunctions available:\n"
        for doc in category_docs:
            context += f"- {doc.metadata.get('function')} ({doc.metadata.get('difficulty')})\n"
            context += f"  {doc.page_content[:150]}...\n\n"
        
        chain = self.context_prompt | self.model
        result = chain.invoke({
            "question": question,
            "spss_knowledge": context
        })
        
        return result
    
    def _get_context_aware_response(self, question):
        """Get context-aware response for general queries"""
        # Search for relevant SPSS knowledge
        relevant_docs = self.vector_store.search_spss_function(question)
        
        if not relevant_docs:
            return "I'm here to help with SPSS! You can ask me about:\n• Specific functions (like 't-test' or 'regression')\n• Concepts (like 'What is ANOVA?')\n• Step-by-step instructions\n• Workflow planning\n\nWhat would you like to know?"
        
        # Format retrieved knowledge
        context = ""
        for i, doc in enumerate(relevant_docs[:2]):  # Use top 2 most relevant
            context += f"{doc.metadata.get('function')} ({doc.metadata.get('category')}): {doc.page_content[:200]}...\n\n"
        
        chain = self.context_prompt | self.model
        result = chain.invoke({
            "question": question,
            "spss_knowledge": context
        })
        
        return result
    
    def get_default_help(self):
        """Get default help message"""
        functions = get_spss_functions()
        categories = get_spss_categories()
        
        chain = self.help_prompt | self.model
        result = chain.invoke({
            "functions": ", ".join(functions[:10]) + "...",
            "categories": ", ".join(categories)
        })
        
        return result
    
    def get_available_functions(self):
        """Get list of available SPSS functions for user reference"""
        return get_spss_functions()
    
    def get_available_categories(self):
        """Get list of available SPSS categories for user reference"""
        return get_spss_categories()
    
    def get_functions_by_difficulty(self, difficulty):
        """Get functions filtered by difficulty level"""
        from spss_knowledge_base import get_spss_by_difficulty
        return get_spss_by_difficulty(difficulty)
    
    def get_dataset_recommendations(self, dataset_description):
        """Get SPSS procedure recommendations based on dataset characteristics"""
        return self.vector_store.get_dataset_recommendations(dataset_description)
    
    def get_statistical_insights(self, query):
        """Get statistical insights and recommendations"""
        return self.vector_store.get_statistical_insights(query)
    
    def get_performance_metrics(self):
        """Get performance metrics for the vector store"""
        return self.vector_store.get_performance_metrics()

def main():
    """Main interaction loop for the SPSS bot"""
    bot = SPSSBot()
    
    print("🤖 Welcome to the SPSS Assistant!")
    print("=" * 50)
    print("I can help you with SPSS analysis. Just ask me anything!")
    print("\nType 'help' for options, 'metrics' for performance, 'insights' for enhanced features, 'quit' to exit, or ask your question.")
    print("=" * 50)
    
    while True:
        try:
            question = input("\n📊 What can I help you with? ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye! Happy analyzing!")
                break
            
            elif question.lower() == 'help':
                print("\n📚 Available SPSS Functions:")
                functions = bot.get_available_functions()
                for i, func in enumerate(functions, 1):
                    print(f"  {i:2d}. {func}")
                continue
            
            elif question.lower() == 'categories':
                print("\n📂 Available SPSS Categories:")
                categories = bot.get_available_categories()
                for i, cat in enumerate(categories, 1):
                    print(f"  {i:2d}. {cat}")
                continue
            
            elif question.lower() == 'difficulty':
                print("\n📊 Functions by Difficulty Level:")
                for level in ['Beginner', 'Intermediate', 'Advanced']:
                    funcs = bot.get_functions_by_difficulty(level)
                    print(f"\n{level} Level:")
                    for func in funcs:
                        print(f"  • {func['function']}")
                continue
            
            elif question.lower() == 'metrics':
                print("\n📈 Enhanced Vector Store Performance Metrics:")
                metrics = bot.get_performance_metrics()
                for key, value in metrics.items():
                    print(f"  {key}: {value}")
                continue
            
            elif question.lower() == 'insights':
                print("\n💡 Enhanced Features Available:")
                print("  • dataset_recommendations - Get SPSS procedure recommendations")
                print("  • statistical_insights - Get statistical insights")
                print("  • Try asking: 'What SPSS procedures should I use for survey data?'")
                print("  • Try asking: 'Give me insights on regression analysis'")
                continue
            
            elif not question:
                # Show default help when user just hits enter
                print("\n" + "=" * 50)
                print("💡 Here's what I can help you with:")
                print("=" * 50)
                help_response = bot.get_default_help()
                print(help_response)
                print("=" * 50)
                continue
            
            print("\n" + "=" * 50)
            print("🤖 SPSS Assistant:")
            print("=" * 50)
            response = bot.get_response(question)
            print(response)
            print("=" * 50)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Feel free to return for more SPSS help.")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try rephrasing your question or ask for 'help' to see available functions.")

if __name__ == "__main__":
    main() 