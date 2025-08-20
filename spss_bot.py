from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from chroma_vector_storage import enhanced_spss_vector_store
from spss_knowledge_base import get_spss_categories, get_spss_functions
import re

class SPSSBot:
    def __init__(self):
        self.model = OllamaLLM(model="llama3.2")
        self.vector_store = enhanced_spss_vector_store
        self.conversation_history = []
        self.current_context = {}
        self._setup_prompts()
    
    def _setup_prompts(self):
        """Setup different prompt templates for different types of queries"""
        
        # Context-aware explanation prompt
        self.context_prompt = ChatPromptTemplate.from_template("""
You are a professional SPSS assistant. Provide clear, concise help based on the user's question.

User Question: {question}

Relevant SPSS Information:
{spss_knowledge}

Give clear, concise answers that directly address the question.
Explain concepts simply when needed.
Provide practical guidance or next steps.
Keep the tone professional and helpful, but not overly formal.
Avoid unnecessary lists or numbering unless it improves clarity.

Keep responses focused and actionable. If their question is unclear, ask for specific clarification.
""")

        # Step-by-step focused prompt
        self.steps_prompt = ChatPromptTemplate.from_template("""
You are an SPSS instructor. Provide clear, numbered step-by-step instructions.

Function: {function_name}

Function Details:
{function_details}

Provide ONLY the step-by-step instructions in a clear, numbered format. Keep it concise and actionable.
""")

        # Casual conversation prompt
        self.casual_prompt = ChatPromptTemplate.from_template("""
You are a professional SPSS assistant. The user is having a casual conversation or asking unclear questions.

User Message: {question}

Respond professionally and helpfully. If they ask about SPSS, provide assistance. If their question is unclear, ask for clarification.
If they're testing or being casual, redirect them to SPSS-related topics professionally.
""")

        # Default help prompt
        self.help_prompt = ChatPromptTemplate.from_template("""
You are a professional SPSS assistant. Provide clear guidance on available functions.

Available SPSS Functions: {functions}
Available Categories: {categories}

Give a clear overview of what you can help with and suggest specific questions they might ask.
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
    
    def _analyze_conversation_context(self, question):
        """Analyze question in context of conversation history"""
        question_lower = question.lower()
        
        # Check for follow-up responses
        follow_up_patterns = [
            "yes", "yeah", "sure", "okay", "ok", "please", "show me", "give me",
            "steps", "instructions", "how to", "procedure", "walk me through"
        ]
        
        if any(pattern in question_lower for pattern in follow_up_patterns):
            # Check if we have recent context
            if self.current_context.get('last_topic'):
                return "follow_up", self.current_context['last_topic']
        
        # Check for clarification requests
        clarification_patterns = [
            "what do you mean", "i don't understand", "can you explain", "clarify",
            "more details", "elaborate", "expand on"
        ]
        
        if any(pattern in question_lower for pattern in clarification_patterns):
            if self.current_context.get('last_topic'):
                return "clarification", self.current_context['last_topic']
        
        return None, None
    
    def _update_conversation_context(self, question, response, query_type, specific_item):
        """Update conversation context for future queries"""
        self.conversation_history.append({
            'question': question,
            'response': response,
            'query_type': query_type,
            'specific_item': specific_item,
            'timestamp': len(self.conversation_history)
        })
        
        # Keep only last 10 exchanges to avoid memory bloat
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        # Update current context
        if specific_item:
            self.current_context['last_topic'] = specific_item
        elif query_type == "concept":
            # Extract topic from the question
            words = question.lower().split()
            if "what" in words and "is" in words:
                topic_index = words.index("is") + 1
                if topic_index < len(words):
                    self.current_context['last_topic'] = words[topic_index]
        
        # Store the last response type for context
        self.current_context['last_response_type'] = query_type
    
    def _is_ambiguous_or_casual(self, question):
        """Check if query is ambiguous or casual"""
        question_lower = question.lower()
        
        # Very short or unclear queries
        if len(question.strip()) < 3:
            return True
        
        # Testing queries (but not statistical tests)
        test_patterns = ["testing", "123", "hello", "hi", "hey", "what's up"]
        if any(pattern in question_lower for pattern in test_patterns):
            return True
        
        # Don't classify statistical tests as casual
        if "test" in question_lower and any(stat_term in question_lower for stat_term in ["t-test", "anova", "regression", "correlation", "chi-square", "factor analysis", "descriptive", "inferential"]):
            return False
        
        # Check if it's a statistical concept question (these are NOT casual)
        statistical_concepts = ["t-test", "anova", "regression", "correlation", "chi-square", "factor analysis", "descriptive", "inferential"]
        if any(concept in question_lower for concept in statistical_concepts):
            return False
        
        # Only classify as unclear if it's very vague AND short
        unclear_patterns = ["what is", "explain", "tell me about", "how does"]
        if any(pattern in question_lower for pattern in unclear_patterns) and len(question.split()) < 4:
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
        """Get appropriate response based on query type and conversation context"""
        # First check conversation context
        context_type, context_topic = self._analyze_conversation_context(question)
        
        if context_type == "follow_up":
            # Handle follow-up requests based on context
            return self._handle_follow_up(question, context_topic)
        elif context_type == "clarification":
            # Handle clarification requests based on context
            return self._handle_clarification(question, context_topic)
        
        # If no context match, analyze as normal query
        query_type, specific_item = self.analyze_query(question)
        
        # Get the response
        if query_type == "casual":
            response = self._get_casual_response(question)
        elif query_type == "steps":
            response = self._get_steps_response(question)
        elif query_type == "concept":
            response = self._get_concept_response(question)
        elif query_type == "function_specific":
            response = self._get_function_response(question, specific_item)
        elif query_type == "workflow":
            response = self._get_workflow_response(question)
        elif query_type == "category_overview":
            response = self._get_category_response(question, specific_item)
        else:
            response = self._get_context_aware_response(question)
        
        # Update conversation context
        self._update_conversation_context(question, response, query_type, specific_item)
        
        return response
    
    def _handle_follow_up(self, question, context_topic):
        """Handle follow-up requests based on conversation context"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ["yes", "yeah", "sure", "okay", "ok"]):
            # User is agreeing to get more details
            if self.current_context.get('last_response_type') == "concept":
                # They want step-by-step instructions for the concept we just explained
                return self._get_steps_response(f"step by step instructions for {context_topic}")
            else:
                return f"I'd be happy to help you with {context_topic}. What specifically would you like to know? Step-by-step instructions, examples, or something else?"
        
        elif any(word in question_lower for word in ["steps", "instructions", "how to", "procedure", "walk me through"]):
            # User explicitly wants steps
            return self._get_steps_response(f"step by step instructions for {context_topic}")
        
        elif any(word in question_lower for word in ["show me", "give me", "please"]):
            # User wants to see something
            if "step" in question_lower or "instruction" in question_lower:
                return self._get_steps_response(f"step by step instructions for {context_topic}")
            else:
                return f"Here's what I can show you about {context_topic}:\n\n" + self._get_context_aware_response(context_topic)
        
        # Default follow-up response
        return f"I'm here to help with {context_topic}. Would you like step-by-step instructions, examples, or do you have a different question?"
    
    def _handle_clarification(self, question, context_topic):
        """Handle clarification requests based on conversation context"""
        question_lower = question.lower()
        
        if "don't understand" in question_lower or "unclear" in question_lower:
            return f"Let me clarify {context_topic} in simpler terms:\n\n" + self._get_concept_response(f"What is {context_topic}?")
        
        elif "more details" in question_lower or "elaborate" in question_lower:
            return f"Here are more details about {context_topic}:\n\n" + self._get_context_aware_response(context_topic)
        
        elif "explain" in question_lower:
            return f"Let me explain {context_topic} in simpler terms:\n\n" + self._get_concept_response(f"Explain {context_topic}")
        
        # Default clarification response
        return f"I'd be happy to clarify {context_topic}. What specifically would you like me to explain better?"
    
    def get_conversation_context(self):
        """Get current conversation context for debugging"""
        return {
            'current_context': self.current_context,
            'conversation_history': self.conversation_history[-3:],  # Last 3 exchanges
            'total_exchanges': len(self.conversation_history)
        }
    
    def clear_conversation_context(self):
        """Clear conversation context (useful for starting fresh)"""
        self.conversation_history = []
        self.current_context = {}
        return "Conversation context cleared. Starting fresh!"
    
    def _get_casual_response(self, question):
        """Handle casual conversation and ambiguous queries"""
        if "test" in question.lower():
            return "I'm ready to assist with SPSS. What specific SPSS procedure or function would you like to test or learn about?"
        
        if len(question.strip()) < 5:
            return "I can help you with SPSS analysis. Please specify what you need assistance with:\n• Statistical procedures (t-tests, ANOVA, regression)\n• SPSS functions and commands\n• Data analysis workflows\n\nWhat would you like to know?"
        
        # Generic casual response
        chain = self.casual_prompt | self.model
        result = chain.invoke({"question": question})
        return result
    
    def _get_concept_response(self, question):
        """Explain concepts clearly and concisely"""
        # Search for relevant SPSS knowledge
        relevant_docs = self.vector_store.search_spss_function(question)
        
        if not relevant_docs:
            return "I'm not familiar with that specific SPSS concept. Could you clarify what you're asking about? For example: 'What is ANOVA?' or 'Explain factor analysis'"
        
        # Use the most relevant document
        doc = relevant_docs[0]
        
        # Extract the actual content without the metadata header
        content = doc.page_content
        
        # Remove the metadata header if it exists
        if "Function:" in content and "Category:" in content:
            # Find where the actual description starts
            lines = content.split('\n')
            description_start = 0
            for i, line in enumerate(lines):
                if line.strip() and not any(header in line for header in ['Function:', 'Category:', 'Difficulty:']):
                    description_start = i
                    break
            
            if description_start > 0:
                content = '\n'.join(lines[description_start:])
        
        # Clean up the content and provide a concise answer
        clean_content = content.strip()
        if len(clean_content) > 200:
            clean_content = clean_content[:200] + "..."
        
        return clean_content
    
    def _get_steps_response(self, question):
        """Get step-by-step instructions for any SPSS procedure"""
        # Search for relevant SPSS knowledge
        relevant_docs = self.vector_store.search_spss_function(question)
        
        if not relevant_docs:
            return "I can provide step-by-step SPSS instructions. Please specify what procedure you need help with. For example: 'How to run a t-test step by step' or 'Step-by-step instructions for creating a histogram'"
        
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
            return f"I can assist with SPSS functions. What specifically would you like to know about '{function_name}'? You can request step-by-step instructions or a general explanation."
        
        # Check if user wants steps
        if any(word in question.lower() for word in ["step", "how to", "procedure"]):
            return self._get_steps_response(question)
        
        # Give simple concept explanation
        content = function_details.page_content
        
        # Remove metadata headers
        if "Function:" in content and "Category:" in content:
            lines = content.split('\n')
            description_start = 0
            for i, line in enumerate(lines):
                if line.strip() and not any(header in line for header in ['Function:', 'Category:', 'Difficulty:']):
                    description_start = i
                    break
            
            if description_start > 0:
                content = '\n'.join(lines[description_start:])
        
        # Clean up and provide concise answer
        clean_content = content.strip()
        if len(clean_content) > 200:
            clean_content = clean_content[:200] + "..."
        
        return clean_content
    
    def _get_workflow_response(self, question):
        """Get workflow planning response"""
        workflow_suggestions = self.vector_store.get_workflow_suggestions(question)
        
        if not workflow_suggestions:
            return "I can help you plan SPSS workflows. What specific analysis are you trying to accomplish? For example: 'Workflow for survey analysis' or 'Process for regression analysis'"
        
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
            return f"I can help you explore SPSS categories. What would you like to know about '{category}' specifically?"
        
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
            return "I can assist with SPSS analysis. You can ask me about specific functions (like 't-test' or 'regression'), concepts (like 'What is ANOVA?'), step-by-step instructions, or workflow planning. What would you like to know?"
        
        # Get the most relevant document and provide a clean response
        doc = relevant_docs[0]
        content = doc.page_content
        
        # Remove metadata headers
        if "Function:" in content and "Category:" in content:
            lines = content.split('\n')
            description_start = 0
            for i, line in enumerate(lines):
                if line.strip() and not any(header in line for header in ['Function:', 'Category:', 'Difficulty:']):
                    description_start = i
                    break
            
            if description_start > 0:
                content = '\n'.join(lines[description_start:])
        
        # Clean up and provide concise answer
        clean_content = content.strip()
        if len(clean_content) > 200:
            clean_content = clean_content[:200] + "..."
        
        return clean_content
    
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
    
    print("Welcome to the SPSS Assistant.")
    print("Type 'help' for options, 'quit' to exit, or ask your question.")
    
    while True:
        try:
            question = input("\nWhat can I help you with today? \n").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye.")
                break
            
            elif question.lower() == 'help':
                print("\nAvailable SPSS Functions:")
                functions = bot.get_available_functions()
                for i, func in enumerate(functions, 1):
                    print(f"  {i:2d}. {func}")
                continue
            
            elif question.lower() == 'categories':
                print("\nAvailable SPSS Categories:")
                categories = bot.get_available_categories()
                for i, cat in enumerate(categories, 1):
                    print(f"  {i:2d}. {cat}")
                continue
            
            elif question.lower() == 'difficulty':
                print("\nFunctions by Difficulty Level:")
                for level in ['Beginner', 'Intermediate', 'Advanced']:
                    funcs = bot.get_functions_by_difficulty(level)
                    print(f"\n{level} Level:")
                    for func in funcs:
                        print(f"  • {func['function']}")
                continue
            
            elif question.lower() == 'metrics':
                print("\nEnhanced Vector Store Performance Metrics:")
                metrics = bot.get_performance_metrics()
                for key, value in metrics.items():
                    print(f"  {key}: {value}")
                continue
            
            elif question.lower() == 'insights':
                print("\nEnhanced Features Available:")
                print("  • dataset_recommendations - Get SPSS procedure recommendations")
                print("  • statistical_insights - Get statistical insights")
                print("  • Try asking: 'What SPSS procedures should I use for survey data?'")
                print("  • Try asking: 'Give me insights on regression analysis'")
                continue
            
            elif not question:
                # Show default help when user just hits enter
                help_response = bot.get_default_help()
                print(help_response)
                continue
            
            response = bot.get_response(question)
            print(response)
            
        except KeyboardInterrupt:
            print("\nGoodbye.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please try rephrasing your question or ask for 'help' to see available functions.")

if __name__ == "__main__":
    main() 