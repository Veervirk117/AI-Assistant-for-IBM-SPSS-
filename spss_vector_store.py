from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from spss_knowledge_base import create_spss_documents, get_spss_categories, get_spss_functions
import os

class SPSSVectorStore:
    def __init__(self, db_location="./spss_knowledge_db"):
        self.db_location = db_location
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        self.vector_store = None
        self.retriever = None
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize the vector store with SPSS knowledge"""
        add_documents = not os.path.exists(self.db_location)
        
        self.vector_store = Chroma(
            collection_name="spss_knowledge",
            persist_directory=self.db_location,
            embedding_function=self.embeddings
        )
        
        if add_documents:
            print("Initializing SPSS knowledge base...")
            documents = create_spss_documents()
            print(f"Created {len(documents)} documents")
            
            # Add documents one by one to debug
            for i, doc in enumerate(documents):
                try:
                    self.vector_store.add_documents(documents=[doc])
                    print(f"Added document {i+1}: {doc.metadata.get('function')}")
                except Exception as e:
                    print(f"Error adding document {i+1}: {e}")
            
            print("SPSS knowledge base initialization completed")
        
        # Create retriever with enhanced search
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={
                "k": 8  # Retrieve more relevant documents
            }
        )
    
    def search_spss_function(self, query, category=None, difficulty=None):
        """Search for SPSS functions with optional filtering"""
        # Enhance query with SPSS-specific context
        enhanced_query = f"SPSS statistical analysis: {query}"
        
        # Get relevant documents
        documents = self.retriever.invoke(enhanced_query)
        
        # Filter by category and difficulty if specified
        if category or difficulty:
            filtered_docs = []
            for doc in documents:
                metadata = doc.metadata
                if category and metadata.get('category') != category:
                    continue
                if difficulty and metadata.get('difficulty') != difficulty:
                    continue
                filtered_docs.append(doc)
            documents = filtered_docs
        
        return documents
    
    def get_function_details(self, function_name):
        """Get detailed information about a specific SPSS function"""
        query = f"SPSS function {function_name} step by step instructions"
        documents = self.retriever.invoke(query)
        
        # Find the most relevant document for this specific function
        for doc in documents:
            if function_name.lower() in doc.metadata.get('function', '').lower():
                return doc
        
        return None
    
    def get_category_overview(self, category):
        """Get overview of all functions in a specific category"""
        query = f"SPSS {category} functions and procedures"
        documents = self.retriever.invoke(query)
        
        # Filter by category
        category_docs = [doc for doc in documents if doc.metadata.get('category') == category]
        return category_docs
    
    def get_difficulty_level_functions(self, difficulty):
        """Get all functions for a specific difficulty level"""
        query = f"SPSS {difficulty} level statistical procedures"
        documents = self.retriever.invoke(query)
        
        # Filter by difficulty
        difficulty_docs = [doc for doc in documents if doc.metadata.get('difficulty') == difficulty]
        return difficulty_docs
    
    def get_workflow_suggestions(self, user_goal):
        """Get workflow suggestions based on user's goal"""
        query = f"SPSS workflow steps to achieve: {user_goal}"
        documents = self.retriever.invoke(query)
        
        # Group by category for workflow planning
        workflow_steps = {}
        for doc in documents:
            category = doc.metadata.get('category')
            if category not in workflow_steps:
                workflow_steps[category] = []
            workflow_steps[category].append(doc)
        
        return workflow_steps
    
    def get_related_functions(self, function_name):
        """Get related functions that are commonly used together"""
        query = f"SPSS functions related to {function_name} statistical analysis"
        documents = self.retriever.invoke(query)
        
        # Return functions from different categories that might be related
        related = []
        for doc in documents:
            if doc.metadata.get('function') != function_name:
                related.append(doc)
        
        return related[:5]  # Return top 5 related functions

# Global instance
spss_vector_store = SPSSVectorStore() 