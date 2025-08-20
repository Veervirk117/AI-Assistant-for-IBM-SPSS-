#!/usr/bin/env python3
"""
Enhanced SPSS Vector Store
Handles large-scale SPSS documentation with optimized performance
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

# Suppress verbose output
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""

# LangChain
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document

# Suppress httpx logging
import httpx
import logging
httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

# Vector Operations
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Setup logging
logging.basicConfig(level=logging.WARNING)  # Only show warnings and errors
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class EnhancedSPSSVectorStore:
    """Enhanced vector store for large-scale SPSS documentation"""
    
    def __init__(self, db_location="./enhanced_spss_db", chunks_file="spss_chunks.json"):
        self.db_location = db_location
        self.chunks_file = chunks_file
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large")
        self.vector_store = None
        self.retriever = None
        self.chunks = []
        self._initialize_store()
    
    def _initialize_store(self):
        """Initialize the enhanced vector store"""
        add_documents = not os.path.exists(self.db_location)
        
        # Create vector store
        self.vector_store = Chroma(
            collection_name="enhanced_spss_knowledge",
            persist_directory=self.db_location,
            embedding_function=self.embeddings
        )
        
        if add_documents:
            print("🚀 Initializing Enhanced SPSS Knowledge Base...")
            self._load_and_process_chunks()
            self._add_documents_to_store()
            print("✅ Enhanced knowledge base initialization completed!")
        
        # Create enhanced retriever
        self.retriever = EnhancedSPSSRetriever(
            vector_store=self.vector_store,
            embeddings=self.embeddings
        )
    
    def _load_and_process_chunks(self):
        """Load and process chunks from JSON file"""
        if not os.path.exists(self.chunks_file):
            raise FileNotFoundError(f"Chunks file not found: {self.chunks_file}")
        
        print(f"📖 Loading chunks from {self.chunks_file}...")
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        # Convert back to Document objects
        self.chunks = []
        for chunk_data in chunks_data:
            doc = Document(
                page_content=chunk_data['content'],
                metadata=chunk_data['metadata']
            )
            self.chunks.append(doc)
        
        print(f"📚 Loaded {len(self.chunks)} chunks")
    
    def _add_documents_to_store(self):
        """Add documents to the vector store with progress tracking"""
        print("🔧 Adding documents to vector store...")
        
        # Clean metadata to remove lists
        cleaned_chunks = []
        for chunk in self.chunks:
            cleaned_metadata = {}
            for key, value in chunk.metadata.items():
                if isinstance(value, list):
                    cleaned_metadata[key] = " | ".join(str(v) for v in value)
                else:
                    cleaned_metadata[key] = value
            
            cleaned_doc = Document(
                page_content=chunk.page_content,
                metadata=cleaned_metadata
            )
            cleaned_chunks.append(cleaned_doc)
        
        # Add in batches for better performance
        batch_size = 20
        total_batches = (len(cleaned_chunks) + batch_size - 1) // batch_size
        
        for i in range(0, len(cleaned_chunks), batch_size):
            batch = cleaned_chunks[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            try:
                self.vector_store.add_documents(documents=batch)
                if batch_num % 3 == 0:  # Show progress every 3 batches
                    print(f"✅ Added batch {batch_num}/{total_batches} ({len(batch)} documents)")
            except Exception as e:
                print(f"❌ Error adding batch {batch_num}: {e}")
                continue
        
        print(f"🎯 Successfully added {len(cleaned_chunks)} documents to vector store")
    
    def search_spss_function(self, query: str, category: str = None, difficulty: str = None, 
                           max_results: int = 10) -> List[Document]:
        """Enhanced search with filtering and ranking"""
        # Enhance query with SPSS context
        enhanced_query = f"SPSS statistical analysis: {query}"
        
        # Get relevant documents
        documents = self.retriever.invoke(enhanced_query)
        
        # Apply filters if specified
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
        
        # Limit results
        return documents[:max_results]
    
    def get_function_details(self, function_name: str) -> Optional[Document]:
        """Get detailed information about a specific SPSS function"""
        query = f"SPSS function {function_name} detailed information"
        documents = self.retriever.invoke(query)
        
        # Find the most relevant document for this specific function
        for doc in documents:
            if function_name.lower() in doc.metadata.get('function', '').lower():
                return doc
        
        return None
    
    def get_category_overview(self, category: str) -> List[Document]:
        """Get overview of all functions in a specific category"""
        query = f"SPSS {category} functions and procedures"
        documents = self.retriever.invoke(query)
        
        # Filter by category
        category_docs = [doc for doc in documents if doc.metadata.get('category') == category]
        return category_docs
    
    def get_difficulty_level_functions(self, difficulty: str) -> List[Document]:
        """Get all functions for a specific difficulty level"""
        query = f"SPSS {difficulty} level statistical procedures"
        documents = self.retriever.invoke(query)
        
        # Filter by difficulty
        difficulty_docs = [doc for doc in documents if doc.metadata.get('difficulty') == difficulty]
        return difficulty_docs
    
    def get_workflow_suggestions(self, user_goal: str) -> Dict[str, List[Document]]:
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
    
    def get_dataset_recommendations(self, dataset_description: str) -> List[Document]:
        """Get SPSS procedure recommendations based on dataset characteristics"""
        query = f"SPSS analysis recommendations for dataset: {dataset_description}"
        documents = self.retriever.invoke(query)
        
        # Rank by relevance to dataset characteristics
        ranked_docs = self._rank_by_dataset_relevance(documents, dataset_description)
        return ranked_docs[:10]  # Top 10 recommendations
    
    def _rank_by_dataset_relevance(self, documents: List[Document], dataset_description: str) -> List[Document]:
        """Rank documents by relevance to dataset characteristics"""
        # Simple keyword-based ranking
        dataset_keywords = dataset_description.lower().split()
        
        scored_docs = []
        for doc in documents:
            score = 0
            content_lower = doc.page_content.lower()
            
            # Score based on keyword matches
            for keyword in dataset_keywords:
                if keyword in content_lower:
                    score += 1
            
            # Bonus for SPSS-specific terms
            spss_terms = ['spss', 'analysis', 'statistical', 'data', 'variable']
            for term in spss_terms:
                if term in content_lower:
                    score += 0.5
            
            scored_docs.append((doc, score))
        
        # Sort by score (highest first)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs]
    
    def get_statistical_insights(self, query: str) -> Dict[str, Any]:
        """Get statistical insights and recommendations"""
        documents = self.retriever.invoke(query)
        
        insights = {
            'recommended_procedures': [],
            'statistical_concepts': [],
            'best_practices': [],
            'common_pitfalls': [],
            'related_functions': []
        }
        
        for doc in documents[:5]:  # Analyze top 5 most relevant
            content = doc.page_content.lower()
            
            # Extract insights based on content
            if any(term in content for term in ['procedure', 'method', 'technique']):
                insights['recommended_procedures'].append(doc.metadata.get('section_header', 'Unknown'))
            
            if any(term in content for term in ['concept', 'theory', 'principle']):
                insights['statistical_concepts'].append(doc.metadata.get('section_header', 'Unknown'))
            
            if any(term in content for term in ['best practice', 'recommendation', 'guideline']):
                insights['best_practices'].append(doc.metadata.get('section_header', 'Unknown'))
            
            if any(term in content for term in ['pitfall', 'error', 'mistake', 'caution']):
                insights['common_pitfalls'].append(doc.metadata.get('section_header', 'Unknown'))
        
        return insights
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the vector store"""
        return {
            'total_documents': len(self.chunks),
            'vector_store_size': len(os.listdir(self.db_location)) if os.path.exists(self.db_location) else 0,
            'categories_available': list(set(doc.metadata.get('category', 'Unknown') for doc in self.chunks)),
            'difficulty_levels': list(set(doc.metadata.get('difficulty', 'Unknown') for doc in self.chunks))
        }

class EnhancedSPSSRetriever(BaseRetriever):
    """Enhanced retriever with better search capabilities"""
    
    def __init__(self, vector_store: Chroma, embeddings: OllamaEmbeddings):
        super().__init__()
        self._vector_store = vector_store
        self._embeddings = embeddings
        self._base_retriever = vector_store.as_retriever(search_kwargs={"k": 20})
    
    def _get_relevant_documents(self, query: str, *, run: CallbackManagerForRetrieverRun) -> List[Document]:
        """Get relevant documents with enhanced search"""
        # Use base retriever
        documents = self._base_retriever.invoke(query)
        
        # Enhance with semantic similarity if needed
        if len(documents) > 0:
            enhanced_docs = self._enhance_with_semantic_search(query, documents)
            return enhanced_docs
        
        return documents
    
    def invoke(self, query: str, config: Optional[Dict] = None, **kwargs) -> List[Document]:
        """Invoke method for compatibility"""
        return self._get_relevant_documents(query, run=None)
    
    def _enhance_with_semantic_search(self, query: str, documents: List[Document]) -> List[Document]:
        """Enhance search results with semantic similarity"""
        try:
            # Get query embedding
            query_embedding = self._embeddings.embed_query(query)
            
            # Get document embeddings
            doc_embeddings = []
            for doc in documents:
                doc_embedding = self._embeddings.embed_query(doc.page_content[:1000])  # First 1000 chars
                doc_embeddings.append(doc_embedding)
            
            # Calculate similarities
            similarities = []
            for doc_embedding in doc_embeddings:
                similarity = cosine_similarity([query_embedding], [doc_embedding])[0][0]
                similarities.append(similarity)
            
            # Sort documents by similarity
            doc_sim_pairs = list(zip(documents, similarities))
            doc_sim_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Return sorted documents
            return [doc for doc, sim in doc_sim_pairs]
            
        except Exception as e:
            logger.warning(f"Semantic enhancement failed: {e}")
            return documents

# Global instance
enhanced_spss_vector_store = EnhancedSPSSVectorStore()

def main():
    """Test the enhanced vector store"""
    print("🧪 Testing Enhanced SPSS Vector Store")
    print("=" * 50)
    
    store = enhanced_spss_vector_store
    
    # Test basic functionality
    print("Testing search functionality...")
    results = store.search_spss_function("t-test")
    print(f"Found {len(results)} relevant documents for 't-test'")
    
    # Test performance metrics
    print("\nPerformance Metrics:")
    metrics = store.get_performance_metrics()
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Enhanced vector store is ready!")

if __name__ == "__main__":
    main() 