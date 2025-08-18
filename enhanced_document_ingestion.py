#!/usr/bin/env python3
"""
Enhanced Document Ingestion System for SPSS Documentation
Handles large PDFs with intelligent chunking and metadata extraction
"""

import os
import re
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

# Suppress verbose output
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_ENDPOINT"] = ""

# PDF Processing
try:
    import PyPDF2
    from PyPDF2 import PdfReader
except ImportError:
    print("PyPDF2 not found. Installing...")
    os.system("pip install PyPDF2")
    import PyPDF2
    from PyPDF2 import PdfReader

# Text Processing
import tiktoken
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Data Handling
import pandas as pd
import numpy as np

# LangChain
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Setup logging
logging.basicConfig(level=logging.WARNING)  # Only show warnings and errors
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class SPSSDocumentProcessor:
    """Processes SPSS documentation with intelligent chunking and metadata extraction"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.pdf_reader = None
        self.text_content = ""
        self.metadata = {}
        self.chunks = []
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def extract_pdf_content(self) -> str:
        """Extract text content from PDF with structure preservation"""
        logger.info(f"Extracting content from {self.pdf_path}")
        
        try:
            with open(self.pdf_path, 'rb') as file:
                self.pdf_reader = PdfReader(file)
                total_pages = len(self.pdf_reader.pages)
                logger.info(f"PDF has {total_pages} pages")
                
                extracted_text = []
                page_metadata = []
                
                for page_num, page in enumerate(self.pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text.strip():
                            # Clean up the extracted text
                            cleaned_text = self._clean_extracted_text(text)
                            if cleaned_text.strip():
                                extracted_text.append(cleaned_text)
                                
                                # Extract page metadata
                                page_info = {
                                    'page_number': page_num + 1,
                                    'text_length': len(cleaned_text),
                                    'has_content': bool(cleaned_text.strip())
                                }
                                page_metadata.append(page_info)
                                
                                if (page_num + 1) % 500 == 0:  # Less frequent updates
                                    print(f"📖 Processed page {page_num + 1}/{total_pages}")
                    
                    except Exception as e:
                        logger.warning(f"Error processing page {page_num + 1}: {e}")
                        continue
                
                self.text_content = "\n\n".join(extracted_text)
                self.metadata['total_pages'] = total_pages
                self.metadata['pages_processed'] = len(page_metadata)
                self.metadata['page_metadata'] = page_metadata
                
                logger.info(f"Successfully extracted {len(self.text_content)} characters")
                return self.text_content
                
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            raise
    
    def extract_spss_structure(self) -> Dict[str, Any]:
        """Extract SPSS-specific structure and metadata"""
        logger.info("Extracting SPSS document structure")
        
        structure = {
            'commands': [],
            'functions': [],
            'sections': [],
            'examples': [],
            'parameters': []
        }
        
        # Extract SPSS commands (usually in ALL CAPS)
        spss_commands = re.findall(r'\b[A-Z][A-Z0-9_]*\b', self.text_content)
        structure['commands'] = list(set(spss_commands))[:100]  # Top 100 unique commands
        
        # Extract function-like patterns
        function_patterns = re.findall(r'(\w+)\s*\([^)]*\)', self.text_content)
        structure['functions'] = list(set(function_patterns))[:100]
        
        # Extract section headers (usually followed by newlines)
        section_pattern = r'^([A-Z][A-Za-z\s]+)$'
        sections = re.findall(section_pattern, self.text_content, re.MULTILINE)
        structure['sections'] = [s.strip() for s in sections if len(s.strip()) > 3][:50]
        
        # Extract examples (usually contain "Example:" or similar)
        example_pattern = r'(?:Example|EXAMPLE|Example:)\s*[:.]?\s*([^\n]+)'
        examples = re.findall(example_pattern, self.text_content)
        structure['examples'] = examples[:50]
        
        # Extract parameters (usually in brackets or after slashes)
        param_pattern = r'[/\[]([^/\]]+)[/\]]'
        params = re.findall(param_pattern, self.text_content)
        structure['parameters'] = list(set(params))[:100]
        
        self.metadata['structure'] = structure
        logger.info(f"Extracted structure: {len(structure['commands'])} commands, {len(structure['sections'])} sections")
        
        return structure
    
    def create_intelligent_chunks(self) -> List[Document]:
        """Create intelligent chunks based on SPSS document structure"""
        logger.info("Creating intelligent chunks")
        
        # First, try to split by natural sections
        section_chunks = self._split_by_sections()
        
        if section_chunks:
            logger.info(f"Created {len(section_chunks)} section-based chunks")
            self.chunks = section_chunks
            return section_chunks
        
        # Fallback to text splitter
        logger.info("Using text splitter as fallback")
        text_chunks = self.text_splitter.split_text(self.text_content)
        
        chunks = []
        for i, chunk in enumerate(text_chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    'chunk_id': i,
                    'chunk_type': 'text_split',
                    'source': 'spss_documentation',
                    'length': len(chunk),
                    'has_spss_content': self._contains_spss_content(chunk)
                }
            )
            chunks.append(doc)
        
        self.chunks = chunks
        logger.info(f"Created {len(chunks)} text-based chunks")
        return chunks
    
    def _split_by_sections(self) -> List[Document]:
        """Split document by natural sections"""
        # Look for section headers (usually in caps, followed by content)
        section_pattern = r'([A-Z][A-Z\s]+)\n+([^A-Z\n]+(?:\n[^A-Z\n]+)*)'
        sections = re.findall(section_pattern, self.text_content)
        
        if len(sections) < 10:  # Not enough sections found
            return []
        
        chunks = []
        for i, (header, content) in enumerate(sections):
            if len(content.strip()) > 100:  # Only keep substantial sections
                doc = Document(
                    page_content=f"{header}\n\n{content.strip()}",
                    metadata={
                        'chunk_id': i,
                        'chunk_type': 'section',
                        'section_header': header.strip(),
                        'source': 'spss_documentation',
                        'length': len(content),
                        'has_spss_content': self._contains_spss_content(content)
                    }
                )
                chunks.append(doc)
        
        return chunks
    
    def _contains_spss_content(self, text: str) -> bool:
        """Check if text contains SPSS-specific content"""
        spss_indicators = [
            'SPSS', 'ANALYZE', 'COMPUTE', 'RECODE', 'FREQUENCIES',
            'DESCRIPTIVES', 'T-TEST', 'ANOVA', 'REGRESSION', 'CORRELATION'
        ]
        
        text_upper = text.upper()
        return any(indicator in text_upper for indicator in spss_indicators)
    
    def enhance_chunks_with_metadata(self) -> List[Document]:
        """Enhance chunks with additional metadata and context"""
        logger.info("Enhancing chunks with metadata")
        
        enhanced_chunks = []
        
        for chunk in self.chunks:
            # Extract additional metadata from chunk content
            enhanced_metadata = chunk.metadata.copy()
            
            # Detect SPSS commands in this chunk
            commands_in_chunk = re.findall(r'\b[A-Z][A-Z0-9_]*\b', chunk.page_content)
            enhanced_metadata['commands_found'] = list(set(commands_in_chunk))[:10]
            
            # Detect difficulty level
            difficulty = self._assess_difficulty(chunk.page_content)
            enhanced_metadata['difficulty'] = difficulty
            
            # Detect category
            category = self._categorize_content(chunk.page_content)
            enhanced_metadata['category'] = category
            
            # Create enhanced document
            enhanced_doc = Document(
                page_content=chunk.page_content,
                metadata=enhanced_metadata
            )
            enhanced_chunks.append(enhanced_doc)
        
        self.chunks = enhanced_chunks
        logger.info(f"Enhanced {len(enhanced_chunks)} chunks with metadata")
        return enhanced_chunks
    
    def _assess_difficulty(self, text: str) -> str:
        """Assess the difficulty level of content"""
        text_lower = text.lower()
        
        # Advanced indicators
        advanced_terms = ['factor analysis', 'multivariate', 'canonical', 'discriminant', 'logistic']
        if any(term in text_lower for term in advanced_terms):
            return 'Advanced'
        
        # Intermediate indicators
        intermediate_terms = ['t-test', 'anova', 'correlation', 'regression', 'chi-square']
        if any(term in text_lower for term in intermediate_terms):
            return 'Intermediate'
        
        return 'Beginner'
    
    def _categorize_content(self, text: str) -> str:
        """Categorize content based on SPSS functions"""
        text_lower = text.lower()
        
        categories = {
            'Data Management': ['import', 'export', 'recode', 'compute', 'select', 'filter'],
            'Descriptive Statistics': ['frequencies', 'descriptives', 'explore', 'crosstabs'],
            'Inferential Statistics': ['t-test', 'anova', 'chi-square', 'nonparametric'],
            'Regression Analysis': ['regression', 'linear', 'logistic', 'multinomial'],
            'Factor Analysis': ['factor', 'reliability', 'canonical', 'discriminant'],
            'Data Visualization': ['graph', 'chart', 'plot', 'histogram', 'scatter'],
            'Output Management': ['output', 'export', 'save', 'print', 'report']
        }
        
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return 'General'
    
    def save_chunks_to_json(self, output_path: str = "spss_chunks.json"):
        """Save processed chunks to JSON for inspection"""
        logger.info(f"Saving chunks to {output_path}")
        
        chunks_data = []
        for chunk in self.chunks:
            chunk_data = {
                'content': chunk.page_content[:500] + "..." if len(chunk.page_content) > 500 else chunk.page_content,
                'metadata': chunk.metadata
            }
            chunks_data.append(chunk_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(chunks_data)} chunks to {output_path}")
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing results"""
        return {
            'total_pages': self.metadata.get('total_pages', 0),
            'pages_processed': self.metadata.get('pages_processed', 0),
            'total_characters': len(self.text_content),
            'chunks_created': len(self.chunks),
            'structure_extracted': bool(self.metadata.get('structure')),
            'chunk_types': list(set(chunk.metadata.get('chunk_type', 'unknown') for chunk in self.chunks)),
            'difficulty_distribution': self._get_difficulty_distribution(),
            'category_distribution': self._get_category_distribution()
        }
    
    def _get_difficulty_distribution(self) -> Dict[str, int]:
        """Get distribution of difficulty levels"""
        distribution = {}
        for chunk in self.chunks:
            difficulty = chunk.metadata.get('difficulty', 'Unknown')
            distribution[difficulty] = distribution.get(difficulty, 0) + 1
        return distribution
    
    def _get_category_distribution(self) -> Dict[str, int]:
        """Get distribution of categories"""
        distribution = {}
        for chunk in self.chunks:
            category = chunk.metadata.get('category', 'Unknown')
            distribution[category] = distribution.get(category, 0) + 1
        return distribution
    
    def _clean_extracted_text(self, text):
        """Clean up extracted text from PDF"""
        if not text:
            return text
        
        # Remove common PDF artifacts
        cleaned = text
        
        # Remove garbled characters (common in PDF extraction)
        import re
        
        # Remove sequences of garbled characters
        cleaned = re.sub(r'[µÂlm{yykzn!µ|omtqtoµ!{Ão~!Ästms!|~ {ÅtytÁtoµ!Áso!Á~kzµq{~ykÁt{z!tµ!m{y|ÂÁon1!\so!noqkÂwÁ]', '', cleaned)
        
        # Remove other common garbled patterns
        cleaned = re.sub(r'[µÂÃÄÅÁ]', '', cleaned)
        cleaned = re.sub(r'[~{!|}]', '', cleaned)
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove lines that are mostly special characters
        lines = cleaned.split('\n')
        clean_lines = []
        for line in lines:
            # Keep lines that have mostly readable text
            if len(line.strip()) > 0 and not self._is_garbled_line(line):
                clean_lines.append(line)
        
        cleaned = '\n'.join(clean_lines)
        
        return cleaned.strip()
    
    def _is_garbled_line(self, line):
        """Check if a line is mostly garbled text"""
        if not line.strip():
            return True
        
        # Count readable vs garbled characters
        readable_chars = len(re.findall(r'[a-zA-Z0-9\s\.\,\;\:\!\?\(\)\[\]\-\+\=\*\/]', line))
        total_chars = len(line.strip())
        
        # If less than 50% is readable, consider it garbled
        return total_chars > 0 and (readable_chars / total_chars) < 0.5

def main():
    """Main processing function"""
    pdf_path = "IBM_SPSS_Statistics_Command_Syntax_Reference.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    print("🚀 Starting SPSS Document Processing")
    print("=" * 50)
    
    try:
        # Initialize processor
        processor = SPSSDocumentProcessor(pdf_path)
        
        # Extract PDF content
        print("📖 Extracting PDF content...")
        processor.extract_pdf_content()
        
        # Extract SPSS structure
        print("🔍 Extracting SPSS structure...")
        processor.extract_spss_structure()
        
        # Create intelligent chunks
        print("✂️ Creating intelligent chunks...")
        processor.create_intelligent_chunks()
        
        # Enhance chunks with metadata
        print("✨ Enhancing chunks with metadata...")
        processor.enhance_chunks_with_metadata()
        
        # Save chunks for inspection
        print("💾 Saving chunks to JSON...")
        processor.save_chunks_to_json()
        
        # Display summary
        print("\n📊 Processing Summary:")
        print("=" * 50)
        summary = processor.get_processing_summary()
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        print(f"\n✅ Successfully processed {summary['total_pages']} pages into {summary['chunks_created']} chunks!")
        print(f"📁 Chunks saved to: spss_chunks.json")
        print(f"🔧 Ready for vector database ingestion!")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 