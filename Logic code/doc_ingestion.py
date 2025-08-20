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
    import fitz  # PyMuPDF
except ImportError:
    print("PyMuPDF not found. Installing...")
    os.system("pip install PyMuPDF")
    import fitz

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
    
    def __init__(self, pdf_path: str = None):
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
        """Extract text content from PDF with structure preservation using PyMuPDF"""
        logger.info(f"Extracting content from {self.pdf_path}")
        
        try:
            doc = fitz.open(self.pdf_path)
            total_pages = len(doc)
            logger.info(f"PDF has {total_pages} pages")
            
            extracted_text = []
            page_metadata = []
            
            for page_num in range(total_pages):
                try:
                    page = doc[page_num]
                    
                    # Method 1: Extract text in blocks (paragraphs)
                    blocks = page.get_text("blocks")
                    
                    # Method 2: Extract text with bounding boxes for better structure
                    dict_text = page.get_text("dict")
                    
                    # Combine both methods for best results
                    page_text = self._extract_structured_text(blocks, dict_text)
                    
                    if page_text and page_text.strip():
                        # Clean up the extracted text
                        cleaned_text = self._clean_extracted_text(page_text)
                        if cleaned_text.strip():
                            extracted_text.append(cleaned_text)
                            
                            # Extract page metadata
                            page_info = {
                                'page_number': page_num + 1,
                                'text_length': len(cleaned_text),
                                'has_content': bool(cleaned_text.strip()),
                                'blocks_count': len(blocks),
                                'lines_count': len(dict_text.get('lines', []))
                            }
                            page_metadata.append(page_info)
                            
                            if (page_num + 1) % 500 == 0:  # Less frequent updates
                                print(f"📖 Processed page {page_num + 1}/{total_pages}")
                
                except Exception as e:
                    logger.warning(f"Error processing page {page_num + 1}: {e}")
                    continue
            
            doc.close()
            
            self.text_content = "\n\n".join(extracted_text)
            self.metadata['total_pages'] = total_pages
            self.metadata['pages_processed'] = len(page_metadata)
            self.metadata['page_metadata'] = page_metadata
            
            logger.info(f"Successfully extracted {len(self.text_content)} characters")
            return self.text_content
            
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            raise
    
    def _extract_structured_text(self, blocks, dict_text):
        """Extract structured text using both blocks and dict methods"""
        text_parts = []
        
        # Method 1: Use blocks for paragraph structure
        for block in blocks:
            if block[6] == 0:  # Text block
                text = block[4].strip()
                if text:
                    # Clean up the text
                    cleaned_text = self._clean_pdf_text(text)
                    if cleaned_text:
                        text_parts.append(cleaned_text)
        
        # Method 2: Use dict method for line-based structure with bounding boxes
        lines = dict_text.get('lines', [])
        if lines:
            # Group lines into paragraphs based on line distances
            paragraphs = self._group_lines_into_paragraphs(lines)
            for para in paragraphs:
                if para.strip():
                    cleaned_para = self._clean_pdf_text(para)
                    if cleaned_para:
                        text_parts.append(cleaned_para)
        
        # Combine and deduplicate
        combined_text = "\n\n".join(text_parts)
        return combined_text
    
    def _clean_pdf_text(self, text):
        """Clean up PDF text by removing corrupted characters and fixing encoding"""
        if not text:
            return ""
        
        # Remove common PDF corruption patterns
        import re
        
        # Remove control characters and non-printable characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        # Fix common character substitutions
        char_fixes = {
            'Ç': 'C', 'ç': 'c',
            'Š': 'S', 'š': 's',
            'Ž': 'Z', 'ž': 'z',
            'À': 'A', 'à': 'a',
            'Á': 'A', 'á': 'a',
            'Â': 'A', 'â': 'a',
            'Ã': 'A', 'ã': 'a',
            'Ä': 'A', 'ä': 'a',
            'Å': 'A', 'å': 'a',
            'È': 'E', 'è': 'e',
            'É': 'E', 'é': 'e',
            'Ê': 'E', 'ê': 'e',
            'Ë': 'E', 'ë': 'e',
            'Ì': 'I', 'ì': 'i',
            'Í': 'I', 'í': 'i',
            'Î': 'I', 'î': 'i',
            'Ï': 'I', 'ï': 'i',
            'Ò': 'O', 'ò': 'o',
            'Ó': 'O', 'ó': 'o',
            'Ô': 'O', 'ô': 'o',
            'Õ': 'O', 'õ': 'o',
            'Ö': 'O', 'ö': 'o',
            'Ù': 'U', 'ù': 'u',
            'Ú': 'U', 'ú': 'u',
            'Û': 'U', 'û': 'u',
            'Ü': 'U', 'ü': 'u',
            'Ý': 'Y', 'ý': 'y',
            'Ÿ': 'Y', 'ÿ': 'y',
            'Ñ': 'N', 'ñ': 'n'
        }
        
        for corrupted, fixed in char_fixes.items():
            text = text.replace(corrupted, fixed)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Only return if we have meaningful content
        if len(text.strip()) < 3:
            return ""
        
        return text.strip()
    
    def _group_lines_into_paragraphs(self, lines):
        """Group lines into paragraphs based on line distances"""
        if not lines:
            return []
        
        paragraphs = []
        current_paragraph = []
        last_y = None
        
        for line in lines:
            if not line.get('spans'):
                continue
                
            # Get the first span's y-coordinate
            span = line['spans'][0]
            y = span['bbox'][1]  # y-coordinate
            
            if last_y is None:
                # First line
                current_paragraph.append(self._extract_line_text(line))
                last_y = y
            else:
                # Calculate distance between lines
                distance = y - last_y
                
                # If distance is small, it's part of the same paragraph
                # If distance is large, it's a new paragraph
                if distance < 20:  # Adjust this threshold as needed
                    current_paragraph.append(self._extract_line_text(line))
                else:
                    # Start new paragraph
                    if current_paragraph:
                        paragraphs.append(" ".join(current_paragraph))
                        current_paragraph = []
                    current_paragraph.append(self._extract_line_text(line))
                
                last_y = y
        
        # Add the last paragraph
        if current_paragraph:
            paragraphs.append(" ".join(current_paragraph))
        
        return paragraphs
    
    def _extract_line_text(self, line):
        """Extract text from a line"""
        text_parts = []
        for span in line.get('spans', []):
            text = span.get('text', '').strip()
            if text:
                text_parts.append(text)
        return " ".join(text_parts)
    
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
    
    def create_chunks_from_knowledge_base(self) -> List[Document]:
        """Create chunks from the SPSS knowledge base instead of PDF"""
        from spss_knowledge_base import create_spss_documents
        
        print("Creating chunks from SPSS knowledge base...")
        documents = create_spss_documents()
        
        chunks = []
        for doc in documents:
            # Split the document content into chunks
            doc_chunks = self.text_splitter.split_text(doc.page_content)
            
            for i, chunk_text in enumerate(doc_chunks):
                chunk = Document(
                    page_content=chunk_text,
                    metadata={
                        'chunk_id': len(chunks),
                        'chunk_type': 'knowledge_base',
                        'source': 'spss_knowledge_base',
                        'function': doc.metadata.get('function', 'Unknown'),
                        'category': doc.metadata.get('category', 'Unknown'),
                        'difficulty': doc.metadata.get('difficulty', 'Unknown'),
                        'length': len(chunk_text),
                        'has_spss_content': True,
                        'chunk_index': i,
                        'total_chunks': len(doc_chunks)
                    }
                )
                chunks.append(chunk)
        
        print(f"Created {len(chunks)} chunks from knowledge base")
        return chunks
    
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
        print("Extracting PDF content...")
        processor.extract_pdf_content()
        
        # Extract SPSS structure
        print("Extracting SPSS structure...")
        processor.extract_spss_structure()
        
        # Create intelligent chunks
        print("Creating intelligent chunks...")
        processor.create_intelligent_chunks()
        
        # Enhance chunks with metadata
        print("Enhancing chunks with metadata...")
        processor.enhance_chunks_with_metadata()
        
        # Save chunks for inspection
        print("Saving chunks to JSON...")
        processor.save_chunks_to_json()
        
        # Display summary
        print("\n Processing Summary:")
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