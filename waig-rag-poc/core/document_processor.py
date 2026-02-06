"""
Document processing module with LlamaParse integration for PDF extraction
and page-level chunking with metadata enhancement.
"""

import os
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import asyncio
import logging

from llama_parse import LlamaParse
import tiktoken
from config import settings
from core.hierarchical_chunker import HierarchicalChunker, HierarchicalChunk
from core.enhanced_lexical_search import EnhancedLexicalSearch

# Setup logging
logging.basicConfig(level=getattr(logging, settings.log_level))
logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """Represents a chunk of document content with metadata."""
    content: str
    document_name: str
    page_number: int
    chunk_id: str
    section_title: Optional[str] = None
    subsection_title: Optional[str] = None
    page_title: Optional[str] = None
    document_type: str = "pdf"
    chunk_index: int = 0
    total_chunks: int = 0
    # New fields for hierarchical chunking
    chunk_type: str = "general"
    hierarchical_level: int = 0
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = None
    medical_keywords: List[str] = None
    cross_references: List[str] = None
    bbox: Optional[Dict] = None
    
    def __post_init__(self):
        if self.child_chunk_ids is None:
            self.child_chunk_ids = []
        if self.medical_keywords is None:
            self.medical_keywords = []
        if self.cross_references is None:
            self.cross_references = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)
    
    def get_enhanced_content(self) -> str:
        """Get content enhanced with metadata for better embedding."""
        metadata_prefix = []
        
        if self.document_name:
            metadata_prefix.append(f"Document: {self.document_name}")
        
        if self.page_title:
            metadata_prefix.append(f"Page Title: {self.page_title}")
            
        if self.section_title:
            metadata_prefix.append(f"Section: {self.section_title}")
            
        if self.subsection_title:
            metadata_prefix.append(f"Subsection: {self.subsection_title}")
            
        if self.page_number:
            metadata_prefix.append(f"Page: {self.page_number}")
        
        metadata_str = " | ".join(metadata_prefix)
        return f"{metadata_str}\n\n{self.content}" if metadata_prefix else self.content


class DocumentProcessor:
    """Processes documents using LlamaParse with advanced chunking and metadata extraction."""
    
    def __init__(self):
        """Initialize the document processor."""
        self.llama_parser = LlamaParse(
            api_key=settings.llama_cloud_api_key,
            result_type="markdown",  # Get structured markdown output
            parsing_instruction="""
            Extract content preserving document structure including:
            - Page titles and headers
            - Section and subsection headings
            - Table structures
            - Lists and bullet points
            - Maintain page boundaries
            """,
            max_timeout=60,
            verbose=True
        )
        self.encoding = tiktoken.get_encoding("cl100k_base")
        # Initialize hierarchical chunker and enhanced lexical search
        self.hierarchical_chunker = HierarchicalChunker()
        self.enhanced_lexical = EnhancedLexicalSearch()
    
    async def process_document(self, file_path: str) -> List[DocumentChunk]:
        """
        Process a document (PDF or markdown) and return chunked content with metadata.
        
        Args:
            file_path: Path to the document file (PDF or markdown)
            
        Returns:
            List of DocumentChunk objects
        """
        file_extension = Path(file_path).suffix.lower()
        logger.info(f"Processing {file_extension} document: {file_path}")
        
        try:
            if file_extension == '.pdf':
                # Parse PDF document with LlamaParse
                documents = await self._parse_with_llama(file_path)
                page_contents = self._extract_page_content(documents)
            elif file_extension in {'.md', '.markdown'}:
                # Parse markdown document directly
                page_contents = await self._parse_markdown(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            # Create chunks with enhanced metadata
            chunks = self._create_enhanced_chunks(page_contents, file_path)
            
            logger.info(f"Created {len(chunks)} chunks from {file_path}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {str(e)}")
            raise
    
    # Keep the old method name for backward compatibility
    async def process_pdf(self, file_path: str) -> List[DocumentChunk]:
        """
        Process a PDF file and return chunked content with metadata.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of DocumentChunk objects
        """
        return await self.process_document(file_path)
    
    async def process_json_pages(self, json_pages: List[Dict[str, Any]], document_name: str = "medical_manual") -> List[DocumentChunk]:
        """
        Process JSON pages using hierarchical chunking for medical documents.
        
        Args:
            json_pages: List of JSON page objects
            document_name: Name of the document
            
        Returns:
            List of DocumentChunk objects with hierarchical structure
        """
        logger.info(f"Processing {len(json_pages)} JSON pages with hierarchical chunking")
        
        try:
            # Create hierarchical chunks
            hierarchical_chunks = self.hierarchical_chunker.chunk_document(json_pages)
            
            # Convert to DocumentChunk format
            document_chunks = []
            for chunk in hierarchical_chunks:
                doc_chunk = DocumentChunk(
                    content=chunk.content,
                    document_name=document_name,
                    page_number=chunk.page_number,
                    chunk_id=chunk.chunk_id,
                    section_title=chunk.section_title,
                    subsection_title=chunk.subsection_title,
                    page_title=None,  # Will be extracted from page data if needed
                    document_type="json",
                    chunk_index=len(document_chunks),
                    total_chunks=len(hierarchical_chunks),
                    # Hierarchical fields
                    chunk_type=chunk.chunk_type,
                    hierarchical_level=chunk.level.value,
                    parent_chunk_id=chunk.parent_chunk_id,
                    child_chunk_ids=chunk.child_chunk_ids,
                    medical_keywords=chunk.keywords,
                    cross_references=chunk.cross_references,
                    bbox=chunk.bbox
                )
                document_chunks.append(doc_chunk)
            
            logger.info(f"Created {len(document_chunks)} hierarchical chunks from {len(json_pages)} pages")
            return document_chunks
            
        except Exception as e:
            logger.error(f"Error processing JSON pages: {str(e)}")
            raise
    
    async def _parse_with_llama(self, file_path: str) -> List[Any]:
        """Parse PDF document using LlamaParse."""
        try:
            # LlamaParse expects file upload
            documents = await asyncio.to_thread(
                self.llama_parser.load_data, file_path
            )
            return documents
        except Exception as e:
            logger.error(f"LlamaParse error for {file_path}: {str(e)}")
            raise
    
    async def _parse_markdown(self, file_path: str) -> List[Dict[str, Any]]:
        """Parse markdown document directly without LlamaParse."""
        try:
            # Read markdown file content
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Create a mock document structure similar to LlamaParse output
            page_contents = [{
                'content': content,
                'page_number': 1,  # Markdown files are treated as single page
                'page_title': self._extract_page_title(content),
                'sections': self._extract_sections(content),
                'raw_metadata': {
                    'file_name': Path(file_path).name,
                    'file_type': 'markdown'
                }
            }]
            
            return page_contents
            
        except Exception as e:
            logger.error(f"Error parsing markdown file {file_path}: {str(e)}")
            raise
    
    def _extract_page_content(self, documents: List[Any]) -> List[Dict[str, Any]]:
        """
        Extract content and metadata from parsed documents.
        
        Args:
            documents: Parsed documents from LlamaParse
            
        Returns:
            List of page content dictionaries
        """
        page_contents = []
        
        for doc_idx, document in enumerate(documents):
            content = document.text
            metadata = getattr(document, 'metadata', {})
            
            # Split content by pages if available in metadata
            if 'page_label' in metadata:
                page_number = self._extract_page_number(metadata['page_label'])
            else:
                page_number = doc_idx + 1
            
            # Extract structural elements
            page_title = self._extract_page_title(content)
            sections = self._extract_sections(content)
            
            page_contents.append({
                'content': content,
                'page_number': page_number,
                'page_title': page_title,
                'sections': sections,
                'raw_metadata': metadata
            })
        
        return page_contents
    
    def _extract_page_title(self, content: str) -> Optional[str]:
        """Extract page title from content using heuristics."""
        lines = content.strip().split('\n')
        
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line and not line.startswith('#'):
                # Look for title-like patterns
                if len(line) > 10 and len(line) < 100:
                    # Check if it looks like a title (capitalized, not too long)
                    if line[0].isupper() and '.' not in line[-10:]:
                        return line
            elif line.startswith('# '):
                return line[2:].strip()
        
        return None
    
    def _extract_sections(self, content: str) -> List[Dict[str, str]]:
        """Extract section hierarchies from markdown content."""
        sections = []
        lines = content.split('\n')
        
        current_h1 = None
        current_h2 = None
        current_h3 = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('# '):
                current_h1 = line[2:].strip()
                current_h2 = None
                current_h3 = None
                sections.append({
                    'level': 1,
                    'title': current_h1,
                    'full_path': current_h1
                })
            elif line.startswith('## '):
                current_h2 = line[3:].strip()
                current_h3 = None
                full_path = f"{current_h1} > {current_h2}" if current_h1 else current_h2
                sections.append({
                    'level': 2,
                    'title': current_h2,
                    'full_path': full_path
                })
            elif line.startswith('### '):
                current_h3 = line[4:].strip()
                path_parts = [part for part in [current_h1, current_h2, current_h3] if part]
                full_path = " > ".join(path_parts)
                sections.append({
                    'level': 3,
                    'title': current_h3,
                    'full_path': full_path
                })
        
        return sections
    
    def _extract_page_number(self, page_label: str) -> int:
        """Extract numeric page number from page label."""
        numbers = re.findall(r'\d+', str(page_label))
        return int(numbers[0]) if numbers else 1
    
    def _create_enhanced_chunks(self, page_contents: List[Dict[str, Any]], file_path: str) -> List[DocumentChunk]:
        """
        Create enhanced chunks with metadata augmentation.
        
        Args:
            page_contents: Extracted page content with metadata
            file_path: Original file path
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        document_name = Path(file_path).stem
        
        for page_data in page_contents:
            content = page_data['content']
            page_number = page_data['page_number']
            page_title = page_data['page_title']
            sections = page_data['sections']
            
            # Split page content into chunks if it's too long
            page_chunks = self._split_content_intelligently(
                content, 
                sections,
                settings.max_chunk_size,
                settings.chunk_overlap
            )
            
            for chunk_idx, chunk_data in enumerate(page_chunks):
                chunk_id = self._generate_chunk_id(
                    document_name, 
                    page_number, 
                    chunk_idx
                )
                
                chunk = DocumentChunk(
                    content=chunk_data['content'],
                    document_name=document_name,
                    page_number=page_number,
                    chunk_id=chunk_id,
                    section_title=chunk_data.get('section_title'),
                    subsection_title=chunk_data.get('subsection_title'),
                    page_title=page_title,
                    chunk_index=chunk_idx,
                    total_chunks=len(page_chunks)
                )
                
                chunks.append(chunk)
        
        return chunks
    
    def _split_content_intelligently(
        self, 
        content: str, 
        sections: List[Dict[str, str]], 
        max_chunk_size: int,
        overlap: int
    ) -> List[Dict[str, Any]]:
        """
        Split content intelligently based on sections and token limits.
        
        Args:
            content: Page content to split
            sections: Section metadata
            max_chunk_size: Maximum tokens per chunk
            overlap: Token overlap between chunks
            
        Returns:
            List of chunk data dictionaries
        """
        chunks = []
        
        # Calculate content length in tokens
        content_tokens = len(self.encoding.encode(content))
        
        if content_tokens <= max_chunk_size:
            # Single chunk
            section_info = self._get_primary_section(sections)
            chunks.append({
                'content': content,
                'section_title': section_info.get('section_title'),
                'subsection_title': section_info.get('subsection_title')
            })
        else:
            # Multiple chunks needed
            chunks = self._create_overlapping_chunks(
                content, sections, max_chunk_size, overlap
            )
        
        return chunks
    
    def _create_overlapping_chunks(
        self, 
        content: str, 
        sections: List[Dict[str, str]], 
        max_chunk_size: int,
        overlap: int
    ) -> List[Dict[str, Any]]:
        """Create overlapping chunks with section awareness."""
        chunks = []
        sentences = self._split_into_sentences(content)
        
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(self.encoding.encode(sentence))
            
            if current_tokens + sentence_tokens > max_chunk_size and current_chunk:
                # Finalize current chunk
                chunk_content = ' '.join(current_chunk)
                section_info = self._get_section_for_content(chunk_content, sections)
                
                chunks.append({
                    'content': chunk_content,
                    'section_title': section_info.get('section_title'),
                    'subsection_title': section_info.get('subsection_title')
                })
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk, overlap)
                current_chunk = overlap_sentences + [sentence]
                current_tokens = sum(len(self.encoding.encode(s)) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            section_info = self._get_section_for_content(chunk_content, sections)
            
            chunks.append({
                'content': chunk_content,
                'section_title': section_info.get('section_title'),
                'subsection_title': section_info.get('subsection_title')
            })
        
        return chunks
    
    def _split_into_sentences(self, content: str) -> List[str]:
        """Split content into sentences for better chunking."""
        # Simple sentence splitting - can be enhanced with NLTK/spaCy
        sentences = re.split(r'(?<=[.!?])\s+', content)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_sentences(self, sentences: List[str], overlap_tokens: int) -> List[str]:
        """Get sentences for overlap based on token count."""
        overlap_sentences = []
        current_tokens = 0
        
        for sentence in reversed(sentences):
            sentence_tokens = len(self.encoding.encode(sentence))
            if current_tokens + sentence_tokens <= overlap_tokens:
                overlap_sentences.insert(0, sentence)
                current_tokens += sentence_tokens
            else:
                break
        
        return overlap_sentences
    
    def _get_primary_section(self, sections: List[Dict[str, str]]) -> Dict[str, Optional[str]]:
        """Get primary section information for content."""
        if not sections:
            return {'section_title': None, 'subsection_title': None}
        
        # Find the most specific section (highest level)
        highest_level_section = max(sections, key=lambda x: x['level'])
        
        # Extract section hierarchy
        path_parts = highest_level_section['full_path'].split(' > ')
        
        return {
            'section_title': path_parts[0] if len(path_parts) > 0 else None,
            'subsection_title': path_parts[-1] if len(path_parts) > 1 else None
        }
    
    def _get_section_for_content(self, content: str, sections: List[Dict[str, str]]) -> Dict[str, Optional[str]]:
        """Determine which section this content belongs to."""
        # Simple heuristic: find section titles mentioned in content
        content_lower = content.lower()
        
        for section in reversed(sections):  # Start with most specific
            if section['title'].lower() in content_lower:
                path_parts = section['full_path'].split(' > ')
                return {
                    'section_title': path_parts[0] if len(path_parts) > 0 else None,
                    'subsection_title': path_parts[-1] if len(path_parts) > 1 else None
                }
        
        return self._get_primary_section(sections)
    
    def _generate_chunk_id(self, document_name: str, page_number: int, chunk_index: int) -> str:
        """Generate unique chunk ID."""
        content = f"{document_name}_{page_number}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


# Utility functions for batch processing
async def process_document_batch(file_paths: List[str]) -> List[DocumentChunk]:
    """Process multiple documents in batch."""
    processor = DocumentProcessor()
    all_chunks = []
    
    for file_path in file_paths:
        try:
            chunks = await processor.process_document(file_path)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {str(e)}")
            continue
    
    return all_chunks


def validate_file_type(file_path: str) -> bool:
    """Validate if file type is supported."""
    supported_extensions = {'.pdf', '.md', '.markdown'}
    return Path(file_path).suffix.lower() in supported_extensions 