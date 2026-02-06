"""
Domain-agnostic hierarchical chunker for structured documents with JSON page structure.
Creates multi-level chunks with content awareness and cross-references.
"""

import re
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ChunkLevel(Enum):
    DOCUMENT = 1      # Entire document overview
    SECTION = 2        # Major sections (H1, H2)
    SUBSECTION = 3    # Subsections (H3, H4)
    CONTENT_BLOCK = 4 # Individual content blocks/steps
    DETAIL = 5        # Specific details, lists, notes

@dataclass
class HierarchicalChunk:
    content: str
    chunk_id: str
    level: ChunkLevel
    parent_chunk_id: Optional[str] = None
    child_chunk_ids: List[str] = field(default_factory=list)
    page_number: int = 0
    section_title: Optional[str] = None
    subsection_title: Optional[str] = None
    item_type: Optional[str] = None  # "heading", "text", "list", "note"
    bbox: Optional[Dict] = None
    keywords: List[str] = field(default_factory=list)
    cross_references: List[str] = field(default_factory=list)
    chunk_type: str = "general"  # "procedure", "definition", "list", "note", etc.
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class HierarchicalChunker:
    """Domain-agnostic hierarchical chunker for structured documents."""
    
    def __init__(self, confidence_threshold: float = 0.7, max_chunk_size: int = 4000, min_chunk_size: int = 500, max_chunks_per_page: int = 40):
        self.confidence_threshold = confidence_threshold
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunks_per_page = max_chunks_per_page
        
        # Content type indicators (domain-agnostic with medical enhancements)
        self.content_indicators = {
            'procedure': [
                'steps', 'procedure', 'protocol', 'method', 'process',
                'algorithm', 'workflow', 'guidelines', 'instructions'
            ],
            'definition': [
                'definition', 'define', 'means', 'refers to', 'is defined as',
                'concept', 'term', 'explanation'
            ],
            'list': [
                'list', 'items', 'points', 'bullets', 'enumeration',
                'checklist', 'requirements', 'criteria'
            ],
            'note': [
                'note:', 'nb:', 'important:', 'warning:', 'caution:',
                'remark:', 'observation:', 'comment:'
            ],
            'reference': [
                'see', 'refer to', 'check', 'reference', 'link',
                'section', 'page', 'chapter', 'appendix'
            ],
            'medical': [
                'dosing', 'schedule', 'treatment', 'medication', 'regimen',
                'therapy', 'dose', 'dosage', 'administration', 'prescription',
                'drug', 'pharmaceutical', 'clinical', 'patient', 'diagnosis'
            ]
        }
        
        # Cross-reference patterns
        self.cross_ref_patterns = [
            r'(?:see|refer to|check)\s+(?:section|page|chapter)\s+(\d+[a-z]?)',
            r'(?:see|refer to|check)\s+(\d+\.\d+)',
            r'\(see\s+(\d+)\)',
            r'\[(\d+)\]'
        ]
    
    def chunk_document(self, pages: List[Dict[str, Any]]) -> List[HierarchicalChunk]:
        """Create hierarchical chunks from document pages based on schema analysis recommendations."""
        all_chunks = []
        
        # Process all pages (no confidence filtering)
        logger.info(f"Processing {len(pages)} pages with hierarchical chunking")
        
        # Level 1: Document-level chunks (overview)
        doc_chunks = self._create_document_level_chunks(pages)
        all_chunks.extend(doc_chunks)
        
        # Level 2: Section-level chunks (H1, H2)
        section_chunks = self._create_section_level_chunks(pages)
        all_chunks.extend(section_chunks)
        
        # Level 3: Subsection-level chunks (H3, H4)
        subsection_chunks = self._create_subsection_level_chunks(pages)
        all_chunks.extend(subsection_chunks)
        
        # Level 4: Content-block chunks (procedures, definitions, etc.)
        content_chunks = self._create_content_block_chunks(pages)
        all_chunks.extend(content_chunks)
        
        # Level 5: Detail-level chunks (notes, lists, etc.)
        detail_chunks = self._create_detail_level_chunks(pages)
        all_chunks.extend(detail_chunks)
        
        # Establish parent-child relationships
        self._establish_hierarchical_relationships(all_chunks)
        
        logger.info(f"Created {len(all_chunks)} hierarchical chunks across {len(pages)} pages")
        return all_chunks
    
    def _create_document_level_chunks(self, pages: List[Dict[str, Any]]) -> List[HierarchicalChunk]:
        """Create document-level overview chunks."""
        chunks = []
        
        # Extract major topics across all pages
        major_topics = self._extract_major_topics(pages)
        
        for topic in major_topics:
            chunk = HierarchicalChunk(
                content=topic['content'],
                chunk_id=f"doc_{topic['topic_id']}",
                level=ChunkLevel.DOCUMENT,
                page_number=topic['page_number'],
                section_title=topic['section_title'],
                keywords=topic['keywords'],
                cross_references=topic['cross_refs'],
                chunk_type=topic['chunk_type']
            )
            chunks.append(chunk)
        
        return chunks
    
    
    def _create_section_level_chunks(self, pages: List[Dict[str, Any]]) -> List[HierarchicalChunk]:
        """Create section-level chunks (H1, H2 headings)."""
        chunks = []
        
        for page in pages:
            page_num = page['page']
            items = page['items']
            
            # Group items by major sections
            current_section = None
            section_content = []
            
            for item in items:
                if item['type'] == 'heading' and item['lvl'] <= 2:
                    # Save previous section
                    if current_section and section_content:
                        chunk = self._create_section_chunk(
                            current_section, section_content, page_num
                        )
                        chunks.append(chunk)
                    
                    # Start new section
                    current_section = item
                    section_content = [item]
                else:
                    section_content.append(item)
            
            # Save final section
            if current_section and section_content:
                chunk = self._create_section_chunk(
                    current_section, section_content, page_num
                )
                chunks.append(chunk)
        
        return chunks
    
    def _create_subsection_level_chunks(self, pages: List[Dict[str, Any]]) -> List[HierarchicalChunk]:
        """Create subsection-level chunks (H3, H4 headings)."""
        chunks = []
        
        for page in pages:
            page_num = page['page']
            items = page['items']
            
            # Find H3/H4 headings and their content
            for i, item in enumerate(items):
                if item['type'] == 'heading' and item['lvl'] >= 3:
                    # Get content until next heading or end
                    subsection_content = [item]
                    for j in range(i + 1, len(items)):
                        next_item = items[j]
                        if next_item['type'] == 'heading':
                            break
                        subsection_content.append(next_item)
                    
                    chunk = self._create_subsection_chunk(
                        item, subsection_content, page_num
                    )
                    chunks.append(chunk)
        
        return chunks
    
    def _create_content_block_chunks(self, pages: List[Dict[str, Any]]) -> List[HierarchicalChunk]:
        """Create content-block chunks (procedures, definitions, etc.)."""
        chunks = []
        
        for page in pages:
            page_num = page['page']
            items = page['items']
            
            # Look for content indicators
            for i, item in enumerate(items):
                if self._is_content_item(item):
                    # Get the content block
                    content_items = [item]
                    
                    # Include related items (lists, notes)
                    for j in range(i + 1, len(items)):
                        next_item = items[j]
                        if self._is_content_related(next_item):
                            content_items.append(next_item)
                        else:
                            break
                    
                    chunk_result = self._create_content_block_chunk(
                        item, content_items, page_num
                    )
                    
                    # Handle both single chunks and sliding window chunks
                    if isinstance(chunk_result, list):
                        chunks.extend(chunk_result)
                    else:
                        chunks.append(chunk_result)
        
        return chunks
    
    def _create_detail_level_chunks(self, pages: List[Dict[str, Any]]) -> List[HierarchicalChunk]:
        """Create detail-level chunks (individual items, notes, lists)."""
        chunks = []
        
        for page in pages:
            page_num = page['page']
            items = page['items']
            page_chunks = []
            
            for item in items:
                if self._is_detail_item(item):
                    chunk = self._create_detail_chunk(item, page_num)
                    if chunk is not None:
                        page_chunks.append(chunk)
                        
                        # Limit chunks per page to prevent memory issues
                        if len(page_chunks) >= self.max_chunks_per_page:
                            logger.warning(f"Page {page_num} has too many chunks, limiting to {self.max_chunks_per_page}")
                            break
            
            chunks.extend(page_chunks)
        
        return chunks
    
    def _create_section_chunk(self, section_item: Dict, content_items: List[Dict], page_num: int) -> HierarchicalChunk:
        """Create a section-level chunk."""
        content = self._combine_item_content(content_items)
        keywords = self._extract_keywords(content)
        chunk_type = self._classify_chunk_type(content)
        cross_refs = self._extract_cross_references(content)
        
        section_title = section_item.get('value', '')
        return HierarchicalChunk(
            content=content,
            chunk_id=f"sec_{page_num}_{hash(section_title) % 10000}",
            level=ChunkLevel.SECTION,
            page_number=page_num,
            section_title=section_title,
            item_type="section",
            keywords=keywords,
            bbox=section_item.get('bBox'),
            chunk_type=chunk_type,
            cross_references=cross_refs,
            metadata=self._extract_page_metadata(section_item)
        )
    
    def _create_subsection_chunk(self, subsection_item: Dict, content_items: List[Dict], page_num: int) -> HierarchicalChunk:
        """Create a subsection-level chunk."""
        content = self._combine_item_content(content_items)
        keywords = self._extract_keywords(content)
        chunk_type = self._classify_chunk_type(content)
        cross_refs = self._extract_cross_references(content)
        
        subsection_title = subsection_item.get('value', '')
        return HierarchicalChunk(
            content=content,
            chunk_id=f"sub_{page_num}_{hash(subsection_title) % 10000}",
            level=ChunkLevel.SUBSECTION,
            page_number=page_num,
            section_title=subsection_title,
            item_type="subsection",
            keywords=keywords,
            bbox=subsection_item.get('bBox'),
            chunk_type=chunk_type,
            cross_references=cross_refs,
            metadata=self._extract_page_metadata(subsection_item)
        )
    
    def _create_content_block_chunk(self, content_item: Dict, content_items: List[Dict], page_num: int) -> HierarchicalChunk:
        """Create a content-block chunk with sliding window for long table content."""
        content = self._combine_item_content(content_items)
        keywords = self._extract_keywords(content)
        chunk_type = self._classify_chunk_type(content)
        cross_refs = self._extract_cross_references(content)
        
        content_title = content_item.get('value', '')
        
        # Check if this is a long table that needs sliding window chunking
        if chunk_type == "table_figure" and len(content) > 2000:
            # Use sliding window for long table content
            return self._create_sliding_window_chunks(content_item, content_items, page_num)
        
        return HierarchicalChunk(
            content=content,
            chunk_id=f"content_{page_num}_{hash(content_title) % 10000}",
            level=ChunkLevel.CONTENT_BLOCK,
            page_number=page_num,
            section_title=content_title,
            item_type="content_block",
            keywords=keywords,
            bbox=content_item.get('bBox'),
            chunk_type=chunk_type,
            cross_references=cross_refs,
            metadata=self._extract_page_metadata(content_item)
        )
    
    def _create_detail_chunk(self, item: Dict, page_num: int) -> HierarchicalChunk:
        """Create a detail-level chunk."""
        item_type = item.get('type', '')
        
        # Get content from appropriate field
        if item_type == 'table':
            content = item.get('md', '')
        else:
            content = item.get('value', '')
            
        if not content:
            # Skip items without content
            return None
            
        keywords = self._extract_keywords(content)
        chunk_type = self._classify_chunk_type(content)
        cross_refs = self._extract_cross_references(content)
        
        return HierarchicalChunk(
            content=content,
            chunk_id=f"det_{page_num}_{hash(content) % 10000}",
            level=ChunkLevel.DETAIL,
            page_number=page_num,
            item_type=item_type,
            keywords=keywords,
            bbox=item.get('bBox'),
            chunk_type=chunk_type,
            cross_references=cross_refs,
            metadata=self._extract_page_metadata(item)
        )
    
    def _is_content_item(self, item: Dict) -> bool:
        """Check if item should be processed as content (now processes all text and table items)."""
        item_type = item.get('type', '')
        if item_type not in ['text', 'table']:
            return False
        
        # For tables, check the 'md' field; for text, check the 'value' field
        if item_type == 'table':
            content = item.get('md', '')
        else:
            content = item.get('value', '')
            
        if not content or len(content.strip()) < 3:  # Skip very short content
            return False
            
        # Process all text and table content
        return True
    
    def _is_content_related(self, item: Dict) -> bool:
        """Check if item is related to content (now includes all non-heading text and table content)."""
        if item.get('type') in ['heading']:
            return False
        
        # For tables, check the 'md' field; for text, check the 'value' field
        if item.get('type') == 'table':
            content = item.get('md', '')
        else:
            content = item.get('value', '')
            
        if not content or len(content.strip()) < 3:  # Skip very short content
            return False
            
        # Include all non-heading text and table content
        return True
    
    def _is_detail_item(self, item: Dict) -> bool:
        """Check if item should be a detail chunk (now includes all short text items)."""
        if item.get('type') in ['heading']:
            return False
        
        # For tables, check the 'md' field; for text, check the 'value' field
        if item.get('type') == 'table':
            content = item.get('md', '')
        else:
            content = item.get('value', '')
            
        if not content or len(content.strip()) < 3:
            return False
            
        # Include all short text and table items as detail chunks
        return len(content) < 500
    
    def _combine_item_content(self, items: List[Dict]) -> str:
        """Combine multiple items into coherent content."""
        content_parts = []
        
        for item in items:
            item_type = item.get('type', '')
            
            if item_type == 'heading':
                item_value = item.get('value', '')
                if item_value:
                    content_parts.append(f"## {item_value}")
            elif item_type == 'text':
                item_value = item.get('value', '')
                if item_value:
                    content_parts.append(item_value)
            elif item_type == 'table':
                # Use markdown content for tables
                table_md = item.get('md', '')
                if table_md:
                    content_parts.append(table_md)
        
        return '\n\n'.join(content_parts)
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content (comprehensive extraction)."""
        keywords = []
        content_lower = content.lower()
        
        # Extract all meaningful words (3+ characters)
        words = content_lower.split()
        meaningful_words = [word.strip('.,!?;:()[]{}"\'') for word in words if len(word.strip('.,!?;:()[]{}"\'')) >= 3]
        
        # Add meaningful words as keywords
        keywords.extend(meaningful_words[:10])  # Limit to first 10 meaningful words
        
        # Extract cross-references
        keywords.extend(self._extract_cross_references(content))
        
        # Remove duplicates and return
        return list(set(keywords))
    
    def _classify_chunk_type(self, content: str) -> str:
        """Classify chunk type based on content (simplified classification)."""
        content_lower = content.lower()
        
        # Simple classification based on content characteristics
        if any(word in content_lower for word in ['table', 'figure', 'chart', 'diagram']):
            return "table_figure"
        elif any(word in content_lower for word in ['note:', 'nb:', 'important:', 'warning:']):
            return "note"
        elif any(word in content_lower for word in ['step', 'procedure', 'method']):
            return "procedure"
        elif len(content.split()) < 20:  # Short content
            return "detail"
        else:
            return "content"
    
    def _extract_cross_references(self, content: str) -> List[str]:
        """Extract cross-references from content."""
        cross_refs = []
        for pattern in self.cross_ref_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            cross_refs.extend(matches)
        return cross_refs
    
    def _extract_page_metadata(self, item: Dict) -> Dict[str, Any]:
        """Extract metadata from page item."""
        return {
            'item_type': item.get('type', 'unknown'),
            'bbox': item.get('bBox'),
            'has_bbox': 'bBox' in item,
            'content_length': len(item.get('value', '')),
            'markdown': item.get('md', '')
        }
    
    def _extract_major_topics(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract major topics from pages."""
        topics = []
        
        for page in pages:
            page_num = page['page']
            items = page['items']
            
            # Look for major headings
            for item in items:
                if item.get('type') == 'heading' and item.get('lvl', 0) <= 2:
                    item_value = item.get('value', '')
                    if item_value:
                        topics.append({
                            'topic_id': f"topic_{page_num}_{hash(item_value) % 1000}",
                            'content': item_value,
                            'page_number': page_num,
                            'section_title': item_value,
                            'keywords': self._extract_keywords(item_value),
                            'cross_refs': [],
                            'chunk_type': self._classify_chunk_type(item_value)
                        })
        
        return topics
    
    def _establish_hierarchical_relationships(self, chunks: List[HierarchicalChunk]):
        """Establish parent-child relationships between chunks."""
        # Group chunks by page and level
        chunks_by_page = {}
        for chunk in chunks:
            if chunk.page_number not in chunks_by_page:
                chunks_by_page[chunk.page_number] = {}
            if chunk.level not in chunks_by_page[chunk.page_number]:
                chunks_by_page[chunk.page_number][chunk.level] = []
            chunks_by_page[chunk.page_number][chunk.level].append(chunk)
        
        # Establish relationships
        for page_num, page_chunks in chunks_by_page.items():
            # Document -> Section -> Subsection -> Content Block -> Detail
            levels = [ChunkLevel.DOCUMENT, ChunkLevel.SECTION, ChunkLevel.SUBSECTION, 
                     ChunkLevel.CONTENT_BLOCK, ChunkLevel.DETAIL]
            
            for i in range(len(levels) - 1):
                parent_level = levels[i]
                child_level = levels[i + 1]
                
                if parent_level in page_chunks and child_level in page_chunks:
                    self._link_chunks(page_chunks[parent_level], page_chunks[child_level])
    
    def _link_chunks(self, parent_chunks: List[HierarchicalChunk], child_chunks: List[HierarchicalChunk]):
        """Link parent and child chunks based on content similarity."""
        for parent in parent_chunks:
            for child in child_chunks:
                if self._chunks_are_related(parent, child):
                    parent.child_chunk_ids.append(child.chunk_id)
                    child.parent_chunk_id = parent.chunk_id
    
    def _chunks_are_related(self, parent: HierarchicalChunk, child: HierarchicalChunk) -> bool:
        """Check if chunks are related based on content and keywords."""
        # Simple heuristic: check for keyword overlap
        parent_keywords = set(parent.keywords)
        child_keywords = set(child.keywords)
        
        overlap = len(parent_keywords.intersection(child_keywords))
        return overlap > 0 or self._content_similarity(parent.content, child.content) > 0.3
    
    def _content_similarity(self, content1: str, content2: str) -> float:
        """Calculate simple content similarity."""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _create_sliding_window_chunks(self, content_item: Dict, content_items: List[Dict], page_num: int) -> List[HierarchicalChunk]:
        """Create multiple chunks using sliding window for long table content."""
        content = self._combine_item_content(content_items)
        content_title = content_item.get('value', '')
        
        # Split content into lines for better table handling
        lines = content.split('\n')
        window_size = 20  # Number of lines per chunk
        overlap = 5  # Lines of overlap between chunks
        
        chunks = []
        for i in range(0, len(lines), window_size - overlap):
            window_lines = lines[i:i + window_size]
            window_content = '\n'.join(window_lines)
            
            if len(window_content.strip()) < 50:  # Skip very small windows
                continue
                
            keywords = self._extract_keywords(window_content)
            cross_refs = self._extract_cross_references(window_content)
            
            chunk = HierarchicalChunk(
                content=window_content,
                chunk_id=f"table_{page_num}_{hash(content_title) % 10000}_{i}",
                level=ChunkLevel.CONTENT_BLOCK,
                page_number=page_num,
                section_title=f"{content_title} (Part {i//window_size + 1})",
                item_type="table_window",
                keywords=keywords,
                bbox=content_item.get('bBox'),
                chunk_type="table_figure",
                cross_references=cross_refs,
                metadata=self._extract_page_metadata(content_item)
            )
            chunks.append(chunk)
        
        return chunks
