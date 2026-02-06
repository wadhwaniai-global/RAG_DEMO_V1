"""
Domain-agnostic enhanced lexical search for structured documents.
Provides sophisticated lexical matching with content awareness.
"""

import re
from typing import List, Dict, Any, Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)

class EnhancedLexicalSearch:
    """Domain-agnostic enhanced lexical search for structured documents."""
    
    def __init__(self, domain_patterns: Optional[Dict[str, str]] = None, 
                 domain_synonyms: Optional[Dict[str, List[str]]] = None,
                 term_weights: Optional[Dict[str, float]] = None):
        # Default domain-agnostic patterns
        self.content_patterns = domain_patterns or {
            'procedures': r'\b(?:steps|procedure|protocol|method|process|algorithm|workflow|guidelines|instructions)\b',
            'definitions': r'\b(?:definition|define|means|refers to|is defined as|concept|term|explanation)\b',
            'lists': r'\b(?:list|items|points|bullets|enumeration|checklist|requirements|criteria)\b',
            'notes': r'\b(?:note|important|warning|caution|remark|observation|comment)\b',
            'references': r'\b(?:see|refer to|check|reference|link|section|page|chapter|appendix)\b',
            'actions': r'\b(?:do|perform|execute|implement|apply|use|follow|complete)\b',
            'conditions': r'\b(?:if|when|where|unless|provided that|in case|depending on)\b'
        }
        
        # Default domain-agnostic synonyms
        self.content_synonyms = domain_synonyms or {
            'procedure': ['process', 'method', 'approach', 'technique', 'way'],
            'definition': ['meaning', 'explanation', 'description', 'concept'],
            'list': ['items', 'points', 'elements', 'components', 'parts'],
            'note': ['comment', 'remark', 'observation', 'annotation'],
            'reference': ['link', 'citation', 'source', 'mention'],
            'important': ['critical', 'essential', 'vital', 'crucial', 'key'],
            'steps': ['stages', 'phases', 'procedures', 'actions', 'tasks'],
            'guidelines': ['rules', 'standards', 'principles', 'directives']
        }
        
        # Default term importance weights
        self.term_weights = term_weights or {
            'procedure': 1.0,
            'definition': 0.9,
            'important': 0.9,
            'steps': 0.8,
            'guidelines': 0.8,
            'note': 0.7,
            'list': 0.6,
            'reference': 0.5
        }
    
    def enhance_query(self, query: str) -> Dict[str, Any]:
        """Enhance query with domain-agnostic lexical improvements."""
        enhanced_query = {
            'original': query,
            'expanded_terms': [],
            'content_categories': [],
            'synonym_expansions': [],
            'pattern_matches': {},
            'boosted_terms': [],
            'weighted_terms': {},
            'cross_references': []
        }
        
        query_lower = query.lower()
        
        # Extract content categories
        for category, pattern in self.content_patterns.items():
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            if matches:
                enhanced_query['content_categories'].append(category)
                enhanced_query['pattern_matches'][category] = matches
        
        # Generate synonym expansions
        for term, synonyms in self.content_synonyms.items():
            if term.lower() in query_lower:
                enhanced_query['synonym_expansions'].extend(synonyms)
                enhanced_query['expanded_terms'].extend(synonyms)
        
        # Weight important terms
        for term, weight in self.term_weights.items():
            if term.lower() in query_lower:
                enhanced_query['weighted_terms'][term] = weight
                enhanced_query['boosted_terms'].append(term)
        
        # Extract cross-references
        cross_ref_patterns = [
            r'(?:see|refer to|check)\s+(?:section|page|chapter)\s+(\d+[a-z]?)',
            r'(?:see|refer to|check)\s+(\d+\.\d+)',
            r'\(see\s+(\d+)\)',
            r'\[(\d+)\]'
        ]
        for pattern in cross_ref_patterns:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            enhanced_query['cross_references'].extend(matches)
        
        return enhanced_query
    
    def create_lexical_queries(self, enhanced_query: Dict[str, Any]) -> List[str]:
        """Create multiple lexical search queries."""
        queries = [enhanced_query['original']]
        
        # Add synonym expansions
        if enhanced_query['synonym_expansions']:
            synonym_query = enhanced_query['original']
            for synonym in enhanced_query['synonym_expansions'][:3]:  # Limit to 3 synonyms
                synonym_query += f" OR {synonym}"
            queries.append(synonym_query)
        
        # Add category-specific queries
        for category, terms in enhanced_query['pattern_matches'].items():
            if terms:
                category_query = " OR ".join(terms)
                queries.append(category_query)
        
        # Add boosted term queries
        if enhanced_query['boosted_terms']:
            boosted_query = " AND ".join(enhanced_query['boosted_terms'])
            queries.append(boosted_query)
        
        # Add cross-reference queries
        if enhanced_query['cross_references']:
            for ref in enhanced_query['cross_references']:
                queries.append(f"section {ref} OR page {ref}")
        
        return queries
    
    def calculate_lexical_score(self, content: str, query: str, enhanced_query: Dict[str, Any]) -> float:
        """Calculate enhanced lexical similarity score."""
        content_lower = content.lower()
        query_lower = query.lower()
        
        # Base lexical score
        base_score = self._calculate_base_lexical_score(content_lower, query_lower)
        
        # Content term boost
        content_boost = self._calculate_content_boost(content_lower, enhanced_query)
        
        # Synonym boost
        synonym_boost = self._calculate_synonym_boost(content_lower, enhanced_query)
        
        # Category boost
        category_boost = self._calculate_category_boost(content_lower, enhanced_query)
        
        # Weighted term boost
        weighted_boost = self._calculate_weighted_boost(content_lower, enhanced_query)
        
        # Cross-reference boost
        cross_ref_boost = self._calculate_cross_reference_boost(content_lower, enhanced_query)
        
        # Combine scores
        total_score = (base_score * 0.4 + 
                      content_boost * 0.2 + 
                      synonym_boost * 0.15 + 
                      category_boost * 0.15 + 
                      weighted_boost * 0.1)
        
        return min(1.0, total_score)  # Cap at 1.0
    
    def _calculate_base_lexical_score(self, content: str, query: str) -> float:
        """Calculate base lexical similarity score."""
        query_words = set(query.split())
        content_words = set(content.split())
        
        if not query_words:
            return 0.0
        
        # Exact matches
        exact_matches = len(query_words.intersection(content_words))
        exact_score = exact_matches / len(query_words)
        
        # Partial matches (substring)
        partial_score = 0.0
        for query_word in query_words:
            if any(query_word in content_word for content_word in content_words):
                partial_score += 0.5
        
        partial_score = partial_score / len(query_words) if query_words else 0.0
        
        return (exact_score * 0.7) + (partial_score * 0.3)
    
    def _calculate_content_boost(self, content: str, enhanced_query: Dict[str, Any]) -> float:
        """Calculate boost for content terms."""
        boost = 0.0
        
        # Boost for content categories
        for category, terms in enhanced_query['pattern_matches'].items():
            for term in terms:
                if term.lower() in content:
                    boost += 0.1
        
        return min(0.3, boost)  # Cap content boost at 0.3
    
    def _calculate_synonym_boost(self, content: str, enhanced_query: Dict[str, Any]) -> float:
        """Calculate boost for synonym matches."""
        boost = 0.0
        
        for synonym in enhanced_query['synonym_expansions']:
            if synonym.lower() in content:
                boost += 0.05
        
        return min(0.2, boost)  # Cap synonym boost at 0.2
    
    def _calculate_category_boost(self, content: str, enhanced_query: Dict[str, Any]) -> float:
        """Calculate boost for content category matches."""
        boost = 0.0
        
        # Boost for content patterns
        for pattern in self.content_patterns.values():
            matches = re.findall(pattern, content, re.IGNORECASE)
            boost += len(matches) * 0.02
        
        return min(0.2, boost)  # Cap category boost at 0.2
    
    def _calculate_weighted_boost(self, content: str, enhanced_query: Dict[str, Any]) -> float:
        """Calculate boost for weighted terms."""
        boost = 0.0
        
        for term, weight in enhanced_query['weighted_terms'].items():
            if term.lower() in content:
                boost += weight * 0.1
        
        return min(0.3, boost)  # Cap weighted boost at 0.3
    
    def _calculate_cross_reference_boost(self, content: str, enhanced_query: Dict[str, Any]) -> float:
        """Calculate boost for cross-reference matches."""
        boost = 0.0
        
        for ref in enhanced_query['cross_references']:
            if f"section {ref}" in content or f"page {ref}" in content:
                boost += 0.1
        
        return min(0.1, boost)  # Cap cross-reference boost at 0.1
    
    def extract_content_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract content entities from text."""
        entities = {}
        
        for category, pattern in self.content_patterns.items():
            matches = re.findall(pattern, content.lower(), re.IGNORECASE)
            entities[category] = list(set(matches))
        
        return entities
    
    def calculate_content_relevance(self, content: str, query: str) -> float:
        """Calculate content relevance score between content and query."""
        content_entities = self.extract_content_entities(content)
        query_entities = self.extract_content_entities(query)
        
        relevance_score = 0.0
        
        for category in content_entities:
            content_terms = set(content_entities[category])
            query_terms = set(query_entities[category])
            
            if query_terms:
                overlap = len(content_terms.intersection(query_terms))
                category_score = overlap / len(query_terms)
                relevance_score += category_score * 0.2  # Weight each category
        
        return min(1.0, relevance_score)
    
    def get_content_keywords(self, content: str) -> List[str]:
        """Extract content keywords from text."""
        keywords = []
        
        for category, pattern in self.content_patterns.items():
            matches = re.findall(pattern, content.lower(), re.IGNORECASE)
            keywords.extend(matches)
        
        return list(set(keywords))
    
    def create_content_query_expansions(self, query: str) -> List[str]:
        """Create content-specific query expansions."""
        expansions = []
        query_lower = query.lower()
        
        # Add synonyms
        for term, synonyms in self.content_synonyms.items():
            if term in query_lower:
                for synonym in synonyms:
                    expanded_query = query_lower.replace(term, synonym)
                    expansions.append(expanded_query)
        
        # Add related terms based on content patterns
        if 'procedure' in query_lower:
            expansions.extend([
                query_lower + ' steps',
                query_lower + ' guidelines',
                query_lower + ' instructions'
            ])
        
        if 'definition' in query_lower:
            expansions.extend([
                query_lower + ' explanation',
                query_lower + ' meaning',
                query_lower + ' concept'
            ])
        
        if 'important' in query_lower:
            expansions.extend([
                query_lower + ' note',
                query_lower + ' warning',
                query_lower + ' critical'
            ])
        
        return expansions[:5]  # Limit to 5 expansions
