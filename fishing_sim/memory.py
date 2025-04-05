from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.memory import VectorStoreRetrieverMemory
import numpy as np
from datetime import datetime
from .llm_config import LLMConfig

class PersonalMemory:
    def __init__(self, config: Any, llm_config: LLMConfig):
        self.embeddings = llm_config.embeddings
        self.memory = FAISS.from_texts([""], self.embeddings)
        self.config = config
        
    def add_memory(self, memory_text: str, metadata: Dict = None):
        """Add a new memory to the personal memory store"""
        doc = Document(page_content=memory_text, metadata=metadata or {})
        self.memory.add_documents([doc])
        
    def retrieve_memories(self, query: str, k: int = None) -> List[Document]:
        """Retrieve relevant memories"""
        k = k or self.config.personal_memory_size
        return self.memory.similarity_search(query, k=k)
    
    def get_all_memories(self) -> List[Document]:
        """Get all memories in the store"""
        return self.memory.docstore.search("")

class SocialMemory:
    def __init__(self, config: Any, llm_config: LLMConfig):
        self.embeddings = llm_config.embeddings
        self.memory = FAISS.from_texts([""], self.embeddings)
        self.config = config
        self.current_month = 0
        
    def add_norm(self, norm_text: str, importance: float, metadata: Dict = None):
        """Add a new social norm to the memory store with importance score"""
        base_metadata = {
            'importance': importance,
            'recency': 1.0,  # New norms start with full recency
            'month_added': self.current_month,
            'last_updated': self.current_month
        }
        if metadata:
            base_metadata.update(metadata)
        doc = Document(page_content=norm_text, metadata=base_metadata)
        self.memory.add_documents([doc])
        
    def update_recency_scores(self):
        """Update recency scores for all norms based on current month"""
        self.current_month += 1
        all_docs = self.memory.docstore.search("")
        for doc in all_docs:
            # Calculate recency score: 1.0 for current month, decreasing by 0.1 each month
            months_old = self.current_month - doc.metadata['last_updated']
            recency = max(0.1, 1.0 - (months_old * 0.1))
            doc.metadata['recency'] = recency
            doc.metadata['last_updated'] = self.current_month
        
    def retrieve_norms(self, query: str, k: int = None) -> List[Document]:
        """Retrieve relevant social norms with weighted scoring"""
        k = k or self.config.social_memory_size
        
        # Get initial similarity search results
        docs = self.memory.similarity_search(query, k=k*2)  # Get more results than needed
        
        # Calculate weighted scores
        scored_docs = []
        for doc in docs:
            # Combine similarity, importance, and recency
            importance = doc.metadata.get('importance', 0.5)
            recency = doc.metadata.get('recency', 0.5)
            # Weight the scores (adjust weights as needed)
            weighted_score = (importance * 0.4) + (recency * 0.6)
            scored_docs.append((doc, weighted_score))
        
        # Sort by weighted score and return top k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:k]]
    
    def get_all_norms(self) -> List[Document]:
        """Get all norms in the store"""
        return self.memory.docstore.search("")
    
    def inherit_to_next_generation(self) -> 'SocialMemory':
        """Create a new instance with the same memory store"""
        new_memory = SocialMemory(self.config, self.embeddings)
        new_memory.memory = self.memory
        new_memory.current_month = self.current_month
        return new_memory 