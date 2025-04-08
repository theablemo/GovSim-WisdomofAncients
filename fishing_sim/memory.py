from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.memory import VectorStoreRetrieverMemory
import numpy as np
from datetime import datetime
from .llm_config import LLMConfig
from collections import deque


class RunningMemory:
    def __init__(self, max_memories: int = 5):
        self.memories = deque(maxlen=max_memories)

    def add_memory(self, memory_text: str, current_month: int):
        """Add a new memory with timestamp to the running memory"""
        # timestamp = datetime.now().strftime("%Y-%m-%d")
        timestamp = f"Month #{current_month}"
        formatted_memory = f"{timestamp}: {memory_text}"
        self.memories.append(formatted_memory)

    def get_recent_memories(self, n: int = None) -> List[str]:
        """Get the n most recent memories"""
        n = n or len(self.memories)
        return list(self.memories)[-n:]


class InsightMemory:
    def __init__(self, config: Any, llm_config: LLMConfig):
        self.embeddings = llm_config.embeddings
        self.memory = None  # Initialize as None
        self.config = config

    def add_memory(self, memory_text: str, metadata: Dict = None):
        """Add a new memory to the personal memory store"""
        doc = Document(page_content=memory_text, metadata=metadata or {})
        if self.memory is None:
            self.memory = FAISS.from_texts([memory_text], self.embeddings)
        else:
            self.memory.add_documents([doc])

    def retrieve_memories(self, query: str, k: int = None) -> List[Document]:
        """Retrieve relevant memories"""
        if self.memory is None:
            return []
        k = k or self.config.personal_memory_size
        return self.memory.similarity_search(query, k=k)

    def get_all_memories(self) -> List[Document]:
        """Get all memories in the store"""
        if self.memory is None:
            return []
        return self.memory.docstore.search("")


class SocialMemory:
    def __init__(self, config: Any, llm_config: LLMConfig):
        self.embeddings = llm_config.embeddings
        self.memory = None  # Initialize as None
        self.config = config
        self.current_month = 0

    def add_norm(self, norm_text: str, importance: float, metadata: Dict = None):
        """Add a new social norm to the memory store with importance score"""
        base_metadata = {
            "importance": importance,
            "recency": 1.0,  # New norms start with full recency
            "month_added": self.current_month,
            "last_updated": self.current_month,
        }
        if metadata:
            base_metadata.update(metadata)
        doc = Document(page_content=norm_text, metadata=base_metadata)
        if self.memory is None:
            self.memory = FAISS.from_texts([norm_text], self.embeddings)
        else:
            self.memory.add_documents([doc])

    def update_recency_scores(self):
        """Update recency scores for all norms based on current month"""
        if self.memory is None:
            return
        self.current_month += 1
        # Get all documents from the FAISS store
        all_docs = self.memory.docstore._dict.values()
        for doc in all_docs:
            if isinstance(doc, Document):
                # Calculate recency score: 1.0 for current month, decreasing by 0.1 each month
                months_old = self.current_month - doc.metadata["last_updated"]
                recency = max(0.1, 1.0 - (months_old * 0.1))
                doc.metadata["recency"] = recency
                doc.metadata["last_updated"] = self.current_month

    def retrieve_norms(self, query: str, k: int = None) -> List[Document]:
        """Retrieve relevant social norms with weighted scoring"""
        if self.memory is None:
            return []
        k = k or self.config.social_memory_size

        # Get initial similarity search results
        docs = self.memory.similarity_search(query, k=k * 2)  # Get more results than needed

        # Calculate weighted scores
        scored_docs = []
        for doc in docs:
            # Combine similarity, importance, and recency
            importance = doc.metadata.get("importance", 0.5)
            recency = doc.metadata.get("recency", 0.5)
            # Weight the scores (adjust weights as needed)
            weighted_score = (importance * 0.4) + (recency * 0.6)
            scored_docs.append((doc, weighted_score))

        # Sort by weighted score and return top k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:k]]

    def get_all_norms(self) -> List[Document]:
        """Get all norms in the store"""
        if self.memory is None:
            return []
        return self.memory.docstore.search("")

    def inherit_to_next_generation(self) -> "SocialMemory":
        """Create a new instance with the same memory store"""
        new_memory = SocialMemory(self.config, self.embeddings)
        new_memory.memory = self.memory
        new_memory.current_month = self.current_month
        return new_memory
