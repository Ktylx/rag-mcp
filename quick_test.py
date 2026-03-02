#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from src.indexer.chroma_manager import get_chroma_manager
chroma = get_chroma_manager()
stats = chroma.get_stats()
print('Stats:', stats)
print('Testing find_relevant_docs...')
results = chroma.similarity_search('test', k=3)
print(f'Found {len(results)} docs')