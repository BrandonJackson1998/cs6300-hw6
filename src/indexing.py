"""
Clean Indexing Pipeline for Fitness Exercise RAG System
"""

import os
# Set environment variables before importing chromadb
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
import logging

# Suppress ChromaDB logging
logging.getLogger("chromadb").setLevel(logging.ERROR)


class VectorIndexer:
    """Clean, concise indexer for fitness exercise data"""
    
    def __init__(self, data_dir: str = "data", persist_dir: str = "chroma_db"):
        self.data_dir = Path(data_dir)
        self.persist_dir = Path(persist_dir)
    
    def run_indexing_pipeline(self):
        """Main indexing pipeline with clean output"""
        print("🔧 Building vector database...")
        
        # Step 1: Load and process data
        print("  📊 Processing exercise datasets...")
        exercises_df, mega_gym_df = self._load_datasets()
        exercises_df, mega_gym_df = self._apply_name_normalization(exercises_df, mega_gym_df)
        overlapping_names = self._find_overlapping_names(exercises_df, mega_gym_df)
        
        # Step 2: Create documents
        print("  📝 Creating exercise documents...")
        documents, metadatas, ids = self._create_documents(exercises_df, mega_gym_df, overlapping_names)
        
        # Step 3: Setup ChromaDB
        print("  🗂️ Initializing vector database...")
        client, collection = self._setup_chromadb()
        
        # Step 4: Index documents
        print(f"  🚀 Indexing {len(documents)} documents...")
        self._index_documents(collection, documents, metadatas, ids)
        
        # Step 5: Verify
        total_count = collection.count()
        
        print(f"✅ Vector database ready!")
        print(f"   • {total_count} exercises indexed")
        print(f"   • Stored in: {self.persist_dir}")
        print(f"   • Ready for queries!")
        
        return client, collection
    
    def _load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load both exercise datasets"""
        exercises_df = pd.read_csv(self.data_dir / "exercises.csv")
        mega_gym_df = pd.read_csv(self.data_dir / "megaGymDataset.csv")
        return exercises_df, mega_gym_df
    
    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize exercise names for consistent matching"""
        if pd.isna(name) or not isinstance(name, str):
            return ""
        
        # Convert to lowercase, strip whitespace, replace hyphens, consolidate spaces
        normalized = name.lower().strip().replace('-', ' ')
        return ' '.join(normalized.split())
    
    def _apply_name_normalization(self, exercises_df: pd.DataFrame, mega_gym_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply name normalization to both datasets"""
        exercises_df = exercises_df.copy()
        mega_gym_df = mega_gym_df.copy()
        
        exercises_df['normalized_name'] = exercises_df['name'].apply(self._normalize_name)
        mega_gym_df['normalized_name'] = mega_gym_df['Title'].apply(self._normalize_name)
        
        return exercises_df, mega_gym_df
    
    def _find_overlapping_names(self, exercises_df: pd.DataFrame, mega_gym_df: pd.DataFrame) -> Set[str]:
        """Find overlapping exercise names between datasets"""
        exercises_names = set(exercises_df['normalized_name'].dropna())
        mega_gym_names = set(mega_gym_df['normalized_name'].dropna())
        return exercises_names.intersection(mega_gym_names)
    
    def _create_documents(self, exercises_df: pd.DataFrame, mega_gym_df: pd.DataFrame, overlapping_names: set) -> Tuple[List[str], List[Dict], List[str]]:
        """Create all documents with clean processing"""
        
        # Filter high-quality overlaps
        merged_docs, _, _ = self._create_merged_documents(exercises_df, mega_gym_df, overlapping_names)
        merged_names = set(overlap['normalized_name'] for overlap in merged_docs) if merged_docs else set()
        
        # Create single-source documents
        niharika_docs, niharika_meta, niharika_ids = self._create_niharika_documents(mega_gym_df, merged_names)
        omarxadel_docs, omarxadel_meta, omarxadel_ids = self._create_omarxadel_documents(exercises_df, merged_names)
        
        # Combine all documents
        all_documents = []
        all_metadatas = []
        all_ids = []
        
        # Process merged documents
        for i, overlap in enumerate(merged_docs):
            doc, meta = self._build_merged_document(overlap['omar_row'], overlap['niharika_row'])
            all_documents.append(doc)
            all_metadatas.append(self._clean_metadata(meta))
            all_ids.append(f"merged_{i}")
        
        # Add Niharika documents
        all_documents.extend(niharika_docs)
        all_metadatas.extend(niharika_meta)
        all_ids.extend(niharika_ids)
        
        # Add Omarxadel documents
        all_documents.extend(omarxadel_docs)
        all_metadatas.extend(omarxadel_meta)
        all_ids.extend(omarxadel_ids)
        
        return all_documents, all_metadatas, all_ids
    
    def _create_merged_documents(self, exercises_df: pd.DataFrame, mega_gym_df: pd.DataFrame, overlapping_names: set) -> Tuple[List[Dict], List[Dict], List[str]]:
        """Create high-quality merged documents"""
        high_quality_overlaps = []
        
        for normalized_name in overlapping_names:
            # Get rows from both datasets
            omar_rows = exercises_df[exercises_df['normalized_name'] == normalized_name]
            niharika_rows = mega_gym_df[mega_gym_df['normalized_name'] == normalized_name]
            
            if omar_rows.empty or niharika_rows.empty:
                continue
                
            omar_row = omar_rows.iloc[0]
            niharika_row = niharika_rows.iloc[0]
            
            # Quality checks
            if (pd.notna(niharika_row.get('Rating')) and 
                self._count_instructions(omar_row, exercises_df.columns) >= 3 and
                pd.notna(omar_row.get('equipment')) and pd.notna(niharika_row.get('Equipment'))):
                
                high_quality_overlaps.append({
                    'normalized_name': normalized_name,
                    'omar_row': omar_row,
                    'niharika_row': niharika_row
                })
        
        return high_quality_overlaps, [], []
    
    def _create_niharika_documents(self, mega_gym_df: pd.DataFrame, merged_names: set) -> Tuple[List[str], List[Dict], List[str]]:
        """Create Niharika-only documents"""
        niharika_only_df = mega_gym_df[~mega_gym_df['normalized_name'].isin(merged_names)]
        
        documents = []
        metadatas = []
        ids = []
        
        for i, (_, row) in enumerate(niharika_only_df.iterrows()):
            doc, meta = self._build_niharika_document(row)
            documents.append(doc)
            metadatas.append(self._clean_metadata(meta))
            ids.append(f"niharika_{i}")
        
        return documents, metadatas, ids
    
    def _create_omarxadel_documents(self, exercises_df: pd.DataFrame, merged_names: set) -> Tuple[List[str], List[Dict], List[str]]:
        """Create Omarxadel-only documents"""
        omarxadel_only_df = exercises_df[~exercises_df['normalized_name'].isin(merged_names)]
        
        documents = []
        metadatas = []
        ids = []
        
        for i, (_, row) in enumerate(omarxadel_only_df.iterrows()):
            doc, meta = self._build_omarxadel_document(row)
            documents.append(doc)
            metadatas.append(self._clean_metadata(meta))
            ids.append(f"omarxadel_{i}")
        
        return documents, metadatas, ids
    
    def _build_merged_document(self, omar_row, niharika_row) -> Tuple[str, Dict]:
        """Build a merged document combining both sources"""
        title = niharika_row['Title']
        level = niharika_row.get('Level', 'Not specified')
        exercise_type = niharika_row.get('Type', 'Strength')
        rating = niharika_row['Rating']
        equipment = niharika_row['Equipment']
        target = omar_row['target']
        body_part = niharika_row['BodyPart']
        desc = niharika_row.get('Desc', 'No description available')
        
        # Get secondary muscles
        secondary = []
        for i in range(6):
            col = f'secondaryMuscles/{i}'
            if col in omar_row and pd.notna(omar_row[col]):
                secondary.append(str(omar_row[col]))
        secondary_str = ', '.join(secondary) if secondary else 'None'
        
        # Get instructions
        instructions = []
        for i in range(11):
            col = f'instructions/{i}'
            if col in omar_row and pd.notna(omar_row[col]) and str(omar_row[col]).strip():
                instructions.append(str(omar_row[col]))
        instructions_text = '\\n'.join([f"{i+1}. {inst}" for i, inst in enumerate(instructions)])
        
        document = f"""{title} - {level} {exercise_type}
Rating: {rating}/10
Equipment: {equipment}
Target: {target} (Primary)
Secondary: {secondary_str}
Body Part: {body_part}

{desc}

Step-by-step Instructions:
{instructions_text}"""
        
        metadata = {
            'name': title,
            'source': 'merged',
            'has_rating': True,
            'has_instructions': True,
            'rating': float(rating),
            'body_part': body_part,
            'equipment': equipment,
            'target_muscle': target,
            'secondary_muscles': ', '.join(secondary) if secondary else 'None'
        }
        
        return document, metadata
    
    def _build_niharika_document(self, row) -> Tuple[str, Dict]:
        """Build a Niharika-only document"""
        title = row['Title']
        level = row.get('Level', 'Not specified')
        exercise_type = row.get('Type', 'Exercise')
        rating = row.get('Rating')
        equipment = row.get('Equipment', 'Not specified') if pd.notna(row.get('Equipment')) else 'Not specified'
        body_part = row.get('BodyPart', 'Not specified')
        desc = row.get('Desc', 'No description available')
        
        rating_str = f"{rating}/10" if pd.notna(rating) else "Not rated"
        has_rating = pd.notna(rating)
        
        document = f"""{title} - {level} {exercise_type}
Rating: {rating_str}
Equipment: {equipment}
Body Part: {body_part}

{desc}"""
        
        metadata = {
            'name': title,
            'source': 'niharika',
            'has_rating': has_rating,
            'has_instructions': False,
            'rating': float(rating) if has_rating else 0.0,
            'body_part': body_part,
            'equipment': equipment
        }
        
        return document, metadata
    
    def _build_omarxadel_document(self, row) -> Tuple[str, Dict]:
        """Build an Omarxadel-only document"""
        name = row['name']
        equipment = row.get('equipment', 'Not specified')
        body_part = row.get('bodyPart', 'Not specified')
        target = row.get('target', 'Not specified')
        
        # Get secondary muscles
        secondary = []
        for i in range(6):
            col = f'secondaryMuscles/{i}'
            if col in row and pd.notna(row[col]):
                secondary.append(str(row[col]))
        secondary_str = ', '.join(secondary) if secondary else 'None'
        
        # Get instructions
        instructions = []
        for i in range(11):
            col = f'instructions/{i}'
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                instructions.append(str(row[col]))
        instructions_text = '\\n'.join([f"{i+1}. {inst}" for i, inst in enumerate(instructions)])
        
        document = f"""{name} - Strength exercise
Equipment: {equipment}
Body Part: {body_part}
Target: {target} (Primary)
Secondary: {secondary_str}

Step-by-step Instructions:
{instructions_text}"""
        
        metadata = {
            'name': name,
            'source': 'omarxadel',
            'has_rating': False,
            'has_instructions': True,
            'body_part': body_part,
            'equipment': equipment,
            'target_muscle': target,
            'secondary_muscles': ', '.join(secondary) if secondary else 'None'
        }
        
        return document, metadata
    
    def _count_instructions(self, row, columns) -> int:
        """Count non-null instructions for a row"""
        instruction_cols = [col for col in columns if col.startswith('instructions')]
        return sum(1 for col in instruction_cols 
                  if col in row and pd.notna(row[col]) and str(row[col]).strip())
    
    def _clean_metadata(self, metadata: Dict) -> Dict:
        """Clean metadata for ChromaDB compatibility"""
        cleaned = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            else:
                cleaned[key] = str(value)
        return cleaned
    
    def _setup_chromadb(self) -> Tuple[chromadb.PersistentClient, chromadb.Collection]:
        """Setup ChromaDB with minimal logging"""
        client = chromadb.PersistentClient(path=str(self.persist_dir))
        
        # Delete existing collection
        try:
            client.delete_collection(name="fitness_exercises")
        except ValueError:
            pass
        
        # Create new collection with embeddings
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        collection = client.create_collection(
            name="fitness_exercises",
            embedding_function=embedding_fn
        )
        
        return client, collection
    
    def _index_documents(self, collection, documents: List[str], metadatas: List[Dict], ids: List[str]):
        """Index documents in batches with minimal logging"""
        batch_size = 1000
        
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            collection.add(
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )


def main():
    """Run the clean indexing pipeline"""
    indexer = VectorIndexer()
    client, collection = indexer.run_indexing_pipeline()
    return client, collection


if __name__ == "__main__":
    main()