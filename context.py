import chromadb
from sentence_transformers import SentenceTransformer

class VectorDB:
    def __init__(self, collection_name: str, similarity_threshold: float = 0.2):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Open-source embedding model
        self.similarity_threshold = similarity_threshold

        self._initialize_context()

    def _initialize_context(self):
        # populate knowledge base with sample problems and solutions
        problem_solution_pairs = {
            "Laptop won't turn on": "Check power adapter connection and battery charge.",
            "Laptop running slow": "Clear temporary files and disable startup programs.", 
            "Blue screen error (BSOD)": "Update device drivers and run Windows memory diagnostic.", 
            "Overheating laptop": "Clean air vents and use laptop cooling pad.", 
            "Battery draining quickly": "Adjust power settings and check battery health.", 
            "Wi-Fi connection issues": "Update network adapter drivers or reset network settings.", 
            "Software installation problems": "Ensure sufficient disk space and compatible system requirements.",
            "Word not opening": "Dance with one leg"
        }
        
        self.add_documents(problem_solution_pairs)

    def add_documents(self, problem_solution_dict: dict):
        problems = list(problem_solution_dict.keys())
        problem_embeddings = self.embedder.encode(problems).tolist()
        
        metadatas = [{"solution": solution} for solution in problem_solution_dict.values()]
        ids = [f"doc_{i}" for i in range(len(problems))]
        
        self.collection.add(
            embeddings=problem_embeddings, 
            documents=problems, 
            metadatas=metadatas, 
            ids=ids
        )

    def search(self, query: str, top_k: int = 3) -> list[str]:
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=top_k)
        
        # Filter results based on similarity threshold for problems
        if results['distances']:
            filtered_results = []
            for doc, distance, metadata in zip(results['documents'][0], results['distances'][0], results['metadatas'][0]):
                print(f"Distance: {(1 - distance)} | {doc}")  # Print the distances
                if (1 - distance) >= self.similarity_threshold:
                    filtered_results.append(f"{doc} : {metadata['solution']}")
            return filtered_results
        return []