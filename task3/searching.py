import faiss
import argparse 
import numpy as np
from sentence_transformers import SentenceTransformer



parser = argparse.ArgumentParser(description='Create Faiss database')
parser.add_argument('--query', type=str, help='search query')
parser.add_argument('--k', type=str, help='k nearest neighbors')
parser.add_argument('--model_name', type=str, help='name of model used')
parser.add_argument('--index', type=str, help='path to index')
args = parser.parse_args()


def search(query, model, index, k=5):
    query_embedding = model.encode([query])
    _, indices = index.search(np.array(query_embedding), k)
    return indices[0]


if __name__ == "__main__":
    query = args.query
    model_name = args.model_name
    index_path = args.index
    k = int(args.k)


    index = faiss.read_index(index_path)

    model = SentenceTransformer(model_name)

    results = search(query, model, index, k=k)
 
    print(f"Результаты поиска для запроса '{query}':")
    for result in results:
        print(result)