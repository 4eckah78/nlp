import faiss
import argparse 
import numpy as np


parser = argparse.ArgumentParser(description='Create Faiss database')
parser.add_argument('--save_db_path', type=str, help='path where to save database')
parser.add_argument('--embeddings_path', type=str, help='filename of the embeddings file')
args = parser.parse_args()


if __name__ == "__main__":
    embeddings_path = args.embeddings_path
    db_path = args.save_db_path

    embeddings = np.load(embeddings_path)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings))

    faiss.write_index(index, db_path)