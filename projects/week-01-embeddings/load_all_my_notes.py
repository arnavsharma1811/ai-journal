from sentence_transformers import SentenceTransformer
import chromadb

print("loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("connecting to database...")
client = chromadb.PersistentClient(path= "./my_gate_notes")
collection = client.create_collection("my_notes")

with open("my_os_notes.txt", "r" , encoding= "utf-8") as f:
    lines = f.readlines()

notes = [line.strip() for line in lines if line.strip()]
print( f"Found {len(notes)} notes. ")

batch_size = 20

for i in range(0, len(notes), batch_size):
    batch = notes[i: i+batch_size]
    ids = [f"my_note{i+j}" for j in range(len(batch))]
    embeddings = model.encode(batch).tolist()
    collection.add(
        ids = ids,
        embeddings = embeddings,
        documents = batch
    )
    print( f"Added batch {i//batch_size+1}")

    print(f"Done! Total notes in database: {collection.count()}")