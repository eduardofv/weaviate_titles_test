import json
import numpy as np
import sys
import time

import sentence_transformers as st
import weaviate

import warnings
warnings.filterwarnings("ignore")


# Instantiate the client with the auth config
client = weaviate.Client(
    url="http://localhost:8080",  # Replace w/ your endpoint
    #auth_client_secret=weaviate.AuthApiKey(api_key="YOUR-WEAVIATE-API-KEY"),  # Replace w/ your Weaviate instance API key
)


# Class definition object. Weaviate's autoschema feature will infer properties when importing.
class_obj = {
    "class": "title",
    "vectorizer": "none",
}


# Add the class to the schema
if len(sys.argv) > 1 and sys.argv[1] == 'reload':
    client.schema.delete_all()
    client.schema.create_class(class_obj)

    #load data
    with open("titles.txt") as fin:
        lines = fin.readlines()
    titles = [line.strip()[1:-1] for line in lines]


    embed = np.loadtxt("embed.txt", delimiter=",")

    with client.batch as batch:
        batch.batch_size = 100
        for i, title in enumerate(titles):
            if i%1000 == 0:
                print(f"loading row {i}")
            properties = {
                    "title": title
            }

            client.batch.add_data_object(properties, 
                                         "title", vector=list(embed[i]))



model = st.SentenceTransformer("all-mpnet-base-v2")
while True:
    print("CT> ", end="")
    query = input()
    if query == "exit":
        break

    start_time = time.time()
    query_emb = model.encode(query)
    end_embedding_time = time.time()
    #print(query_emb)

    nearVector = {
            "vector": query_emb
    }
    result = client.query.get(
            "title", ["title"]
        ).with_near_vector(
            nearVector
        ).with_limit(10).with_additional(['certainty']).do()
    end_time = time.time()

    time_embedding = end_embedding_time - start_time
    time_searching = end_time - end_embedding_time
    total_time = end_time - start_time
    #print(json.dumps(result, indent=4))
    for r in result['data']['Get']['Title']:
        print(f"{r['_additional']['certainty']:.3f}\t{r['title']}")
    print(f"{time_embedding:.2f} embedding\t{time_searching:.2f} searching\t{total_time:.2f} total[sec]")
    print()
