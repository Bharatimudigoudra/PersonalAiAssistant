"""
Vector Store Service.
"""

from app.vectorstore.factory import VectorStoreFactory


class VectorStoreService:

    def __init__(self):

        self.store = VectorStoreFactory.create()

    def add(
        self,
        ids,
        documents,
        embeddings,
        metadatas,
    ):

        self.store.add(
            ids,
            documents,
            embeddings,
            metadatas,
        )

    def search(
        self,
        embedding,
        k=5,
    ):

        return self.store.search(
            embedding,
            k,
        )

    def health_check(self):

        return self.store.health_check()