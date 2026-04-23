from ingest import ingest_pdf
from graph import build_graph

def run():
    print("🔄 Processing PDF...")
    ingest_pdf()

    app = build_graph()

    while True:
        query = input("\nAsk your question (or type 'exit'): ")

        if query.lower() == "exit":
            break

        result = app.invoke({"query": query})

        print("\n💬 Response:")
        print(result["response"])


if __name__ == "__main__":
    run()