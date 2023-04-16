import logging, os
from haystack import Pipeline
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import TextConverter, PreProcessor, PromptNode, EmbeddingRetriever, FARMReader
from haystack.pipelines import ExtractiveQAPipeline
from haystack.utils import print_answers
from haystack.agents import Agent, Tool


# config
logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)
doc_dir = "data/silmarillion"
first_run = True # disable after first run to avoid document processing
api_key = None # add for using OpenAI agent


# Get the host where Elasticsearch is running, default to localhost
host = os.environ.get("ELASTICSEARCH_HOST", "localhost")

document_store = ElasticsearchDocumentStore(
    host=host,
    username="",
    password="",
    index="silmarillion"
)

retriever = EmbeddingRetriever(document_store=document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1", use_gpu=True)

if first_run:
    # read and store documents
    indexing_pipeline = Pipeline()
    text_converter = TextConverter()
    preprocessor = PreProcessor(
        clean_whitespace=True,
        clean_header_footer=True,
        clean_empty_lines=True,
        split_by="word",
        split_length=200,
        split_overlap=20,
        split_respect_sentence_boundary=True,
    )
    indexing_pipeline.add_node(component=text_converter, name="TextConverter", inputs=["File"])
    indexing_pipeline.add_node(component=preprocessor, name="PreProcessor", inputs=["TextConverter"])
    indexing_pipeline.add_node(component=document_store, name="DocumentStore", inputs=["PreProcessor"])

    files_to_index = [doc_dir + "/" + f for f in os.listdir(doc_dir)]
    indexing_pipeline.run_batch(file_paths=files_to_index)

    document_store.update_embeddings(retriever=retriever)


reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)
presidents_qa = ExtractiveQAPipeline(reader=reader, retriever=retriever)

while True:

    question = input("question: ")

    if api_key == None:
        result = presidents_qa.run(question)
        print_answers(result, "minimum")

    else:
        prompt_node = PromptNode(model_name_or_path="text-davinci-003", api_key=api_key, stop_words=["Observation:"])
        agent = Agent(prompt_node=prompt_node)

        search_tool = Tool(name="Silmarillion_QA",pipeline_or_node=presidents_qa,
                        description="useful for when you need to answer questions related to the Silmarillion.",
                        output_variable="answers"
                    )
        agent.add_tool(search_tool)

        result = agent.run(question)
        print(result["transcript"].split('---')[1])
