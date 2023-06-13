import time
from langchain.llms.base import LLM
from llama_index import (
    PromptHelper,
    SimpleDirectoryReader,
    GPTListIndex,
    LLMPredictor,
    ServiceContext,
)
from transformers import pipeline
import torch
from config import settings

max_token = settings.get("model.max_token")

prompt_helper = PromptHelper(
    max_input_size=1024,
    num_output=max_token,
    max_chunk_overlap=20,
)


def timeit():
    """
    a utility decoration to time running time
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            args = [str(arg) for arg in args]

            print(f"[{(end - start):.8f} seconds]: f({args}) -> {result}")
            return result

        return wrapper

    return decorator


class LocalOPT(LLM):
    """Custom OPT Model for training"""

    model_name = "facebook/opt-iml-1.3b"
    pipeline = pipeline(
        "text-generation",
        model=model_name,
        device="cuda:0",
        model_kwargs={"torch_dtype": torch.bfloat16},
    )

    def _call(self, prompt: str, stop=None) -> str:
        model_response = self.pipeline(prompt, max_new_tokens=max_token)[0]["generated_text"]
        return model_response[len(prompt) :]

    @property
    def _identifying_params(self):
        return {"name_of_model": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "custom"



def create_index():
    """
    Responsible for creating an index using llama_index
    """
    print("Creating index")
    # Wrapper around LLMChain from LangChain
    llm = LLMPredictor(llm=LocalOPT())
    # Create a service context container
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm, prompt_helper=prompt_helper
    )
    # Load custom data from the directory
    docs = SimpleDirectoryReader("news").load_data()
    gpt_index = GPTListIndex.from_documents(
        documents=docs, service_context=service_context
    )

    print("Done creating index")
    return gpt_index

def execute_query(index_param):
    """Executes query for index"""
    print("Executing query on index.")
    query_engine = index_param.as_query_engine()
    query_response = query_engine.query("Summerize Australia's coal exports in 2023.")
    print("Done executing query on index")
    print(query_response)

    return query_response


if __name__ == "__main__":
    index = create_index()
    response = execute_query(index)
    print(response)
