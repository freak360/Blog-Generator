import streamlit as st
from langchain.prompts import PromptTemplate
from langchain import HuggingFaceHub
from langchain.chains import LLMChain
import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


# Write a function to get a response from llama 2 model
def getLLamaresponse(inputt_text, noo_words):
    ### LLama2 model
    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    model = HuggingFaceHub(huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"], repo_id=repo_id,
                           model_kwargs={"temperature":0.6, "max_new_tokens":250})
    ## Prompt Template
    template = """
        Write a blog for a topic {inputt_text}
        within {noo_words} words.
            """
    prompt = PromptTemplate(input_variables=['inputt_text', 'noo_words'],
                            template=template)

    ## Generate the ressponse from the LLama 2 model
    story_llm = LLMChain(llm= model, prompt=prompt, verbose=True)
    response = story_llm.predict(inputt_text=inputt_text, noo_words=noo_words)
    print(response)
    return response

def main():
    st.set_page_config(page_title="Generate Blogs",
                        page_icon='🧠',
                        layout='centered',
                        initial_sidebar_state='collapsed')

    st.header("Generate a Blog 🧠")

    input_text = st.text_input("Topic")
    no_words = st.text_input("Number of Words")

    submit = st.button('Generate')

    if submit:
        with st.expander("Response"):
            st.write(getLLamaresponse(input_text, no_words))

if __name__ == "__main__":
    main()