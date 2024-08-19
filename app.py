"""Streamlit app that uses the Azure OpenAI service to generate product descriptions."""

import services.document_handler as dh
import services.agent as agent

from PIL import Image
import time
import math
import pandas as pd
import streamlit as st

from langchain.callbacks import get_openai_callback

def main():

#system context and guidelines are put here in order to allow the users to customize these within the UI.
    SYSTEM_CONTEXT = """You are a helpful content creator for a consumer electronics retailer. \
Your job is to generate product descriptions based on product information passed to you in a JSON format. \
The goal of the product description is to be informative for the end consumer, and ultimately persuade the consumer into buying the product."""

    GUIDELINES = """\
1. The overall sentiment should be positive and persuasive. 
2. ONLY use product information provided in the provided json input file. You are not allowed to add product information or specifications that are not in the provided input JSON.
3. Start with a short introductory text that explains the overall benefit of the product.
4. Extract key features of the product from fields "KEY_SELLING_POINTS" and "SHORTDESCRIPTION" in the provided JSON file. Each key feature should have its own headline focusing on the name of the feature and its primary functionality. 
5. Provide details about the key features below the title of the key feature. Focus on benefits of the feature instead of just listing feature specifications. 
6. Supplement the key features with technical attribute information from the input JSON, if applicable. This is optional.
7. Technical specifications about the product can be provided in list format at the bottom of the description.
8. If you are unable to find details about named key features, write [MORE INFORMATION REQUIRED] for that feature.
9. Do not use emojis.
10. Lastly, format the description in HTML. Exclude HTML and BODY tags. Use <b> instead of headers (h1, h2, h3, h4, h5)"""

    #these files include the examples that are used when "few-shot-prompting" is performed. 
    PDEX_CSV = "data/product_description_examples.csv"
    ADEX_CSV = "data/product_description_examples_attributes.csv"

    MODELS = {
        "gpt-4" : {
            "prompt_token_cost" : 0.32/1000,
            "completion_token_cost" : 0.639/1000
        },
        "gpt-35-turbo-16k" : {
            "prompt_token_cost" : 0.032/1000,
            "completion_token_cost" : 0.043/1000
        }
    }

    #logo = Image.open("img/logo.png")

    st.set_page_config(page_title="Product Description Generator")

    #st.image(logo, width=52)
    st.header("Product Description Generator")

    upload_data_tab, query_tab, param_tab = st.tabs(["Upload data", "Customize Prompt", "LLM Parameters"])
    with upload_data_tab:
        pdcsv = st.file_uploader("Upload product description file", type="csv")
        adcsv = st.file_uploader("Upload attribute data file", type="csv")

    with query_tab:
        use_examples = st.toggle("Include examples in prompt (\"few-shot-prompting\")", value=True)
        system_context = st.text_area("Customize the system context passed to the model:", SYSTEM_CONTEXT, height=200)
        query_guidelines = st.text_area("Customize the product description guidelines passed to the model:", GUIDELINES, height=400)

    with param_tab:
        modelSelect = st.radio(
            "Select GPT-model",
            list(MODELS.keys()),
            index=1,
            horizontal=True)
        st.write(" ")
        st.write("Adjust Model Parameters:")
        temperature = st.slider("Temperature", min_value=0.0,max_value=2.2,step=0.1, value=0.2)
        top_p = st.slider("Top P", min_value=0.0, max_value=1.0, step=0.1, value=0.3)
        frequency_penalty = st.slider("Frequency Penalty", min_value=-2.0, max_value=2.0, step=0.1, value=0.0)
        presence_penalty = st.slider("Presence Penalty", min_value=-2.0, max_value=2.0, step=0.1, value=0.0)

    generate_descriptions = st.button("Generate product descriptions", type="primary")

    st.divider()

    if generate_descriptions:
        timestart = time.time()
        dfpd = pd.read_csv(pdcsv)
        dfad = pd.read_csv(adcsv)
        dfpd_ex = pd.read_csv(PDEX_CSV, index_col=0)
        dfad_ex = pd.read_csv(ADEX_CSV, index_col=0)
        docs = dh.Document_Handler(dfpd, dfad, dfpd_ex, dfad_ex)
        input = docs.generate_input_file(use_examples)
        rows = len(dfpd)
        counter = 0

        model = agent.product_description_agent(system_context, query_guidelines, modelSelect, temperature, top_p, frequency_penalty, presence_penalty)

        progress_text = f"Generating descriptions for {rows} products..."
        
        with st.spinner(text=progress_text):
            bar = st.progress(0)
            output = {}
            with get_openai_callback() as cb:
                for key, value in input.items():
                    result, title = model.generate_description(value["request"], value["examples"])
                    output[key] = {
                        "title": title,
                        "result": result
                    }
                    counter += 1
                    bar.progress(math.ceil(counter/rows*100))
        
        timeend = time.time()
        elapsed = timeend-timestart
        bar.progress(100, text = "Done!")

        st.balloons()

        totalCost = cb.prompt_tokens * MODELS[modelSelect]["prompt_token_cost"] + cb.completion_tokens * MODELS[modelSelect]["completion_token_cost"]

        with st.expander("View execution details"):
            st.text(f"""Time elapsed: {time.strftime('%Hh %Mm %Ss', time.gmtime(elapsed))}\
                     \nAvg. time per description: {time.strftime('%Hh %Mm %Ss', time.gmtime(elapsed/rows))} """)
            st.text(f"""Total Tokens: {cb.total_tokens}\
                \n  Prompt Tokens: {cb.prompt_tokens}\
                \n  Completion Tokens: {cb.completion_tokens}""")
            st.text(f"""Estimated total cost: NOK {totalCost:.2f}\
                \nEst. Avg. cost per description: NOK {totalCost/rows:.2f}""")

        st.divider()

        st.subheader("Results")

        outputdf = pd.DataFrame.from_dict(output, orient="index").reset_index()
        outputdf.columns = ["ARTICLECODE", "TITLE", "LONGDESCRIPTION"]
        outputcsv = outputdf.to_csv(index=False).encode("utf-8")
        st.download_button('Download results as .CSV', outputcsv, "product-descriptions.csv", "text/csv", type="primary")#, key="download-tools-csv")

        for key, value in output.items():

            with st.expander(f"{key} - {value['title']}"):
                st.markdown(f"##### {key} - {value['title']}")
                st.markdown(value["result"], unsafe_allow_html=True)
                st.divider()

if __name__ == "__main__":
    main()