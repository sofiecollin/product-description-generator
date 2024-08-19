from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.chat_models import AzureChatOpenAI

import os
from dotenv import load_dotenv
load_dotenv(os.getcwd()+"/azureopenaiapikey.env")



class product_description_agent:
    """A class for generating the LLM model object and call the OpenAI service. """

    def __init__(self, system_context, product_description_guidelines, model = "gpt-4", temperature=1, top_p=1, frequency_penalty=0, presence_penalty=0):
        """Initialize an LLM agent with the provided model parameters"""
        BASE_URL = os.getenv("OPENAI_API_BASE")
        API_KEY = os.getenv("OPENAI_API_KEY")
        API_TYPE = os.getenv("OPENAI_API_TYPE")
        API_VERSION = os.getenv("OPENAI_API_VERSION")
        DEPLOYMENT_NAME = model

        self.model = AzureChatOpenAI(
            temperature=temperature,
            openai_api_base=BASE_URL,
            openai_api_version=API_VERSION,
            deployment_name=DEPLOYMENT_NAME,
            openai_api_key=API_KEY,
            openai_api_type=API_TYPE,
            model_kwargs = {
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty
                }
        )

        self.final_context = f"""
            {system_context}

            The product descriptions must follow the following guidelines: 
            {product_description_guidelines}

            The message history gives you examples of what the expected output looks like
        """     

    def create_prompt(self, input_examples):
        """Generate the Langchain prompt template used to call the model. 

        Args:
            input_examples (dictionary): Optional. Input examples used for few shot prompting given as list of input-output dictionary objects.

        Returns:
            final_prompt(ChatPromptTemplate): A langchain prompt template.
        """

        messages = [
            ("system", self.final_context)
        ]

        if input_examples != "":
            example_prompt = ChatPromptTemplate.from_messages(
                [
                    ("human", "{input}"),
                    ("ai", "{output}"),
                ]
            )
            few_shot_prompt = FewShotChatMessagePromptTemplate(
                example_prompt=example_prompt,
                examples=input_examples,
            )

            messages.append(few_shot_prompt)

        human_message = """{json_input}"""

        messages.append(
            ("human", human_message)
        )
        

        final_prompt = ChatPromptTemplate.from_messages(messages)

        return final_prompt


    def create_chain(self, input_examples):
        """Generates a chain that can be invoked to call the model.

        Args:
            input_examples (dictionary): Optional. Input examples used for few shot prompting given as list of input-output dictionary objects.

        Returns:
            Langchain chain: The Langchain chain that can be used to call the model. 
        """
        prompt = self.create_prompt(input_examples)
        return prompt | self.model
        

    def generate_description(self, request, examples):
        """Generates a langchain and invokes the configured model (self.model) to generate a product description for a given product. 

        Args:
            request (dictionary): The product to request a product description form in json, formatted as a Python Dictionary
            examples (dictionary): Optional. Input examples used for few shot prompting given as list of input-output dictionary objects.

        Returns:
            result.content (string): The generated product description. 
            title (string): Title of the product that the product description is for. 
        """
        chain = self.create_chain(examples)

        result = chain.invoke({
            "json_input": request})
        
        return result.content, request["TITLE"]
    
