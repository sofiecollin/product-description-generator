import pandas as pd
import numpy as np
import json

class Document_Handler:
    """
    Class for handling and generating input data for the LLM model.
    """
    
    def __init__(self, dfpd, dfad, dfpd_ex, dfad_ex):
        """Initialize the document handler object with all dataframes. 

        Args:
            dfpd (Dataframe): Product Information - request. Information about all products for which a description will be generated.
            dfad (Dataframe): Product Attributes - request. Supporting attributes about the products, e.g. height, width, resolution.
            dfpd_ex (Dataframe): Example Product Information. Information about products that will be used as examples if doing "few shot prompting"
            dfad_ex (Dataframe): Example Product Attributes. Supporting attributes about the above examples products. 
        """
        self.dfpd = dfpd
        self.dfad = dfad
        self.dfpd_ex = dfpd_ex
        self.dfad_ex = dfad_ex

    def generate_input_file(self, include_examples):
        """
        Generates an input file based on the initiazed documents for this Document Handler object.

        For each product in the input data, two objects are created: 
        - A request json including all product informaiton formatted as a dictionary (to be used as a json file later on.)
        - A set of examples that includes input as a dictionary (similar to the request), and an output formatted as a string

        Note: Duplicate product attributes per product are skipped.

        Returns:
            Input data (list): A list of article codes with related input data and examples. 
        """

        list_to_return ={}
        for i in self.dfpd.index:
            article_code = self.dfpd.loc[i, "ARTICLECODE"]
            dfrow = self.supplement_df_with_article_attributes(article_code, self.dfpd, self.dfad)
            dfrow = dfrow.loc[:,~dfrow.columns.duplicated()].copy() #deleting duplicate column names. TODO: rename columns that are duplicate

            if include_examples:
                examples = self.generate_examples(self.dfpd.loc[i, "GROUPNAME"])
            else: 
                examples = ""

            list_to_return[str(article_code)] = {
                "request": self.convert_dfrow_to_json(dfrow),
                "examples": examples
            }
            
        return list_to_return

    def supplement_df_with_article_attributes(self, article_code:str, dfpd:pd.DataFrame, dfad:pd.DataFrame)-> pd.DataFrame:
        """Supplements each row with attributes. 

        The product information input dataframe is filtered down to one row based on the article code of the product to be processed. 
        Next, each relevant product attribute (filtered on article code) in the product attributes file is added as a separate column to the above one-row-dataframe. 

        Args:
            article_code (str): Article code for the product that will be supplemented with connected attributes. 
            dfpd (pd.DataFrame): Dataframe with product information. 
            dfad (pd.DataFrame): Dataframe with product attributes.  

        Returns:
            pd.DataFrame: A dataframe consisting of a single row where all product information fields and all product attibutes are given as separate columns. 
        """

        dfpd = dfpd[dfpd['ARTICLECODE']== article_code]
        dfad = dfad[dfad['ARTICLECODE']== article_code]
        prod_cols= dfpd.columns.tolist()
        result_df= dfpd.copy()
        # if we have attributes for the given article, add it to the result_df
        if not dfad.empty:
            result_df = dfpd.merge(dfad, on='ARTICLECODE', how='left')
            result_df_with_att_value = result_df[pd.notna(result_df['VALUE'])]
            # make two seprate dfs with one row from the result_df_with_att_value: df_1 and df_2 
            one_row_df_with_prod_cols= pd.DataFrame(result_df_with_att_value.iloc[0,:len(prod_cols)]).T
            df_1= one_row_df_with_prod_cols.reset_index(drop=True) 
            df_2_temp= pd.DataFrame(result_df_with_att_value.iloc[:,len(prod_cols):]).T      
            df_2= df_2_temp.reset_index(drop=True)
            df_2.columns = df_2.iloc[0]
            df2= df_2.iloc[1:].reset_index(drop=True)
            # combine two one-row dataframes
            return pd.concat([df_1, df2], axis=1)
        else:
            result_df["ATTRIBUTENAME"]= np.nan
            result_df["VALUE"]= np.nan
            return result_df

    def convert_dfrow_to_json(self, dfrow):
        """Converts a dataframe with one row to a json file where each column is a json attributes with the related value. 

        Args:
            dfrow (dataframe): Dataframe with one row that contains information about a given product. 

        Returns:
            Dictionary: A python dictionary that simulates a json file. 
        """

        return json.loads(dfrow.to_json(orient='records', lines=True))
    
    def generate_examples(self, groupname):
        """Generates a list of examples based on the example data files in the Documnet Handler object for a given group name.

        For the given group name, e.g. SmartPhones, a list of example input output pairs are generated. 
        - Input as a product informaiton formatted as a dictionary (to be used as a json file later on.). The LONGDESCRIPTION is treated as the "output" and excluded from this object.
        - Output formatted as a string, based on the LONGDESCRIPTION field of the example product information dataset (self.dfpd_ex)

        Args:
            groupname (string): Groupname for which to generate product description examples. 

        Returns:
            List: A list of input-output pairs.
        """
        examples = []
        df_examples = self.dfpd_ex[self.dfpd_ex["GROUPNAME"] == groupname]
        for i in df_examples.index:
            result = df_examples.loc[i, "LONGDESCRIPTION"]
            input_df = df_examples.drop(["LONGDESCRIPTION", "language"], axis=1)
            article_code = df_examples.loc[i, "ARTICLECODE"]
            dfrow = self.supplement_df_with_article_attributes(article_code, input_df, self.dfad_ex)
            dfrow = dfrow.loc[:,~dfrow.columns.duplicated()].copy()
            examples.append({
                "input": self.convert_dfrow_to_json(dfrow),
                "output": result
            })

        return examples