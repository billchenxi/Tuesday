'''
Name: mini_model.py
Time: 11-14 6:40

Author: Bill Chen
---

This is a mini version of the actual model, here it will extract title, 
effective date, terminate date, and parties

Return JSON file with the above information for front end to show.
'''

import sys
import os
import dataclasses

from os import path
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import wordnet
from nltk.chunk import tree2conlltags

import pandas as pd
import numpy as np
import argparse
import json
import nltk
import spacy
import re

#optimize this by dowloading during container creation
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

sys.path.append(path.join(path.dirname(__file__), '../'))
from FeatureGeneration.PDF_paser import Features_Generation
from collections import OrderedDict
from pprint import pprint

@dataclasses.dataclass(init=True)
class Data:
    trigger_terms = ["( The","( the","("]

    defined_terms = ["Company", "Buyer", "Seller", "Sellers", 
                    "Purchaser", "Parent", "Guarantor", "Lender", 
                    "Borrower", "Lessor", "Lessee", "Landlord", 
                    "Tenant", "Creditor", "Contractor", "Customer",
                    "Indemnitee", "Employer", "Employee", "Bank",
                    "Trustee", "Supplier", "Licensee", "Licensor",
                    "Investor", "Debtor"]
                    
    known_org     = ["a Delaware corporation", "a Kansas corporation", 
                    "an Arizona corporation", "an Illinois corporation",
                    "a California corporation"]

    non_info_list = ["Art.", "Art", "Article", "Sec.", "Sect.", "Section", 
                    "Sec", "Part", "Exhibit"]

    punc = [","]

    frag = ["Inc.", "INC.", "Incorp.", "INCORP.", "LLC", "N.A.", "L.L.C.", 
            "LP", "L.P.", "B.V.", "BV", "N.V.", "NV", "Corp.", "CORP."]
    
    effect_date_signals = ["Effective Date", "Dated", "dated", 
            "effective as of", "Effective as of", "effective",
            "entered into as of", "Entered into as of", "as of"]


class Model(Data):
    def __init__(self, pdf_input_path, json_output_path=None, to_df=False):
        """Parse PDF and output Date, Parties, and Title
        
        Args:
            pdf_input (str): path of input pdf file
            json_output (str): path of output json file
            to_df (bool): whether to convert results to data frame for 
                generating csv.
        """
        super().__init__()
        self.pdf_input_path = pdf_input_path
        self.json_output_path = json_output_path
        self.to_df = to_df
        self.features_obj = Features_Generation(pdf_path=self.pdf_input_path,\
            page_list=[0, -1], convert_to_text=True)
        self.tokenize_words = self.features_obj.tokenize_words
        self.tokenize_raw_words = self.features_obj.tokenize_raw_words
        self.tokenize_sentences = self.features_obj.tokenize_sentences
        self.features = self.features_obj.features
        self.title = ' '.join(self.extract_title())
        # self.text = " ".join(self.features_obj.text)
        # self.results_300 = self.parsing_content()

    @property
    def pdf_input_path(self):
        """Path where input PDF located.
        
        Returns:
            str: Input path
        """
        return self._pdf_input_path
    
    @pdf_input_path.setter
    def pdf_input_path(self, val):
        if not isinstance(val, str) and val is not None:
            raise TypeError("Input path should be of class str.")
        if val is not None and not os.path.isfile(val):
            raise ValueError("Input path doesn't exist")
        self._pdf_input_path = val
    
    @property
    def json_output_path(self):
        """Path where output JSON located.
        
        Returns:
            str: Output path
        """
        return self._json_output_path

    @json_output_path.setter
    def json_output_path(self, val):
        if not isinstance(val, str) and val is not None:
            raise TypeError("Output path should be of class str.") 
        self._json_output_path = val


    def extract_title(self):
        title_list = []
        first_10 = self.features[:10]

        for indx, val in enumerate(first_10):
            if val != {} and val != None:
                s = val["textline"]
                words = val["textline"].split(' ')
                # Find title index
                if val["textalign"] == 1 and \
                    indx < 10 and \
                    len(words) < 10 and \
                    not all([i.lower() in val["textline"].lower() for i \
                        in self.non_info_list]) and \
                    (words[0].istitle() or words[0].isupper()):
                    title_list.append(s)
        return title_list

        def extract_parties(self):
            parties_list = []
            for indx, word in enumerate(self.tokenize_raw_words):
                whether_person_name = _whether_person_name(indx)
                # If it's a person's name
                if  whether_person_name != False:
                    parties_list.append(whether_person_name)

        def _whether_person_name(self, indx):
            [first_name, second_name, third_name, forth_name] = self.tokenize_raw_words[indx:indx+4]
            if (first_name.istitle() or first_name.isupper()) and \
                (forth_name.istitle() or forth_name.isupper()) and \
                (len(second_name)==1) and third_name=='.' and \
                first_name not in self.non_info_list:
                return ' '.join([first_name, second_name, third_name, forth_name])

            elif (first_name.istitle() or first_name.isupper()) and \
                (third_name.istitle() or third_name.isupper()) and \
                (len(second_name)==2) and second_name[1]=='.' and \
                first_name not in self.non_info_list:
                return ' '.join([first_name, second_name, third_name])
            
            else:
                return False
        

                

        # model_300_output = OrderedDict([
        #     ("parties", ' '.join(list(subset_parties_df))),
        #     ("effective_date", date_dict["effective_date"]),
        #     ("signed_date", date_dict["signed_date"]),
        #     ("titles", ' '.join(title_list))
        # ])

        # return model_300_output

    def extract_parties(self):
        """ Read in textline and extract names of parties basing on predefined
        rules


        """
        # self.parties
        pass

    def write_json_output(self, output_dict):
        with open(self.json_output_path, 'w') as fp:
            json.dump(output_dict, fp, indent=4, sort_keys=True)




if __name__ == '__main__':
    arguments_parser = argparse.ArgumentParser(description='Process PDF file\
        and output Data, Title, and Parties')
    arguments_parser.add_argument('-i', '--input', required=True,\
        help='path of input PDF file', type=str)

    ret = arguments_parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        raise ValueError(f'unknow argument: \
            {arguments_parser.parse_known_args()[1]}')
    pdf_parser = Model(options.input)
    print(pdf_parser.title)




