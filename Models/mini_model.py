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
    eng_word = set(nltk.corpus.words.words())

    # trigger_terms = ["(The","(the","("]

    defined_terms = ["Company", "Companies", "Buyer", "Buyers", "Seller", "Sellers",
                    "Purchaser", "Purchasers", "Parent", "Parents", "Guarantor",
                    "Guarantors", "Lender", "Lenders", "Borrower", "Borrowers",
                    "Lessor", "Lessors", "Lessee", "Lessees", "Landlord", "Landlords",
                    "Tenant", "Tenants", "Creditor", "Creditors", "Contractor",
                    "Contractors", "Customer", "Customers", 
                    "Indemnitee", "Indemnitees", "Employer", "Employers",
                    "Employee", "Employees", "Bank", "Banks",
                    "Trustee", "Trustees", "Supplier", "Suppliers", "Licensee",
                    "Licensees", "Licensor", "Licensors", "Investors", "Investor",
                    "Debtor", "Debtors"]
                    
    # known_org     = ["a Delaware corporation", "a Kansas corporation", 
    #                 "an Arizona corporation", "an Illinois corporation",
    #                 "a California corporation"]

    non_info_list = ["Art.", "Art", "Article", "Sec.", "Sect.", "Section", 
                    "Sec", "Part", "Exhibit", "Name:", "name:"]
    
    effect_date_signals = ["Effective Date", "Dated", "dated", 
            "effective as of", "Effective as of", "effective",
            "entered into as of", "Entered into as of", "entered into as of", "as of"]


class Model(Data):
    def __init__(self, pdf_input_path, json_output_path=None, to_df=False):
        """Parse PDF and output Date, Parties, and Title
        
        Args:
            pdf_input (str): path of input pdf file
            json_output (str): path of output json file
            to_df (bool): whether to convert results to data frame for 
                generating csv.
        """
        super(Model).__init__()
        self.pdf_input_path = pdf_input_path
        self.json_output_path = json_output_path
        self.to_df = to_df
        self.features_obj = Features_Generation(pdf_path=self.pdf_input_path,\
            page_list=None, convert_to_text=True)
        self.raw_text = self.features_obj.raw_text
        self.tokenize_words = self.features_obj.tokenize_words
        self.tokenize_raw_words = self.features_obj.tokenize_raw_words
        self.tokenize_original = self.features_obj.tokenize_original
        self.tokenize_sentences = self.features_obj.tokenize_sentences
        self.features = self.features_obj.features

        self.title = ' '.join(self.extract_title())
        self.clean_title = " ".join(w for w in nltk.wordpunct_tokenize(self.title) \
            if w.lower() in self.eng_word or not w.isalpha())

        self.parties = self.extract_parties(self.raw_text)
        self.persons = self.extract_persons(self.tokenize_original)
        self.effective_dates = self.extract_effective_dates(self.tokenize_sentences)
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



    def extract_effective_dates(self, tokenize_sentences):
        effective_date_list = []
        match_date = r'(?:Jan|January|Feb|February|Mar|March|Apr|April|May|Jun|June|Jul|July|Aug|August|Sep|September|Oct|October|Nov|November|Dec|December)\s*(?:\d{1,2}),\s*(?:19{2}\d|2\d{3})'
        
        for sentence in tokenize_sentences:
            if any([True for signal in self.effect_date_signals if signal in sentence]):
                effective_date_list.extend(re.findall(match_date, sentence))
        return list(set(effective_date_list))

    def extract_persons(self, tokens):
        persons_list = []
        for indx, token in enumerate(tokens):
            # Extract the person's name
            party_name, _ = self._whether_person_name(tokens, indx)
            if party_name != None:
                party_name.encode('utf-8')
                persons_list.append(party_name)  
        return list(set(persons_list))

    def extract_parties(self, raw_text):
        parties_list = []

        match_1 = r'([A-Z][a-z]+\s?|\,?)+?\s*(and\s*[A-Z][a-z]+)?(, \bL\.?L\.?C\.?\b|\bL\.?P\.?\b|\bB\.?V\.?\b|\bN\.?V\.?\b|\bInc.\b|\bINC.\b|\bIncorp.\b|\bINCORP.\b|\bN\.A\.\b|\bCorp.\b|\bCORP.\b)'
        match_2 = r'([A-Z][a-z]+\s?|\,?)+?\s*(and\s*[A-Z][a-z]+)?(\s+\bCorporation\b|\bCompany\b|\bBank\b)'

        parties_list.extend(["".join(x) for x in re.findall(match_1, raw_text)])
        parties_list.extend(["".join(x) for x in re.findall(match_2, raw_text)])
        
        final_parties_list = []
        problem_list = ['Trust Company', 'Growth Company', 'Matrix  Corporation', 'Trust Company', 'Investment Company', 'Public Company', 'Investment Company']
        for party in list(set(parties_list)):
            condition_1 = (party.strip() not in problem_list)
            clean_party = party.replace(',', ' ').strip().split()
            condition_2 = (len(clean_party) > 1)
            print(condition_2)
            if condition_2 and (clean_party[0] in ['The', 'Each']):
                condition_3 = (clean_party[1] not in self.defined_terms)
            else:
                condition_3 = True
        
            if all([condition_1, condition_2, condition_3]):
                final_parties_list.append(party)

        return final_parties_list

    def _whether_person_name(self, tokens, indx):
        try:
            first_name, second_name, third_name, forth_name = tokens[indx], tokens[indx+1], tokens[indx+2], tokens[indx+3]

            if (first_name.istitle() or first_name.isupper()) and \
                (forth_name.istitle() or forth_name.isupper()) and \
                (len(second_name)==1 and second_name.isalpha()) and third_name=='.' and \
                first_name not in self.non_info_list:
                return ' '.join([first_name, second_name, third_name, forth_name]), indx + 5

            elif (first_name.istitle() or first_name.isupper()) and \
                (third_name.istitle() or third_name.isupper()) and \
                (len(second_name)==2 and  second_name[0].isalpha()) and second_name[1]=='.' and \
                first_name not in self.non_info_list:
                return ' '.join([first_name, second_name, third_name]), indx + 4
            else:
                return None, indx
        except: 
            return None, indx


        # model_300_output = OrderedDict([
        #     ("parties", ' '.join(list(subset_parties_df))),
        #     ("effective_date", date_dict["effective_date"]),
        #     ("signed_date", date_dict["signed_date"]),
        #     ("titles", ' '.join(title_list))
        # ])

        # return model_300_output


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
    # print('(the' in pdf_parser.tokenize_original)
    print(pdf_parser.parties, pdf_parser.persons, pdf_parser.effective_dates)
    # with open("sample.txt", "w") as text_file:
    #     text_file.write(pdf_parser.raw_text)

    # print(pdf_parser.parties, pdf_parser.persons)




