#!python3
"""
Name: PDF_paser.py
Date: 11-14-19 10:20
Author: Jainam, Brian, Bill

---
Parsing through PDF document and generate featrues for machine learning
model.

"""

import io
import nltk
import datetime
import pdfminer
import datefinder
import json
import re
import argparse

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage, PDFTextExtractionNotAllowed
from pdfminer.converter import PDFPageAggregator
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfdevice import PDFDevice
from pdfminer.layout import LAParams, LTTextBox, LTTextLine
from nltk.corpus import stopwords


class Features_Generation:
    """Class for parsing PDF file and return features that will be used 
    for machine learning.
    """

    def __init__(self, pdf_path, page_list=None, convert_to_text=False):
        """[summary]
        
        Args:
            pdf_path (string): path to the PDF file will be processed

            page_list ([int], optional): List of pages selected to be 
            processed. Defaults to None.

            convert_to_text (bool, optional): whether parsing content, 
            default is to parsing only the meta or structure information
            of the PDF. Defaults to False.
        """
        self.pdf_path = pdf_path
        self.page_list = page_list
        if convert_to_text:
            self.raw_text = self._convert_pdf_to_txt(self.pdf_path, self.page_list)
            self.text = " ".join(self.raw_text)

            self.tokenize_words = nltk.word_tokenize(self.text)
            self.unique_tokenize_words = sorted(set(self.tokens))

            self.tokenize_sentences = nltk.sent_tokenize(self.text)

        self.features = self._extract_features(self.pdf_path, self.page_list)

    def summary(self):
        print(f'Word token count: {len(self.tokenize_words)}')
        print(f'Sentence token count: {len(self.tokenize_sentences)}')

    def _convert_pdf_to_txt(self, pdf_path, page_list, codec='utf-8', password="",\
        maxpages=0, caching=True):
        """
        This is a functhion that extract all the text from a pdf file.

        Args:
            pdf_path (str): path of the pdf need to be processed
        """

        rsrcmgr = PDFResourceManager()
        retstr = io.StringIO()
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, codec=codec, \
            laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        pagenos = set()

        with open(pdf_path, 'rb') as fp:
            pages_objs_list = list(PDFPage.get_pages(fp, pagenos, maxpages=maxpages,
                                        password=password,
                                        caching=caching,
                                        check_extractable=True))
            if page_list != None:
                pages_objs_list = [pages_objs_list[i] for i in page_list]
            for page in pages_objs_list:
                interpreter.process_page(page)

        text = retstr.getvalue().split('\n')
        test = [x.strip() for x in text if x.strip()]
        device.close()
        retstr.close()
        return test

    def _extract_features(self, pdf_path, page_list, password=""):
        """This function will parse pdf file and extract features.

        Args:
            pdf_path (str): location of pdf file to be processed
            password (str, optional): password to read in the pdf file. 
            Defaults to "".

            page_list (list of int): pages will be need to extract 
            features from
            Defaults to None: extract features from all the pages.

        Raises:
            PDFTextExtractionNotAllowed: raise error if pdf was decipted
            and not allowed to be extract

        Returns:
            list of dict: A dictionary of features extracted from the 
            pdf file.
        """
        fp = open(pdf_path, "rb")
        parser = PDFParser(fp)
        document = PDFDocument(parser, password)
        if not document.is_extractable:
            raise PDFTextExtractionNotAllowed

        laparams = LAParams() # set
        resource_manager = PDFResourceManager()
        device = PDFPageAggregator(resource_manager, laparams=laparams)
        interpreter = PDFPageInterpreter(resource_manager, device)

        all_attributes = []

        list_of_page_obj = list(PDFPage.create_pages(document))
        total_pages = len(list_of_page_obj)

        # check whether to remove the first page:
        if page_list != None:
            interpreter.process_page(list_of_page_obj[0])
            layout = device.get_result()
            list_of_line_obj = list(layout._objs)
            total_lines_per_page = len(list_of_line_obj)
            if total_lines_per_page < 5:
                _ = list_of_page_obj.pop(0)
            list_of_page_obj = [list_of_page_obj[i] for i in page_list]
        # print(len(list_of_page_obj))
        # for page in PDFPage.create_pages(document):
        for page in list_of_page_obj:
            interpreter.process_page(page)
            layout = device.get_result()
            
            list_of_line_obj = list(layout._objs)
            total_lines_per_page = len(list_of_line_obj)
            line_num = 1
            for obj in list_of_line_obj:
                feature_dict_per_line = \
                    self.__extract_features_per_line(obj)
                all_attributes.append(feature_dict_per_line)
                line_num  += 1
        return all_attributes

    def __extract_features_per_line(self, obj):
        if isinstance(obj, LTTextBox) or isinstance(obj, LTTextLine):
            extracted_text = obj.get_text().encode('ascii', 'ignore').decode('unicode_escape')
            attributes_dict = {}
            if extracted_text.rstrip() != "":
                attributes_dict["textline"] = re.sub("\n|\r", "", \
                    extracted_text.rstrip())
                attributes_dict["x0"] = obj.x0
                attributes_dict["x1"] = obj.x1
                attributes_dict["y0"] = obj.y0
                attributes_dict["y1"] = obj.y1
                attributes_dict["is_empty"] = obj.is_empty()
                attributes_dict["is_hoverlap"] = obj.is_hoverlap(obj)
                attributes_dict["hoverlap"] = obj.hoverlap(obj)
                attributes_dict["is_voverlap"] = obj.is_voverlap(obj)
                attributes_dict["voverlap"] = obj.voverlap(obj)
                attributes_dict["wordcount"] = len(extracted_text)
                attributes_dict["Cordinate Points"] = obj.bbox
                attributes_dict["height"] = obj.height
                attributes_dict["width"] = obj.width
                attributes_dict["numberoflines"] = extracted_text.count('\n')
                attributes_dict["index"] = obj.index
                attributes_dict["UpperCase"] = sum([1 for i in \
                    extracted_text if i.isupper()])
                # attributes_dict["analyze"] = str(obj.analyze)
                attributes_dict["bold"] ,attributes_dict["fontsize"] = \
                    self.__extract_font_style(obj)
                attributes_dict["textalign"] = \
                    self.__extract_textalign(obj.x0)
                attributes_dict["stopwordcount"], attributes_dict["bulletpoint"] = \
                    self.__extract_stopword_bulletpoint(obj)
                attributes_dict["currency"], \
                    attributes_dict["data_list"] \
                        = self.__extract_currency_date(obj)
            return attributes_dict

    def __extract_font_style(self, obj):
        """This function will extract font style.

        Args:
            obj (object): pdfminer object

        Returns:
            float: ratio of bold character count to all character count,
            float: ratio of font size count to all character count
        """
        boldcount = 0
        charcount = 0
        fontsize = 0
        for textline in obj:
            for ch in textline:
                try:
                    fontsize += ch.size
                    charcount += 1
                    if("Bold" in ch.fontname):
                        boldcount += 1
                except:
                    pass
        return boldcount/charcount, fontsize/charcount

    def __extract_textalign(self, x0):
        """This function will extract text alignment using x0 information

        Args:
            x0 (ing): left-top corner of the text box

        Returns:
            int: 0 = left align, 
                 1=middle align, 
                 2 = right align
        """
        textalign = -1
        if(int(x0) in range(0,100)):
            textalign = 0
        elif(int(x0) in range(100,300)):
            textalign = 1
        elif(int(x0) in range(300,550)):
            textalign = 2
        return textalign

    def __extract_stopword_bulletpoint(self, obj):
        """This function will extract stopword and bulletpoint.

        Args:
            obj (object): pdfminer object

        Returns:
            int: stopword count
            int: 0 = no bullet point is present, 
                 1 = bullet point is present
        """
        stop_words = set(stopwords.words('english'))

        stopwordcount = 0
        bulletpoint = 0

        for textline in obj:
            s = str(textline.get_text())
            words = s.split()
            for w in words:
                if w in stop_words: 
                    stopwordcount += 1

            bullet1 = re.findall("'[(][a-z]*[A-Z]*[)]\.",s)
            bullet2 = re.findall("'[(0-9)\.(0-9)(0-9)]*\.",s)
            bullet3 = re.findall("'[(][a-z]*[A-Z]*[)]",s)
            bullet4 = re.findall("'[(]*[0-9]*[)]*\.[(]*[0-9]*[)]*[(]*[0-9]*[)]*"\
                ,s)

            if (len(bullet1) > 0 or len(bullet2) > 0 or len(bullet3) > 0 or \
                len(bullet4) > 0):
                bulletpoint = 1

        return stopwordcount, bulletpoint

    def __extract_currency_date(self, obj):
        """This function will extract currency values

        Args:
            obj (object): pdfminer object

        Returns:
            list of list of string: a list of list of strings contain currency
            information.

        Example:
        If text in the pdf are like:
                ['Some $ 10 here',
                'And 10$ here',
                'And 10 $  here',
                'And 1000005 dollars here',
                'And dollars one million and five here',
                'USD 100']

        Function output:
               [['$ 10'],
                ['10$'],
                ['10 $'],
                ['1000005 dollars'],
                ['dollars one million and five'],
                ['USD 100']]

        https://stackoverflow.com/questions/3276180/extracting-date-from-a-string-in-python
        """
        re_dollar = r'(?:\$(?:usd)?|(?:dollar|buck)s?)'
        re_textnumber = r'\b(?!\s)(?:[\sa-]|zero|one|tw(?:elve|enty|o)|th(?:irt(?:een|y)|ree)|fi(?:ft(?:een|y)|ve)|(?:four|six|seven|nine)(?:teen|ty)?|eight(?:een|y)?|ten|eleven|forty|hundred|thousand|[mb]illion|and)+\b(?<!\s)'
        re_number = r'(?:[1-9][0-9,\.]+|{})'.format(re_textnumber)
        valuta_with_num = r'{0}\s?{1}|{1}\s?{0}'.format(re_number, re_dollar)
        currency_tags = re.compile(valuta_with_num)

        currency_text_list = []
        date_list = []

        for textline in obj:
            s = str(textline.get_text()).rstrip()
            currency_match = currency_tags.findall(s)
            if currency_match:
                currency_text_list.append(currency_match)
            # try:
            #     temp = dateutil.parser.parse(s, fuzzy=True)
            #     date_list.append({"year": temp.year, "month": temp.month, \
            #          "day": temp.day})
            # except:
            #     date_list.append(None)

            # date_matches = datefinder.find_dates(s)
            # for date_ in date_matches:
            #     date_list.append(str(date_))

            re_date = r'(Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?)\s+\d{1,2},\s+\d{4}'
            date_match = re.match(re_date, s)
            if date_match != None:
                date_list.append(date_match.group())


            # effective_date_flags = ['effective date', 
            #                         'dated and effect as of the']
            # if any([True for elem in effective_date_flags if elem in s.lower()]):
            # date_matches = datefinder.find_dates(s)
            # for date_ in date_matches:
            #     effective_date_list.append(str(date_))


        return currency_text_list, date_list
    
    def write_json(self, json_output_path):
        with open(json_output_path, 'w') as fp:
            json.dump(self.features, fp, indent=4, sort_keys=True)



if __name__ == '__main__':
    arguments_parser = argparse.ArgumentParser(description='Feature extraction\
        from PDF')
    arguments_parser.add_argument('-i', '--input', required=True,\
        help='path of input PDF file', type=str)
    arguments_parser.add_argument('-o', '--output', required=True,\
        help='path of output JSON file', type=str)

    ret = arguments_parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        raise ValueError(f'unknow argument: \
            {arguments_parser.parse_known_args()[1]}')
    
    feature_class = Features_Generation(options.input)
    feature_class.write_json(options.output)