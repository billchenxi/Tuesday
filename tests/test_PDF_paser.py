import unittest
non_info_list = ["Art.", "Art", "Article", "Sec.", "Sect.", "Section", 
                    "Sec", "Part", "Exhibit"]

[first_name, middle_name, period, last_name] = ['Ed', 'Jh', '.', 'Thomasons']

if (first_name.istitle() or first_name[0].isupper()) and \
    (last_name.istitle() or last_name[0].isupper()) and \
    (len(middle_name)==1) and period=='.' and \
    first_name not in non_info_list:
    print("Yeah")