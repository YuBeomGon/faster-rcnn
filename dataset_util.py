# Plz check the below link if needs
## https://doaiacropolis.atlassian.net/wiki/spaces/DOP/pages/1628340243/0.+SCL+Dataset+Annotation 
## https://doaiacropolis.atlassian.net/wiki/spaces/DOP/pages/1885700101/SCL

# doai diagnosis name mapper
CLASS_MAPPER = {
    # [DOAI]
    # ASC-US
    "ASC-US": "ASC-US",
    "ASCUS-SIL": "ASC-US",
    "ASC-US with HPV infection": "ASC-US",
    # ASC-H
    "ASC-H": "ASC-H",
    "ASC-H with HPV infection": "ASC-H",
    # LSIL
    "LSIL": "LSIL",
    "LSIL with HPV infection": "LSIL",
    # HSIL
    "HSIL": "HSIL",
    "H": "HSIL",
    "HSIL with HPV infection": "HSIL",
    # Carcinoma
    "Carcinoma": "Carcinoma",
    
    # [SCL]
    # ASC-US
    "AS": "ASC-US",
    # ASC-H
    "AH": "ASC-H",
    # LSIL
    "LS": "LSIL",
    # HSIL
    "HS": "HSIL",
    "HN": "HSIL",
    # Carcinoma
    "SM": "Carcinoma",
    "SC": "Carcinoma",
    "C": "Carcinoma",
    "Negative": 'Negative',
    "판독불가" : 'Negative',
    "Candida" : 'Benign',
    "Benign atypia" : 'Benign'
}