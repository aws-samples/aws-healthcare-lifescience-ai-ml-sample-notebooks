GENOME_GROUPS = {
    'metagene_19' : ['LRIG1', 'HPGD', 'GDF15'],
    'metagene_10' : ['CDH2', 'POSTN', 'VCAN', 'PDGFRA'],
    'metagene_9' : ['VCAM1', 'CD44', 'CD48'],
    'metagene_4' : ['CD4', 'LYL1', 'SPI1', 'CD37'],
    'metagene_3' : ['VIM', 'LMO2', 'EGR2'],
    'metagene_21' : ['BGN', 'COL4A1', 'COL5A1', 'COL5A2'],
}

_all = []
for group in GENOME_GROUPS.values():
    _all.extend(group)

GENOME_GROUPS["ALL"] = _all    



