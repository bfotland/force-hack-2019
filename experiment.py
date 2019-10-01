import pandas as pd


def read_filtered():
    df = pd.read_excel('filterred.xlsx')
    return df


def read_abbreviations():
    df = pd.read_excel('RealPore excel vba to translate litho descriptions and sort.xlsm', sheet_name='Abbreviation')
    return df

filt = read_filtered()

orig_desc = list(filt['Formation description original'])
abb = read_abbreviations()
# print(abb)

abb[['Long', 'abbreviation']]
long_list = list(abb['Long'])
abbreviation_list = list(abb['abbreviation'])
abbreviation_to_long_mapping = {abbreviation_list[i].lower():long_list[i] for i in range(len(long_list))}
# print(abbreviation_to_long_mapping)

long_desc = []
for i, item in enumerate(orig_desc):
    
    if not type(item) is str:
        long_desc.append('')
        print('skipping: {}'.format(item))
        continue
    item = item.lower()
    # print(item)
    long_item = []

    extend_as_above = False
    if item.startswith('a.a.'):
        item = item.replace('a.a.', '')
        extend_as_above = True

    parts = item.replace('.',',').split(',')
    for part in parts:
        # extend_as_above = False
        subpart_list = []
        if part in sorted(abbreviation_to_long_mapping, key=len, reverse=True):
            long_item.append(abbreviation_to_long_mapping[part])
        else:
            for subpart in part.replace('/', '-').split('-'):
                # print(subpart)
                for abb in sorted(abbreviation_to_long_mapping, key=len, reverse=True):
            # for part in abbreviation_to_long_mapping:
                    if abb in subpart:
                        subpart_list.append(abbreviation_to_long_mapping[abb])
                        break
            if len(subpart) > 0:
                long_item.append("-".join(subpart_list))

            
        # print(long_item)

    if i % 1000 == 0:
        print(i)
        
    # print(long_item)
    long_str = " ".join(long_item)
    if extend_as_above:
        long_str = "{} {}".format(long_desc[i-1], long_str)
    long_desc.append(long_str)
    if i > 6:
        break

long_description_pd = pd.DataFrame(long_desc)
print(long_description_pd)