import collections
import json
from collections import defaultdict

import numpy as np
import torch
from torchtext.vocab import Vocab, Vectors

from dataset.utils import tprint


def _get_20newsgroup_classes(args):
    '''
        @return list of classes associated with each split
    '''
    label_dict = {
            'talk politics mideast': 0,
            'sci space': 1,
            'misc forsale': 2,
            'talk politics misc': 3,
            'comp graphics': 4,
            'sci crypt': 5,
            'comp windows x': 6,
            'comp os ms-windows misc': 7,
            'talk politics guns': 8,
            'talk religion misc': 9,
            'rec autos': 10,
            'sci med': 11,
            'comp sys mac hardware': 12,
            'sci electronics': 13,
            'rec sport hockey': 14,
            'alt atheism': 15,
            'rec motorcycles': 16,
            'comp sys ibm pc hardware': 17,
            'rec sport baseball': 18,
            'soc religion christian': 19,
        }

    val_classes = list(range(5))
    train_classes = list(range(5, 13))
    test_classes = list(range(13, 20))

    return train_classes, val_classes, test_classes, label_dict


def _get_amazon_classes(args):
    '''
        @return list of classes associated with each split
    '''
    label_dict = {
        'Amazon Instant Video is a subscription video on-demand over-the-top streaming and rental service of Amazon.com': 0,
        'Apps for Android is a computer program or software application designed to run on a Android device': 1,
        'Automotive is concerned with self-propelled vehicles or machines': 2,
        'Baby is an extremely young child': 3,
        'Beauty products like Cosmetics are constituted from a mixture of chemical compounds derived from either natural sources or synthetically created ones': 4,
        'Books are long written or printed literary compositions': 5,
        'CDs and DVDs are digital optical disc data storage formats to store and play digital audio recordings or music , something similar including compact disc (CD), vinyl, audio tape, or another medium': 6,
        'Cell Phones and Accessories refer to mobile phone and some hardware designed for the phone like microphone , headset': 7,
        'Clothing ，Shoes and Jewelry are items worn on the body to protect and comfort the human or for personal adornment': 8,
        'Albums and Digital Music are collections of audio recordings including popular songs and splendid music': 9,
        'Electronics refer to electronic devices, or the part of a piece of equipment that consists of electronic devices': 10,
        'Grocery and Gourmet Food refer to stores primarily engaged in retailing a general range of food products': 11,
        'Health and Personal Care refer to consumer products used in personal hygiene and for beautification': 12,
        'Home and Kitchen refer to something used in Home and Kitchen such as Kitchenware , Tableware , cleaning tools': 13,
        'Kindle Store is an online e-book e-commerce store operated by Amazon as part of its retail website and can be accessed from any Amazon Kindle': 14,
        'Movies and TV is a work of visual art that tells a story and that people watch on a screen or television or a showing of a motion picture especially in a theater': 15,
        'Musical Instruments are devices created or adapted to make musical sounds': 16,
        'Office Products are consumables and equipment regularly used in offices by businesses and other organizations': 17,
        'Patio Lawn and Garden refer to some tools and devices used in garden or lawn': 18,
        'Pet Supplies refer to food or other consumables or tools that will be used when you keep a pet': 19,
        'Sports and Outdoors refer to some tools and sport equipment used in outdoor sports': 20,
        'Tools and Home Improvement refer to hand tools or implements used in the process of renovating a home': 21,
        'Toys and Games are something used in play , usually undertaken for entertainment or fun, and sometimes used as an educational tool.': 22,
        'Video Games or Computer games are electronic games that involves interaction with a user interface or input device to generate visual feedback , which include arcade games , console games , and personal computer (PC) games': 23,
    }

    val_classes = list(range(5))
    test_classes = list(range(5, 14))
    train_classes = list(range(14, 24))

    return train_classes, val_classes, test_classes, label_dict


def _get_fewrel_classes(args):
    '''
        @return list of classes associated with each split
    '''
    label_dict = {
         'place served by transport hub territorial entity or entities served by this transport hub (airport, train station, etc.)': 0,
         'mountain range range or subrange to which the geographical item belongs': 1,
         'religion religion of a person, organization or religious building, or associated with this subject': 2,
         "participating team Like 'Participant' (P710) but for teams. For an event like a cycle race or a football match you can use this property to list the teams and P710 to list the individuals (with 'member of sports team' (P54) as a qualifier for the individuals)": 3,
         'contains administrative territorial entity (list of) direct subdivisions of an administrative territorial entity': 4,
         'head of government head of the executive power of this town, city, municipality, state, country, or other governmental body': 5,
         'country of citizenship the object is a country that recognizes the subject as its citizen': 6,
         'original network network(s) the radio or television show was originally aired on, not including later re-runs or additional syndication': 7,
         'heritage designation heritage designation of a cultural or natural site': 8,
         'performer actor, musician, band or other performer associated with this role or musical work': 9,
         'participant of event a person or an organization was/is a participant in, inverse of P710 or P1923': 10,
         'position held subject currently or formerly holds the object position or public office': 11,
         'has part part of this subject; inverse property of "part of" (P361). See also "has parts of the class" (P2670).': 12,
         'location of formation location where a group or organization was formed': 13,
         'located on terrain feature located on the specified landform. Should not be used when the value is only political/administrative (P131) or a mountain range (P4552).': 14,
         'architect person or architectural firm that designed this building': 15,
         'country of origin country of origin of this item (creative work, food, phrase, product, etc.)': 16,
         'publisher organization or person responsible for publishing books, periodicals, games or software': 17,
         'director director(s) of film, TV-series, stageplay, video game or similar': 18,
         'father male parent of the subject. For stepfather, use "stepparent" (P3448)': 19,
         'developer organisation or person that developed the item': 20,
         'military branch branch to which this military unit, award, office, or person belongs, e.g. Royal Navy': 21,
         'mouth of the watercourse the body of water to which the watercourse drains': 22,
         'nominated for award nomination received by a person, organisation or creative work (inspired from "award received" (Property:P166))': 23,
         'movement literary, artistic, scientific or philosophical movement associated with this person or work': 24,
         'successful candidate person(s) elected after the election': 25,
         'followed by immediately following item in a series of which the subject is a part [if the subject has been replaced, e.g. political offices, use "replaced by" (P1366)]': 26,
         'manufacturer manufacturer or producer of this product': 27,
         'instance of that class of which this subject is a particular example and member (subject typically an individual member with a proper name label); different from P279; using this property as a qualifier is deprecated—use P2868 or P3831 instead': 28,
         'after a work by artist whose work strongly inspired/ was copied in this item': 29,
         'member of political party the political party of which this politician is or has been a member': 30,
         'licensed to broadcast to place that a radio/TV station is licensed/required to broadcast to': 31,
         'headquarters location specific location where an organization\'s headquarters is or has been situated. Inverse property of "occupant" (P466).': 32,
         'sibling the subject has the object as their sibling (brother, sister, etc.). Use "relative" (P1038) for siblings-in-law (brother-in-law, sister-in-law, etc.) and step-siblings (step-brothers, step-sisters, etc.)': 33,
         'instrument musical instrument that a person plays': 34,
         "country sovereign state of this item; don't use on humans": 35,
         'occupation occupation of a person; see also "field of work" (Property:P101), "position held" (Property:P39)': 36,
         'residence the place where the person is or has been, resident': 37,
         'work location location where persons were active': 38,
         'subsidiary subsidiary of a company or organization, opposite of parent organization (P749)': 39,
         'participant person, group of people or organization (object) that actively takes/took part in an event or process (subject).  Preferably qualify with "object has role" (P3831). Use P1923 for participants that are teams.': 40,
         'operator person, profession, or organization that operates the equipment, facility, or service; use country for diplomatic missions': 41,
         'characters characters which appear in this item (like plays, operas, operettas, books, comics, films, TV series, video games)': 42,
         'occupant a person or organization occupying property': 43,
         "genre creative work's genre or an artist's field of work (P101). Use main subject (P921) to relate creative works to their topic": 44,
         'operating system operating system (OS) on which a software works or the OS installed on hardware': 45,
         'owned by owner of the subject': 46,
         'platform platform for which a work was developed or released, or the specific platform version of a software product': 47,
         'tributary stream or river that flows into this main stem (or parent) river': 48,
         'winner winner of an event - do not use for awards (use P166 instead), nor for wars or battles': 49,
         'said to be the same as this item is said to be the same as that item, but the statement is disputed': 50,
         'composer person(s) who wrote the music [for lyricist, use "lyrics by" (P676)]': 51,
         'league league in which team or player plays or has played in': 52,
         'record label brand and trademark associated with the marketing of subject music recordings and music videos': 53,
         'distributor distributor of a creative work; distributor for a record label': 54,
         'screenwriter person(s) who wrote the script for subject item': 55,
         'sports season of league or competition property that shows the competition of which the item is a season. Use P5138 for "season of club or team".': 56,
         'taxon rank level in a taxonomic hierarchy': 57,
         'location location of the item, physical object or event is within. In case of an administrative entity use P131. In case of a distinct terrain feature use P706.': 58,
         'field of work specialization of a person or organization; see P106 for the occupation': 59,
         'language of work or name language associated with this creative work (such as books, shows, songs, or websites) or a name (for persons use P103 and P1412)': 60,
         'applies to jurisdiction the item (an institution, law, public office ...) or statement belongs to or has power over or applies to the value (a territorial jurisdiction: a country, state, municipality, ...)': 61,
         "notable work notable scientific, artistic or literary work, or other work of significance among subject's works": 62,
         'located in the administrative territorial entity the item is located on the territory of the following administrative entity. Use P276 (location) for specifying the location of non-administrative places and for items about events': 63,
         'crosses obstacle (body of water, road, ...) which this bridge crosses over or this tunnel goes under': 64,
         'original language of film or TV show language in which a film or a performance work was originally created. Deprecated for written works; use P407 ("language of work or name") instead.': 65,
         'competition class official classification by a regulating body under which the subject (events, teams, participants, or equipment) qualifies for inclusion': 66,
         'part of object of which the subject is a part (it\'s not useful to link objects which are themselves parts of other objects already listed as parts of the subject). Inverse property of "has part" (P527, see also "has parts of the class" (P2670)).': 67,
         'sport sport in which the subject participates or belongs to': 68,
         'constellation the area of the celestial sphere of which the subject is a part (from a scientific standpoint, not an astrological one)': 69,
         'position played on team / speciality position or specialism of a player on a team, e.g. Small Forward': 70,
         'located in or next to body of water sea, lake or river': 71,
         "voice type person's voice type. expected values: soprano, mezzo-soprano, contralto, countertenor, tenor, baritone, bass (and derivatives)": 72,
         'follows immediately prior item in a series of which the subject is a part [if the subject has replaced the preceding item, e.g. political offices, use "replaces" (P1365)]': 73,
         'spouse the subject has the object as their spouse (husband, wife, partner, etc.). Use "partner" (P451) for non-married companions': 74,
         'military rank military rank achieved by a person (should usually have a "start time" qualifier), or military rank associated with a position': 75,
         'mother female parent of the subject. For stepmother, use "stepparent" (P3448)': 76,
         'member of organization or club to which the subject belongs. Do not use for membership in ethnic or social groups, nor for holding a position such as a member of parliament (use P39 for that).': 77,
         'child subject has object as biological, foster, and/or adoptive child': 78,
         'main subject primary topic of a work (see also P180: depicts)': 79,
    }

    train_classes = [0, 1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16, 19, 21,
                     22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                     39, 40, 41, 43, 44, 45, 46, 48, 49, 50, 52, 53, 56, 57, 58,
                     59, 61, 62, 63, 64, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75,
                     76, 77, 78]

    val_classes = [7, 9, 17, 18, 20]
    test_classes = [23, 29, 42, 47, 51, 54, 55, 60, 65, 79]

    return train_classes, val_classes, test_classes , label_dict

def _get_huffpost_classes(args):
    '''
        @return list of classes associated with each split
    '''
    label_dict = {
        'entertainment': 2,
        'politics': 0,
        'world news': 28,
        'black voices': 12,
        'crime': 20,
        'women': 17,
        'comedy': 10,
        'weird news': 22,
        'sports': 11,
        'media': 21,
        'queer voices': 7,
        'tech': 30,
        'religion': 25,
        'science': 27,
        'travel': 3,
        'business': 9,
        'latino voices': 38,
        'impact': 18,
        'education': 40,
        'parents': 14,
        'style': 26,
        'healthy living': 6,
        'green': 23,
        'arts & culture': 35,
        'taste': 29,
        'college': 37,
        'the worldpost': 15,
        'good news': 34,
        'worldpost': 24,
        'arts': 32,
        'fifty': 33,
        'wellness': 1,
        'parenting': 5,
        'style & beauty': 4,
        'divorce': 19,
        'weddings': 16,
        'food & drink': 8,
        'home & living': 13,
        'money': 31,
        'culture & arts': 39,
        'environment': 36,


    }

    val_classes = list(range(5))
    train_classes = list(range(5, 25))
    test_classes = list(range(25, 41))

    return train_classes, val_classes, test_classes , label_dict


def _get_reuters_classes(args):
    '''
        @return list of classes associated with each split
    '''
    label_dict = {
        'tariffs': 28,
        'grain': 12,
        'ship': 25,
        'gold': 11,
        'acquisition merge': 0,
        'tin': 27,
        'industrial production': 14,
        'profit': 9,
        'unemployment': 16,
        'sugar': 26,
        'inflation': 7,
        'treasury bank': 18,
        'rate': 13,
        'cocoa': 3,
        'coffee': 4,
        'oil': 8,
        'cotton': 6,
        'cattle': 17,
        'money supply': 19,
        'copper': 5,
        'aluminium': 1,
        'rubber': 24,
        'gas': 20,
        'reserves': 22,
        'current account': 2,
        'gdp gnp': 10,
        'steel': 15,
        'orange': 21,
        'retail': 23,
        'producer price wholesale': 30,
        'oils and fats tax': 29,
    }

    train_classes = list(range(15))
    val_classes = list(range(15, 20))
    test_classes = list(range(20, 31))

    return train_classes, val_classes, test_classes, label_dict

def _get_rcv1_classes(args):
    '''
        @return list of classes associated with each split
    '''

    train_classes = [1, 2, 12, 15, 18, 20, 22, 25, 27, 32, 33, 34, 38, 39,
                     40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
                     54, 55, 56, 57, 58, 59, 60, 61, 66]
    val_classes = [5, 24, 26, 28, 29, 31, 35, 23, 67, 36]
    test_classes = [0, 3, 4, 6, 7, 8, 9, 10, 11, 13, 14, 16, 17, 19, 21, 30, 37,
                    62, 63, 64, 65, 68, 69, 70]

    return train_classes, val_classes, test_classes


def _load_json(path):
    '''
        load data file
        @param path: str, path to the data file
        @return data: list of examples
    '''
    label = {}
    text_len = []
    with open(path, 'r', errors='ignore') as f:
        data = []
        for line in f:
            row = json.loads(line)

            # count the number of examples per label
            if int(row['label']) not in label:
                label[int(row['label'])] = 1
            else:
                label[int(row['label'])] += 1

            item = {
                'label': int(row['label']),
                'text': row['text'][:500]  # truncate the text to 500 tokens
            }

            text_len.append(len(row['text']))

            keys = ['head', 'tail', 'ebd_id']
            for k in keys:
                if k in row:
                    item[k] = row[k]

            data.append(item)

        tprint('Class balance:')

        print(label)

        tprint('Avg len: {}'.format(sum(text_len) / (len(text_len))))

        return data


def _read_words(data, class_name_words):
    '''
        Count the occurrences of all words
        @param data: list of examples
        @return words: list of words (with duplicates)
    '''
    words = []
    for example in data:
        words += example['text']
    for example in class_name_words:
        words += example
    return words


def _meta_split(all_data, train_classes, val_classes, test_classes):
    '''
        Split the dataset according to the specified train_classes, val_classes
        and test_classes

        @param all_data: list of examples (dictionaries)
        @param train_classes: list of int
        @param val_classes: list of int
        @param test_classes: list of int

        @return train_data: list of examples
        @return val_data: list of examples
        @return test_data: list of examples
    '''
    train_data, val_data, test_data = [], [], []

    for example in all_data:
        if example['label'] in train_classes:
            train_data.append(example)
        if example['label'] in val_classes:
            val_data.append(example)
        if example['label'] in test_classes:
            test_data.append(example)

    return train_data, val_data, test_data


def _del_by_idx(array_list, idx, axis):
    '''
        Delete the specified index for each array in the array_lists

        @params: array_list: list of np arrays
        @params: idx: list of int
        @params: axis: int

        @return: res: tuple of pruned np arrays
    '''
    if type(array_list) is not list:
        array_list = [array_list]

    # modified to perform operations in place
    for i, array in enumerate(array_list):
        array_list[i] = np.delete(array, idx, axis)

    if len(array_list) == 1:
        return array_list[0]
    else:
        return array_list


def _data_to_nparray(data, vocab, args):
    '''
        Convert the data into a dictionary of np arrays for speed.
    '''
    doc_label = np.array([x['label'] for x in data], dtype=np.int64)

    raw = np.array([e['text'] for e in data], dtype=object)


    # compute the max text length
    text_len = np.array([len(e['text']) for e in data])
    max_text_len = max(text_len)

    # initialize the big numpy array by <pad>
    text = vocab.stoi['<pad>'] * np.ones([len(data), max_text_len],
                                     dtype=np.int64)

    del_idx = []
    # convert each token to its corresponding id
    for i in range(len(data)):
        text[i, :len(data[i]['text'])] = [
                vocab.stoi[x] if x in vocab.stoi else vocab.stoi['<unk>']
                for x in data[i]['text']]

        # filter out document with only unk and pad
        if np.max(text[i]) < 2:
            del_idx.append(i)

    vocab_size = vocab.vectors.size()[0]

    text_len, text, doc_label, raw = _del_by_idx(
            [text_len, text, doc_label, raw], del_idx, 0)

    new_data = {
        'text': text,
        'text_len': text_len,
        'label': doc_label,
        'raw': raw,
        'vocab_size': vocab_size,
    }

    return new_data


def _split_dataset(data, finetune_split):
    """
        split the data into train and val (maintain the balance between classes)
        @return data_train, data_val
    """

    # separate train and val data
    # used for fine tune
    data_train, data_val = defaultdict(list), defaultdict(list)

    # sort each matrix by ascending label order for each searching
    idx = np.argsort(data['label'], kind="stable")

    non_idx_keys = ['vocab_size', 'classes2id', 'is_train', 'n_t', 'n_d', 'avg_ebd']
    for k, v in data.items():
        if k not in non_idx_keys:
            data[k] = v[idx]

    # loop through classes in ascending order
    classes, counts = np.unique(data['label'], return_counts=True)
    start = 0
    for label, n in zip(classes, counts):
        mid = start + int(finetune_split * n)  # split between train/val
        end = start + n  # split between this/next class

        for k, v in data.items():
            if k not in non_idx_keys:
                data_train[k].append(v[start:mid])
                data_val[k].append(v[mid:end])

        start = end  # advance to next class

    # convert back to np arrays
    for k, v in data.items():
        if k not in non_idx_keys:
            data_train[k] = np.concatenate(data_train[k], axis=0)
            data_val[k] = np.concatenate(data_val[k], axis=0)

    return data_train, data_val


def load_dataset(args):
    if args.dataset == '20newsgroup':
        train_classes, val_classes, test_classes, label_dict = _get_20newsgroup_classes(args)
    elif args.dataset == 'amazon':
        train_classes, val_classes, test_classes, label_dict = _get_amazon_classes(args)
    elif args.dataset == 'fewrel':
        train_classes, val_classes, test_classes, label_dict = _get_fewrel_classes(args)
    elif args.dataset == 'huffpost':
        train_classes, val_classes, test_classes, label_dict = _get_huffpost_classes(args)
    elif args.dataset == 'reuters':
        train_classes, val_classes, test_classes, label_dict = _get_reuters_classes(args)
    elif args.dataset == 'rcv1':
        train_classes, val_classes, test_classes = _get_rcv1_classes(args)
    else:
        raise ValueError(
            'args.dataset should be one of'
            '[20newsgroup, amazon, fewrel, huffpost, reuters, rcv1]')

    assert(len(train_classes) == args.n_train_class)
    assert(len(val_classes) == args.n_val_class)
    assert(len(test_classes) == args.n_test_class)

    tprint('Loading data')
    all_data = _load_json(args.data_path)
    class_names = []
    class_name_words = []
    for ld in label_dict:
        class_name_dic = {}
        class_name_dic['label'] = label_dict[ld]
        class_name_dic['text'] = ld.lower().split()
        class_names.append(class_name_dic)
        class_name_words.append(class_name_dic['text'])

    tprint('Loading word vectors')

    vectors = Vectors(args.word_vector, cache=args.wv_path)
    vocab = Vocab(collections.Counter(_read_words(all_data, class_name_words)), vectors=vectors,
                  specials=['<pad>', '<unk>'], min_freq=5)

    # print word embedding statistics
    wv_size = vocab.vectors.size()
    tprint('Total num. of words: {}, word vector dimension: {}'.format(
        wv_size[0],
        wv_size[1]))

    num_oov = wv_size[0] - torch.nonzero(
            torch.sum(torch.abs(vocab.vectors), dim=1)).size()[0]
    tprint(('Num. of out-of-vocabulary words'
           '(they are initialized to zeros): {}').format(num_oov))

    # Split into meta-train, meta-val, meta-test data
    train_data, val_data, test_data = _meta_split(
            all_data, train_classes, val_classes, test_classes)
    tprint('#train {}, #val {}, #test {}'.format(
        len(train_data), len(val_data), len(test_data)))

    # Convert everything into np array for fast data loading
    class_names = _data_to_nparray(class_names, vocab, args)
    train_data = _data_to_nparray(train_data, vocab, args)
    val_data = _data_to_nparray(val_data, vocab, args)
    test_data = _data_to_nparray(test_data, vocab, args)

    train_data['is_train'] = True
    val_data['is_train'] = True
    test_data['is_train'] = True
    # this tag is used for distinguishing train/val/test when creating source pool

    return train_data, val_data, test_data, class_names, vocab
