
import nltk
import numpy as np
import os
import pickle as pkl
import spacy

from concurrent.futures import ProcessPoolExecutor
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter

from config_updated import TRAINING_DOCS, TESTING_DOCS, ANNS_TRAIN_DEV, \
    ANNS_TEST, \
    IDS_TRAINING, IDS_DEVELOPMENT, IDS_TESTING, RAW_TRAIN_FILE, \
    RAW_DEV_FILE, RAW_TEST_FILE, MLB_TRAIN_FILE, MLB_DEV_FILE, \
    MLB_TEST_FILE, MLB_FILE, DISCARDED_FILE

LABEL_TYPE = 'mixed'


class DataReader:
    def read_doc(self, fname):
        with open(fname, mode="r", encoding="utf-8", errors="ignore") as rf:
            data = list()
            for line in rf:
                line = line.strip()
                if line:
                    data.append(line)
            return data

    def read_docs(self, train_or_test):
        if train_or_test == "train":
            docs_dir = TRAINING_DOCS
        else:
            docs_dir = TESTING_DOCS

        incompl_count = 0
        for datafile in os.listdir(docs_dir):
            if datafile == "id.txt":
                continue

            # filename is uid
            doc_id = int(datafile.rstrip(".txt"))
            data = self.read_doc(os.path.join(docs_dir, datafile))

            # sanity check: each file must have 6 lines of text as in README
            if len(data) != 6:
                incompl_count += 1

            # use special token to recover each text field (if needed)
            data = "<SECTION>".join(data)

            yield doc_id, data

        # report if incompletes
        if incompl_count > 0:
            print("[INFO] %d docs do not have 6 text lines" % incompl_count)

    def read_anns(self, anns_file):
        with open(anns_file) as rf:
            for line in rf:
                line = line.strip()
                if line:
                    doc_id, icd10_codes = line.split("\t")
                    # sanity check: remove any duplicates, if there
                    yield int(doc_id), set(icd10_codes.split("|"))

    def read_ids(self, ids_file):
        ids_file = ids_file
        ids = set()

        with open(ids_file, "r") as rf:
            for line in rf:
                line = line.strip()
                if line:
                    if line == "id":  # line 242 in train ids
                        continue
                    ids.add(int(line))

        return ids

    def read_data(self, train_test):
        def categorize(val):
            return val - set(ICD_CHAPTERS)

        def read(docs, anns, ids, label_type=''):
            id2anns = {a[0]: a[1] for a in anns if a[0] in ids}

            # list of tuple (doc text, doc id, set of annotations)
            if label_type == 'categories':
                data = list()
                for d in docs:
                    if d[0] in id2anns:
                        categories = categorize(id2anns[d[0]])

                        if categories:
                            data.append((d[1], d[0], categories))
                # data = [(d[1], d[0], categorize(id2anns[d[0]])) for d in docs
                #         if d[0] in id2anns]
            elif label_type == 'mixed':
                data = [(d[1], d[0], id2anns[d[0]]) for d in docs if
                        d[0] in id2anns]

            return data

        if train_test == "train":
            # load training-dev common data
            docs_train_dev = list(self.read_docs("train"))
            anns_train_dev = list(self.read_anns(ANNS_TRAIN_DEV))

            print(
                "[INFO] num of annotations in `anns_train_dev.txt`: %d" % len(
                    anns_train_dev))

            # train data
            ids_train = self.read_ids(IDS_TRAINING)

            data_train = read(docs_train_dev, anns_train_dev, ids_train,
                              label_type=LABEL_TYPE)

            # dev data
            ids_dev = self.read_ids(IDS_DEVELOPMENT)
            data_dev = read(docs_train_dev, anns_train_dev, ids_dev,
                              label_type=LABEL_TYPE)

            return data_train, data_dev

        else:
            docs_test = list(self.read_docs("test"))
            anns_test = list(self.read_anns(ANNS_TEST))

            # test data
            ids_test = self.read_ids(IDS_TESTING)
            data_test = read(docs_test, anns_test, ids_test,
                              label_type=LABEL_TYPE)

            return data_test


def load_pkl_datafile(fname, use_data="de", as_sents=False):
    examples = []
    with open(fname, "rb") as rf:
        data = pkl.load(rf)
        # each item is tuple((doc orig, doc de, doc en [opt]), doc id, binary labels)
        for value, doc_id, one_hot_labels in data:
            if use_data == "orig":
                text = value[0]
                if not as_sents:
                    text = " ".join(text.split("<SECTION>"))
                else:
                    text = text.split("<SECTION>")
            else:
                if use_data == "de":
                    text = value[1]
                else:
                    text = value[2] # en
                if not as_sents:
                    text = " ".join(text.replace("<SENT>", " ").split("<SECTION>"))
                else:
                    text = [s.replace("<SECTION>", "") for s in text.split("<SENT>")]
            examples.append((text, one_hot_labels, doc_id))
    return examples


class TextProcessor:
    def __init__(self, split_compound=False):
        # spacy word tokenizers
        self.word_tokenizers = {
            "de": spacy.load('de_core_news_sm',
                             disable=['tagger', 'parser', 'ner']).tokenizer
        }
        # nltk sent tokenizers, do we need this?
        self.sent_tokenizers = {
            "de": nltk.data.load('tokenizers/punkt/german.pickle').tokenize
        }
        # special tokens
        self.sent_sep_tok = "<SENT>"
        self.section_sep_tok = "<SECTION>"

        self.split_compound = split_compound

    def process_doc(self, doc):
        doc = doc.split(self.section_sep_tok)  # returns each section

        doc_de = list()

        for textfield in doc:
            sents_de = list(self.sents_tokenize(textfield, "de"))
            sents_tokens_de = list()
            for sent in sents_de:
                tokenized_text = " ".join(
                    list(self.words_tokenize(sent, "de")))
                sents_tokens_de.append(tokenized_text)
            sents_tokens_de = self.sent_sep_tok.join(sents_tokens_de)
            doc_de.append(sents_tokens_de)

        doc_de = self.section_sep_tok.join(doc_de)

        return doc, doc_de

    def sents_tokenize(self, text, lang):
        for sent in self.sent_tokenizers[lang](text):
            sent = sent.strip()
            if sent:
                yield sent

    def words_tokenize(self, text, lang):
        for token in self.word_tokenizers[lang](text):
            token = token.text.strip()
            if token:
                yield token

    @staticmethod
    def de_compounds_split(word, t=0.8):
        res = char_split.split_compound(word)[0]
        if res[0] >= t:
            return res[1:]
        else:
            return word

    def process_with_context(self, text_and_context):
        text = text_and_context[0]
        context = text_and_context[1:]
        return tuple([self.process_doc(text)] + list(context))

    def mp_process(self, data, max_workers=1, chunksize=512):
        """
        data : tup(doc id, doc text, labels list)
        """
        ret = list()
        if max_workers <= 1:
            for idx, item in enumerate(data):
                if idx % 100 == 0 and idx != 0:
                    print("[INFO]: {} documents proceesed".format(idx))
                ret.append(self.process_with_context(item))
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                emap = executor.map(self.process_with_context, data,
                                    chunksize=chunksize)
                for idx, result in enumerate(emap):
                    if idx % 100 == 0 and idx != 0:
                        print("[INFO]: {} documents proceesed".format(idx))
                    ret.append(result)
        return ret


def save(fname, data):
    with open(fname, "wb") as wf:
        pkl.dump(data, wf)


def read_train_file(processed_train_file, label_threshold=25):
    with open(processed_train_file, "rb") as rf:
        train_data = pkl.load(rf)

    count_labels = Counter([label for val in train_data for label in val[-1]])
    print("[INFO] top most frequent labels:", count_labels.most_common(10))

    if label_threshold is None:
        discard_labels = set()
    else:
        discard_labels = {k for k, v in count_labels.items() if
                          v < label_threshold}
    temp = []
    for val in train_data:
        val_labels = {label for label in val[-1] if
                      label not in discard_labels}
        if val_labels:
            list(val).append(val_labels)
            temp.append(val)
    if discard_labels:
        print(
            "[INFO] discarded %d labels with counts less than %d; remaining "
            "labels %d"
            % (len(discard_labels), label_threshold,
               len(count_labels) - len(discard_labels))
        )
        print("[INFO] no. of data points removed %d" % (
                    len(train_data) - len(temp)))

    train_data = temp[:]
    mlb = MultiLabelBinarizer()
    temp = [val[-1] for val in train_data]
    labels = mlb.fit_transform(temp)
    train_data = [(val[0], val[1], labels[idx, :]) for idx, val in
                  enumerate(train_data)]
    return train_data, mlb, discard_labels


def read_dev_file(proceesed_dev_file, mlb, discard_labels):
    with open(proceesed_dev_file, "rb") as rf:
        dev_data = pkl.load(rf)

    count_labels = Counter([label for val in dev_data for label in val[-1]])
    print("[INFO] top most frequent labels:", count_labels.most_common(10))
    temp = []
    for val in dev_data:
        # discard any labels and keep only ones seen in training
        # val_labels = {
        #     label for label in val[-1]
        #     if label not in discard_labels and label in set(mlb.classes_)
        # }
        val_labels = {
            label for label in val[-1]
        }
        if val_labels:
            list(val).append(val_labels)
            temp.append(val)
    print("[INFO] no. of data points removed %d" % (len(dev_data) - len(temp)))

    dev_data = temp[:]
    temp = [val[-1] for val in dev_data]
    labels = mlb.transform(temp)
    dev_data = [(val[0], val[1], labels[idx, :]) for idx, val in
                enumerate(dev_data)]
    return dev_data


if __name__ == '__main__':
    '''
        Inital thing for reading the source text files, necessary processing
        and saving it as a initial raw file which has source texts for all 
        train, dev and test sets with its correspoinding labels in it
    '''
    # Do not process again if the files are already present
    if not os.path.exists(RAW_TRAIN_FILE):
        print("In initalization")

        TRANSLATE = False

        dr = DataReader()
        train_data, dev_data = dr.read_data("train")
        test_data = dr.read_data("test")

        tp = TextProcessor()

        train_data = tp.mp_process(train_data)
        dev_data = tp.mp_process(dev_data)
        test_data = tp.mp_process(test_data)

        # save data
        save(RAW_TRAIN_FILE, train_data)
        save(RAW_DEV_FILE, dev_data)
        save(RAW_TEST_FILE, test_data)
    print("Initaliazation done..")

    # end initial here

    #########################################################################
    '''
        read the raw files created above and prepare necessary train, 
        devn and test files with multilabel binarized labels 
    '''
    if not os.path.exists(MLB_TRAIN_FILE):
        print("In MLB")

        train_data, mlb, discard_labels = read_train_file(RAW_TRAIN_FILE)
        dev_data = read_dev_file(RAW_DEV_FILE, mlb, discard_labels)
        test_data = read_dev_file(RAW_TEST_FILE, mlb, discard_labels)

        save(MLB_TRAIN_FILE, train_data)
        save(MLB_DEV_FILE, dev_data)
        save(MLB_TEST_FILE, test_data)

        save(MLB_FILE, mlb)
        save(DISCARDED_FILE, discard_labels)
    print("MLB done..")