import csv
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# import seaborn as sns
from sklearn.model_selection import train_test_split

from utils import Timer


# ---------------------------------------------------------------------------
# constants (paths)


data_cross_path = "data_raw/argmining/cross/{}.csv"
data_within_path = "data_raw/argmining/within/{}.csv"
new_within_test = "data_raw/argmining/within/test.csv"
data_ground_truth_path = "data_raw/argmining/ground-truth/{}-topics-ground-truth-subset.csv"

fn_art_eval = "data/artificial_evalset/artificial_evalset.tsv"

names_columns_X = ["argument1", "argument2", "argument1_id", "argument2_id", "topic"]
names_columns_X_arteval = ["argument1", "argument2", "tag"]
names_columns_y = ["is_same_side"]


# ---------------------------------------------------------------------------
# Load data webis


def load_official_data_cross():
    with Timer("read S3C cross train/dev"):
        cross_traindev_df = pd.read_csv(
            data_cross_path.format("training"),
            quotechar='"',
            quoting=csv.QUOTE_ALL,
            encoding="utf-8",
            escapechar="\\",
            doublequote=False,
            index_col="id",
        )
        cross_test_df = pd.read_csv(data_cross_path.format("test"), index_col="id")

    return cross_traindev_df, cross_test_df


def load_official_data_within():
    with Timer("read S3C within train/dev"):
        within_traindev_df = pd.read_csv(
            data_within_path.format("training"),
            quotechar='"',
            quoting=csv.QUOTE_ALL,
            encoding="utf-8",
            escapechar="\\",
            doublequote=False,
            index_col="id",
        )
        # within_test_df = pd.read_csv(data_within_path.format('test'),
        #                              quotechar='"',
        #                              quoting=csv.QUOTE_ALL,
        #                              encoding='utf-8',
        #                              escapechar='\\',
        #                              doublequote=True,  # <-- change, "" as quote escape in text?
        #                              index_col='id')
        # reformatted:
        # within_test_df = pd.read_csv(data_within_path.format("test"), index_col="id")
        # updated test set:
        new_within_test_df = pd.read_csv(new_within_test, index_col="id")

    return within_traindev_df, new_within_test_df


def load_official_data(task="within"):
    if task == "within":
        return load_official_data_within()
    if task == "cross":
        return load_official_data_cross()
    raise Exception("Unknown dataset!")


def load_ground_truth_data(task="within"):
    with Timer("read S3C within train/dev"):
        return pd.read_csv(data_ground_truth_path.format(task), index_col="id")


# ---------------------------------------------------------------------------


def load_distinct_df_raw(name="within"):
    fn = "data/distinct_sets/{name}/{name}_{mode}_arg_pickle.pkl"
    fn_train = fn.format(mode="train", name=name)
    fn_dev = fn.format(mode="dev", name=name)

    with open(fn_train, "rb") as fp:
        train_df = pickle.load(fp)
    with open(fn_dev, "rb") as fp:
        dev_df = pickle.load(fp)

    with Timer("tag distinct {} train/dev".format(name)):
        train_df = add_tag(train_df)
        dev_df = add_tag(dev_df)

    # return pd.concat([train_df, dev_df])
    return train_df, dev_df


def load_distinct_data(name="within", train_df=None, dev_df=None):
    if not train_df or not dev_df:
        train_df, dev_df = load_distinct_df_raw(name)

    X_train = train_df[names_columns_X]
    y_train = train_df[names_columns_y]
    X_dev = dev_df[names_columns_X]
    y_dev = dev_df[names_columns_y]

    return X_train, X_dev, y_train, y_dev


# ---------------------------------------------------------------------------


def load_artificial_dataset():
    with Timer("read artificial evalset"):
        artificial_evalset_df = pd.DataFrame.from_csv(
            fn_art_eval, sep="\t", index_col=None
        )

        new_cols = artificial_evalset_df.columns.to_list()
        new_cols[2] = "type"
        artificial_evalset_df.columns = new_cols

        def fix_cols(row):
            row["argument1_id"] = row["arg_id"]
            row["argument2_id"] = "{}-{}".format(row["arg_id"], row["type"])
            row["topic"] = "gay marriage"
            return row

        artificial_evalset_df = artificial_evalset_df.apply(fix_cols, axis=1)

    with Timer("tag artificial evalset"):
        artificial_evalset_df = add_tag(artificial_evalset_df)

    return artificial_evalset_df


# ---------------------------------------------------------------------------


def add_tag(df):
    # Adding a tag for the topics in focus: "gay marriage" and "abortion"
    def _add_tag(row):
        title = row["topic"].lower().strip()
        if "abortion" in title:
            row["tag"] = "abortion"
        elif "gay marriage" in title:
            row["tag"] = "gay marriage"
        else:
            row["tag"] = "NA"
        return row

    return df.progress_apply(_add_tag, axis=1)


def load_and_prepare_official_data(task="within"):
    traindev_df, test_df = load_official_data(task=task)

    with Timer("tag {} train/dev".format(task)):
        traindev_df = add_tag(traindev_df)
        test_df = add_tag(test_df)

    return traindev_df, test_df


# ---------------------------------------------------------------------------
# Split train/dev


# train dev set - 70% 30%
def get_train_test_sets(df, ratio=0.30, random_state=42):
    X = df[names_columns_X]
    y = df[names_columns_y]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ratio, random_state=random_state, shuffle=True
    )
    return X_train, X_test, y_train, y_test


def split_within_by_topic(within_df):
    groups = within_df.groupby(["tag"])
    abortion_df = groups.get_group("abortion")
    gay_marriage_df = groups.get_group("gay marriage")

    X_abortion = abortion_df[names_columns_X]
    y_abortion = abortion_df[names_columns_y]
    X_gay_marriage = gay_marriage_df[names_columns_X]
    y_gay_marriage = gay_marriage_df[names_columns_y]

    return X_abortion, X_gay_marriage, y_abortion, y_gay_marriage


# ---------------------------------------------------------------------------


def get_bert_tokenizer(
    model_name="bert_12_768_12", dataset_name="book_corpus_wiki_en_uncased"
):
    import gluonnlp as nlp
    import mxnet as mx

    ctx = mx.cpu()

    _, vocabulary = nlp.model.get_model(
        model_name,
        dataset_name=dataset_name,
        pretrained=True,
        ctx=ctx,
        use_pooler=True,
        use_decoder=False,
        use_classifier=False,
    )
    return nlp.data.BERTTokenizer(vocabulary, lower=True)


def add_bert_tokens(df):
    tokenizer = get_bert_tokenizer()

    # tokenizer from BERT
    def tokenize_arguments(row):
        # tokenize
        row["argument1_tokens"] = tokenizer(row["argument1"])
        row["argument2_tokens"] = tokenizer(row["argument2"])

        # count tokens
        row["argument1_len"] = len(row["argument1_tokens"])
        row["argument2_len"] = len(row["argument2_tokens"])
        # token number diff
        row["argument12_len_sum"] = row["argument1_len"] + row["argument2_len"]
        row["argument12_len_sum_half"] = row["argument12_len_sum"] / 2
        row["argument12_len_diff"] = row["argument1_len"] - row["argument2_len"]
        row["argument12_len_diff_abs"] = np.abs(row["argument12_len_diff"])
        return row

    return df.progress_apply(tokenize_arguments, axis=1)


def add_raw_lengths(df):
    def compute_arg_len(row):
        row["argument1_len"] = len(row["argument1"])
        row["argument2_len"] = len(row["argument2"])
        row["argument12_len_sum"] = row["argument1_len"] + row["argument2_len"]
        row["argument12_len_sum_half"] = row["argument12_len_sum"] / 2
        row["argument12_len_diff"] = row["argument1_len"] - row["argument2_len"]
        row["argument12_len_diff_abs"] = np.abs(row["argument12_len_diff"])
        return row

    return df.progress_apply(compute_arg_len, axis=1)


def add_sentence_segments(df):
    from nltk.tokenize import sent_tokenize, word_tokenize

    # nltk.download('punct')

    def sentenize_arguments(row):
        # sentence segment
        row["argument1_sentences"] = sent_tokenize(row["argument1"])
        row["argument2_sentences"] = sent_tokenize(row["argument2"])

        # count tokens
        row["argument1_sent_num"] = len(row["argument1_sentences"])
        row["argument2_sent_num"] = len(row["argument2_sentences"])
        # token number diff
        row["argument12_sent_num_sum"] = (
            row["argument1_sent_num"] + row["argument2_sent_num"]
        )
        row["argument12_sent_num_sum_half"] = row["argument12_sent_num_sum"] / 2
        row["argument12_sent_num_diff"] = (
            row["argument1_sent_num"] - row["argument2_sent_num"]
        )
        row["argument12_sent_num_diff_abs"] = np.abs(row["argument12_sent_num_diff"])
        return row

    return df.progress_apply(sentenize_arguments, axis=1)


# ---------------------------------------------------------------------------


def get_overview(df, task="same-side", description=None, class_name="is_same_side"):
    # Total instance numbers
    total = len(df)
    print("Task: ", task)
    if description:
        print(description)
    print("=" * 40, "\n")

    print("Total instances: ", total, "\n")

    print("For each topic:")
    for tag, tag_df in df.groupby(["tag"]):
        print(tag, ": ", len(tag_df), " instances")
        print("")
        print("\t\tUnique argument1:", len(tag_df["argument1"].unique()))
        print("\t\tUnique argument2:", len(tag_df["argument2"].unique()))
        arguments = np.concatenate(
            [tag_df["argument1"].values, tag_df["argument2"].values]
        )
        print("\t\tUnique total arguments:", len(set(list(arguments))), "\n")
        if class_name in df.columns:
            for is_same_side, side_df in tag_df.groupby([class_name]):
                print("\t\t", is_same_side, ": ", len(side_df), " instances")
    print("\n")

    if class_name in df.columns:
        print("For each class value:")
        for class_value, class_df in df.groupby([class_name]):
            print(class_value, ": ", len(class_df), " instances")
            print("\t\tUnique argument1:", len(class_df["argument1"].unique()))
            print("\t\tUnique argument2:", len(class_df["argument2"].unique()))
            arguments = np.concatenate(
                [class_df["argument1"].values, class_df["argument2"].values]
            )
            print("\t\tUnique total arguments:", len(set(list(arguments))), "\n")
        print("\n")

    print("Unique argument1:", len(df["argument1"].unique()))
    print("Unique argument2:", len(df["argument2"].unique()))
    arguments = df["argument1"].values
    arguments = np.concatenate([arguments, df["argument2"].values])

    print("Unique total arguments:", len(set(list(arguments))), "\n")

    if "argument1_len" in df.columns:
        print("-" * 40, "\n")

        arguments_length_lst = [x for x in df["argument1_len"].values]
        arguments_length_lst.extend([x for x in df["argument2_len"].values])
        print("Words:")
        print("\tshortest argument:", min(arguments_length_lst), " words")
        print("\tlongest argument:", max(arguments_length_lst), " words")
        print("\targument average length:", np.mean(arguments_length_lst), " words")

    if "argument1_sent_num" in df.columns:
        arguments_sent_length_lst = [x for x in df["argument1_sent_num"].values]
        arguments_sent_length_lst.extend([x for x in df["argument2_sent_num"].values])
        print("Sentences:")
        print("\tshortest argument:", min(arguments_sent_length_lst), " sentences")
        print("\tlongest argument:", max(arguments_sent_length_lst), " sentences")
        print(
            "\targument average length:",
            np.mean(arguments_sent_length_lst),
            " sentences",
        )


def plot_lengths(df, slicen=None, abs_diff=True, title=None):
    if df is None or "argument1_len" not in df.columns:
        print("no lengths to plot")
        return

    arg1_lens = df["argument1_len"]
    arg2_lens = df["argument2_len"]
    arg_diff_len = df["argument12_len_diff"]

    if abs_diff:
        arg_diff_len = np.abs(arg_diff_len)

    if slicen is not None:
        arg1_lens = arg1_lens[slicen]
        arg2_lens = arg2_lens[slicen]
        arg_diff_len = arg_diff_len[slicen]

    x = np.arange(len(arg1_lens))  # arange/linspace

    plt.subplot(2, 1, 1)
    plt.plot(x, arg1_lens, label="argument1")  # Linie: '-', 'o-', '.-'
    plt.plot(x, arg2_lens, label="argument2")  # Linie: '-', 'o-', '.-'
    plt.legend()
    plt.title("Lengths of arguments" if not title else title)
    plt.ylabel("Lengths of arguments 1 and 2")

    plt.subplot(2, 1, 2)
    plt.plot(x, arg_diff_len)
    plt.xlabel("Index")
    plt.ylabel("Differences")

    plt.show()


# ---------------------------------------------------------------------------
