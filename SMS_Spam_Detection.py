# Wrote by Manhua
# Begin Date: Nov, 17, 2018
# Last Update: Nov, 26, 2018

# -*- coding: utf-8 -*-

import pandas
import string
import math
import datetime


# read training data set
# the result of this function is a data set of n(the number of messages) * 2 (class, message)
def read_dataset(filename):
    # read data into dataframe
    training_data = pandas.read_csv(filename, encoding="ISO-8859-1")
    # rename the training data set
    training_data.rename(columns={"v1": "Class", "v2": "Message"}, inplace=True)
    # default NaN value as ''
    training_data = training_data.fillna('')
    # compile the messages from different columns
    if training_data.shape[1] > 2:
        for column in training_data:
            if column != "Class" and column != "Message":
                training_data["Message"] += "," + training_data[column]

    training_data.drop(columns=(set(training_data.columns.values.tolist()).difference(set(["Class", "Message"]))),
                       axis=1, inplace=True)

    return training_data


# cut the message into words in list, and save it as a dic:
def split_message_to_words(message):
    irregular_verb = {'got': 'get', 'text': 'txt', 'said': 'say', 'won': 'win', 'sent': 'send', 'became': 'become',
                      'ran': 'run', 'threw': 'throw', 'thrown': 'throw', 'blew': 'blew', 'drew': 'drew', 'grew': 'grew',
                      'knew': 'knew', 'began': 'begin', 'drank': 'drink', 'sang': 'sing', 'swam': 'swim', 'rang': 'ring',
                      'wore': 'wear', 'forgot': 'forget', 'spoke': 'speak', 'froze': 'freeze', 'chose': 'choose',
                      'drove': 'drive', 'mistook': 'mistake', 'shook': 'shake', 'ate': 'ate', 'forbade': 'forbade',
                      'gave': 'gave', 'rode': 'rode', 'saw': 'saw', 'wrote': 'wrote', 'fell': 'fall', 'broken': 'break',
                      'forgave': 'forgive', 'went': 'go', 'took': 'take', 'brought': 'bring', 'bought': 'buy',
                      'fought': 'fight', 'thought': 'think', 'sought': 'seek', 'caught': 'catch', 'taught': 'teach',
                      'fed': 'feed', 'held': 'hold', 'sat': 'sit', 'paid': 'pay', 'sold': 'sell', 'found': 'find',
                      'led': 'lead', 'felt': 'feel', 'kept': 'sleep', 'slept': 'keep', 'swept': 'swept', 'smelt': 'smelt',
                      'built': 'built', 'heard': 'heard', 'made': 'make', 'meant': 'mean', 'spent': 'spend', 'lent': 'lend',
                      'lost': 'lose', 'told': 'tell', 'stood': 'stand', 'misunderstood': 'misunderstood', 'shot': 'shoot',
                      'understood': 'understood', 'pls': 'please'}

    for p in string.punctuation:
        message = message.replace(p, ' ')

    # split message into words into words with ' '
    words_in_message = message.lower().split(' ')

    # get rid of all ' ' in the list
    while ' ' in words_in_message:
        words_in_message.remove(' ')

    # get rid of all '' in the list
    while '' in words_in_message:
        words_in_message.remove('')

    # temporal transformation
    j = 0
    for word in words_in_message:
        if word.isalnum():                                    # ignore those with strange characters
            if word in irregular_verb:
                words_in_message[j] = irregular_verb[word]
            elif word.endswith("ed"):
                words_in_message[j] = word.rstrip("ed")
            elif word.endswith("ing"):
                words_in_message[j] = word.rstrip("ing")
            elif word.endswith("es"):
                words_in_message[j] = word.rstrip("es")
        else:
            words_in_message.remove(word)

        j += 1

    return words_in_message


# check if a message has links
def does_have_links(message):
    link_words = ["www", "http"]
    has_link = 0
    for word in link_words:
        if message.find(word) != -1:
            has_link = 1
            break

    return has_link


# check if messages have spam words
def does_have_spam_words(word_list):
    spam_words_list = ["ham", "reply", "freemsg" "cash", "txt", "text",
                       "msg", "message", "sms", "pobox", "won", "win",
                       "prize", "award", "awarded", "awarded", "contact",
                       "delivery", "guaranteed", "trip", "entry", "urgent",
                       "redeem", "ringtone", "mnths", "winner"]

    spam_count = 0
    for word in word_list:
        if word in spam_words_list:
            spam_count += 1

    return spam_count


# count unique words
def count_vector(number_of_txt, feature_data, message_data):
    stopwords_list = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'back',
                      'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'cant', 'com',
                      'could', 'did', 'didn', 'do', 'does', 'doing', 'don', 'dont', 'down', 'during', 'each', 'even', 'few', 'for',
                      'from', 'further', 'had', 'has', 'have', 'having', 'he', 'hed', 'hell', 'her', 'here', 'heres', 'hers',
                      'herself', 'hes', 'him', 'himself', 'his', 'how', 'hows', 'i', 'id', 'if', 'ill', 'im', 'in', 'into',
                      'is', 'it', 'its', 'itself', 'ive', 'just', "let's", 'll', 'me', 'more', 'most', 'my', 'myself', 'net',
                      'no', 'nor', 'not', 'now', 'of', 'off', 'ok', 'on', 'once', 'only', 'or', 'other', 'ought', 'our',
                      'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'she', 'shed', 'shell', 'shes', 'should',
                      'so', 'some', 'such', 't', 'than', 'that', "thats", 'the', 'their', 'theirs', 'them', 'themselves',
                      'then', 'there', 'theres', 'these', 'they', 'theyd', 'theyll', 'theyre', 'theyve', 'this', 'those',
                      'through', 'to', 'too', 'u', 'under', 'until', 'up', 'ur', 'us', 've', 'very', 'was', 'we', "wed",
                      "well", "were", "weve", 'were', 'what', 'whats', 'when', "when's", 'where', 'wheres', 'which',
                      'while', 'who', 'whom', 'whos', 'whose', 'why', 'whys', 'will', 'with', 'would', 'www', 'you',
                      'youd', 'youll', 'your', 'youre', 'yours', 'yourself', 'yourselves', 'youve', 'th', "wont"]

    # initialize unique word list
    unique_words_dic = dict()               # key words are unique words, element are number_of_txt_with_word

    # count_vector
    i = 0
    for message in message_data:
        for word in message:
            # check if only includes word (all alphabetic letters)
            if word.isalpha() and len(word) > 1:
                # check if the word is stop word, if it is, pass it; if not, continue
                if not (word in stopwords_list):
                    # check if the word already exists
                    if word in unique_words_dic:              # if the word exists, count the word
                        feature_data.at[i, word] = message.count(word)
                        unique_words_dic[word] += 1
                    else:                                  # if the word is a new unique word
                        # create a column with column name as word
                        unique_words_dic[word] = 1
                        feature_data[word] = 0
                        # count the number of this word appear in the messagea
                        feature_data.at[i, word] = message.count(word)
        i += 1

    filter_str = ''
    filter_str = [filter_str+key for key in unique_words_dic if unique_words_dic[key] <= number_of_txt*0.001]
    feature_data.drop(columns=filter_str, axis=1, inplace=True)

    return feature_data


# count the number of words in message
def count_words(message_list):
    return len(message_list)


# calculate TF/IDF vector
def tf_idf_vector(column, number_of_txt, message_data):
    number_of_txt_with_word = number_of_txt - column.value_counts().loc[0]
    tf_idf = (column / message_data) * math.log(number_of_txt/number_of_txt_with_word)
    return round(tf_idf, 4)


# calculate the message length
def message_len(message):
    length = len(message)
    return length


# check if the message includes a phone number:
def does_have_phone_number(words_in_message):
    flag = 0
    for word in words_in_message:
        if word.rstrip(' ').isdigit() and len(word) >= 5:
            flag = 1
            break

    return flag


# out put the feature into .csv file
def output_feature_data(feature_data, filename):
    feature_data.to_csv(filename, index=False)
    return 0


# # if/else block to predict a message as spam or ham
def predict_spam(does_have_link, spam_words_count, length, phone_number):
    categorize = ""
    if phone_number > 0:
        categorize = "spam"
    elif phone_number <= 0:
        if spam_words_count <= 0:
            categorize = "ham"
        elif spam_words_count > 0:
            if spam_words_count > 1:
                categorize = "spam"
            elif spam_words_count <= 1:
                if length <= 137:
                    categorize = "ham"
                elif length > 137:
                    if does_have_link <= 0:
                        categorize = "ham"
                    elif does_have_link > 0:
                        categorize = "spam"

    return categorize


def compute_feature(filename):
    # generate feature data for decision tree
    training_data = read_dataset(filename)
    number_of_txt = len(training_data["Class"])
    training_data["words_in_message"] = training_data["Message"].apply(split_message_to_words)
    training_data["count_words"] = training_data["Message"].apply(count_words)

    feature_basic = pandas.DataFrame(columns=["Class"])
    feature_basic["Class"] = training_data["Class"].copy()
    feature_basic["doesHaveLink"] = training_data["Message"].apply(does_have_links)
    feature_basic["doseHaveSpamWords"] = training_data["words_in_message"].apply(does_have_spam_words)

    # compute count_vector feature
    feature_count_vector = feature_basic.copy()
    feature_count_vector = count_vector(number_of_txt, feature_count_vector, training_data["words_in_message"])

    # compute tf_idf feature
    feature_tf_idf = feature_count_vector.copy()
    feature_tf_idf.iloc[:, 3:feature_count_vector.shape[1]] = \
        feature_count_vector.iloc[:, 3:feature_count_vector.shape[1]].apply(tf_idf_vector, axis=0, args=(number_of_txt, training_data["count_words"]))

    # two extra features
    feature_2_extra = feature_basic.copy()
    feature_2_extra["message_len"] = training_data["Message"].apply(message_len)
    feature_2_extra["doesHavePhoneNumber"] = training_data["words_in_message"].apply(does_have_phone_number)

    # output the feature data into csv file
    output_feature_data(feature_count_vector, "feature_CountVector.csv")
    output_feature_data(feature_tf_idf, "feature_TF_IDF.csv")
    output_feature_data(feature_2_extra, "feature_2_extra.csv")


def main():
    # compute_feature("spam.csv")

    # predict the message
    message = input("Please input a message: ")
    words_in_message = message.lower().split(' ')
    categorize = predict_spam(does_have_links(message), does_have_spam_words(words_in_message),
                              message_len(message), does_have_phone_number(words_in_message))
    print("This is a " + categorize + '.')


main()
