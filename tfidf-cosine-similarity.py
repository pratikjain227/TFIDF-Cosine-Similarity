import pandas as pd
import operator
import math


def read_and_get_data(filename):
    word_list = []
    fr = open(filename, 'r')
    text = fr.read()
    fr.close()
    text = text.lower().split()
    for each_word in text:
        word_list.append(each_word)
    clean_word_list = []
    for each_word in word_list:
        symbols = "~!@#$%^&*()_+`-=[]\;',./{}|:\"<>?"
        for i in range(0, len(symbols)):
            each_word = each_word.replace(symbols[i], "")
        if len(each_word) > 0:
            clean_word_list.append(each_word)
    return clean_word_list


def calculate_word_freq(clean_word_list):
    word_freq = {}
    for each_word in clean_word_list:
        if each_word in word_freq:
            word_freq[each_word] += 1
        else:
            word_freq[each_word] = 1
    return word_freq


# tf = no of times word appears in the doc / total no of words in that doc
def compute_tf(word_freq, clean_word_list):
    word_tf = {}
    for word, count in word_freq.items():
        temp = len(clean_word_list)
        word_tf[word] = count/float(temp)
    return word_tf


# idf = log( no of docs / no of docs containing the word w )
def compute_idf(doc_list):
    word_idf = {}
    no_of_docs = len(doc_list)
    temp = doc_list[0].copy()
    temp.update(doc_list[1])
    word_idf = dict.fromkeys(temp.keys(), 0)
    for each_doc in doc_list:
        for each_word, val in each_doc.items():
            if val > 0:
                word_idf[each_word] += 1
    for each_word, val in word_idf.items():
        word_idf[each_word] = math.log(no_of_docs / float(val))
    return word_idf


# tfidf = tf * idf
def compute_tfidf(tf, idf):
    tfidf = {}
    for word, val in tf.items():
        tfidf[word] = val*idf[word]
    return tfidf


def ignore_word(word_freq):
    ignore_list = []
    fr = open("ignore_text.txt", 'r')
    ignore_text = fr.read()
    fr.close()
    ignore_text = ignore_text.lower().split()
    for word in ignore_text:
        ignore_list.append(word)
    for word in ignore_list:
        word_freq[word] = 0
    return word_freq


def cosine_similarity(word_freq1, word_freq2):
    numerator = 0
    sum2 = 0
    sum3 = 0
    for word, val in word_freq1.items():
        if word not in word_freq2:
            word_freq2[word] = 0
    for word, val in word_freq2.items():
        if word not in word_freq1:
            word_freq1[word] = 0
    for word, val in word_freq1.items():
        numerator += (val*word_freq2[word])
    for word, val in word_freq1.items():
        sum2 += (val*val)
    for word, val in word_freq2.items():
        sum3 += (val*val)
    return (numerator/(math.sqrt(sum2)*(math.sqrt(sum3))))


def idf_modfied_cosine_similarity(tf1, tf2, idf, tfidf1, tfidf2, word_freq1, word_freq2):
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for each_word, val in idf.items():
        if each_word in tf1:
            pass
        else:
            tf1[each_word] = 0
    for each_word, val in idf.items():
        if each_word in tf2:
            pass
        else:
            tf2[each_word] = 0
    print(len(tf1))
    print(len(tf2))
    print(len(idf))
    print(len(tfidf1))
    print(len(tfidf2))
    for word, val in idf.items():
        print(" ----------------- ")
        print("tf1 = ", tf1[word])
        print("tf2 = ", tf2[word])
        print("idf = ", idf[word])
        print(" ----------------- ")
        sum1 += (word_freq1[word]*word_freq2[word]*idf[word]*idf[word])
    for word, val in tfidf1.items():
        sum2 += (word_freq1[word]*tfidf1[word])*(word_freq1[word]*tfidf1[word])
    for word, val in tfidf2.items():
        sum3 += (word_freq2[word]*tfidf2[word])*(word_freq2[word]*tfidf2[word])
    sum2 = math.sqrt(sum2)
    sum3 = math.sqrt(sum3)
    print(sum1)
    print(sum2)
    print(sum3)
    cs = sum1/(sum2*sum3)
    return cs

print(" ======== Reading and cleaning data ==========")
clean_word_list1 = read_and_get_data('ml.txt')
clean_word_list2 = read_and_get_data('battle.txt')

print(" ======== Calculating word freq ==========")
word_freq1 = calculate_word_freq(clean_word_list1)
word_freq2 = calculate_word_freq(clean_word_list2)

print(" ======== Calculating tf ==========")
tf1 = compute_tf(word_freq1, clean_word_list1)
tf2 = compute_tf(word_freq2, clean_word_list2)

print(" ======== Calculating idf ==========")
idf = compute_idf([word_freq1, word_freq2])

print(" ======== Calculating tfidf ==========")
tfidf1 = compute_tfidf(tf1, idf)
tfidf2 = compute_tfidf(tf2, idf)

x = pd.DataFrame([tfidf1, tfidf2])
#print(x)

print(" ======== Ignoring unimportant words listed in ignore_text.txt ==========")
word_freq1 = ignore_word(word_freq1)
word_freq2 = ignore_word(word_freq2)

cs2 = cosine_similarity(word_freq1, word_freq2)

cs3 = idf_modfied_cosine_similarity(tf1, tf2, idf, tfidf1, tfidf2, word_freq1, word_freq2)

print(" ======== Calculating cosine similarity ==========")
print(cs2)
print(" ======== Calculating modified cosine similarity ==========")
print(cs3)
