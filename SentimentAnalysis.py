import re

import matplotlib.pyplot as plt
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import xlwt


def text_preprocessing(tweet_text):
    cleaned_tweets = tweet_text.lower()

    cleaned_tweets = re.sub(r':\(|:\'\(|T_T|😭|😢|😰|😟|😫|☹|😥', ' sedih ', cleaned_tweets)
    cleaned_tweets = re.sub(r'>:\(|😡|😠|😤|😒', ' kesal ', cleaned_tweets)
    cleaned_tweets = re.sub(r':\)|:D|😀|😁|😃|😄|🙂|😊|😇', ' senang ', cleaned_tweets)
    for i in range(len(list_not_normal_words)):
        i = i - 1
        cleaned_tweets = re.sub(r'\b%s\b' % list_not_normal_words[i], list_normalized_words[i], cleaned_tweets)

    cleaned_tweets = re.sub(r'@\S*', '', cleaned_tweets)
    cleaned_tweets = re.sub(r'#\S*', '', cleaned_tweets)
    cleaned_tweets = re.sub(r'http\S*', '', cleaned_tweets)
    cleaned_tweets = re.sub(r'[0-9_]', '', cleaned_tweets)
    cleaned_tweets = re.sub(r'\W', ' ', cleaned_tweets)
    for stopword in list_stop_words:
        if stopword in cleaned_tweets:
            cleaned_tweets = re.sub(r'\b%s\b' % stopword, '', cleaned_tweets)
    cleaned_tweets = re.sub(r'\b[a-z]\b', '', cleaned_tweets)
    cleaned_tweets = re.sub(' +', ' ', cleaned_tweets)

    cleaned_tweets = stemming.stem(cleaned_tweets)

    return cleaned_tweets.lstrip(' ')


def totalling_sentimen():
    for label in list_label_sentimen:
        if 'Positif' in label:
            total = sum('positif' in text for text in list_output_sentimen)
            list_total_sentimen.append((label, total))
        elif 'Negatif' in label:
            total = sum('negatif' in text for text in list_output_sentimen)
            list_total_sentimen.append((label, total))
        else:
            total = sum('netral' in text for text in list_output_sentimen)
            list_total_sentimen.append((label, total))


def insert_category_label():
    for tanggal in range(28):
        list_label_tanggal.append(tanggal + 1)
        list_category_tanggal.append(tanggal + 1)


def insert_output_tanggal():
    for date_tweet in dates_test:
        date = date_tweet.split('-')[2].split(' ')[0]
        list_output_tanggal.append(int(date))


def totalling_sentimen_tanggal():
    for tanggal in list_category_tanggal:
        list_index_positif = [x for x, y in list_output_tanggal_positif if y == tanggal]
        list_index_negatif = [x for x, y in list_output_tanggal_negatif if y == tanggal]
        list_index_netral = [x for x, y in list_output_tanggal_netral if y == tanggal]
        list_total_tanggal_positif_temp.append((tanggal, len(list_index_positif), list_index_positif))
        list_total_tanggal_negatif_temp.append((tanggal, len(list_index_negatif), list_index_negatif))
        list_total_tanggal_netral.append((tanggal, len(list_index_netral), list_index_netral))


def categorize_kota():
    for tweet in list_cleaned_tweets[500:]:
        for category in list_category_kota:
            if category == list_category_kota[-1]:
                if re.search(r'\b%s\b' % list_category_kota[-1], tweet):
                    list_output_kota.append(category)
                else:
                    list_output_kota.append('Tidak Spesifik')
            elif re.search(r'\b%s\b' % category, tweet):
                list_output_kota.append(category)
                break


def totalling_sentimen_kota():
    for category in list_category_kota:
        category_position = list_category_kota.index(category)
        list_index_positif = [x for x, y in list_output_kota_positif if y in category]
        list_index_negatif = [x for x, y in list_output_kota_negatif if y in category]
        list_index_netral = [x for x, y in list_output_kota_netral if y in category]
        list_total_kota_positif_temp.append(
            (list_label_kota[category_position], len(list_index_positif), list_index_positif))
        list_total_kota_negatif_temp.append(
            (list_label_kota[category_position], len(list_index_negatif), list_index_negatif))
        list_total_kota_netral.append((list_label_kota[category_position], len(list_index_netral), list_index_netral))


def categorize_pelayanan(tweet_text, list_category):
    for category in list_category:
        if category == list_category[-1]:
            for sub_category in category:
                if sub_category == category[-1]:
                    if re.search(sub_category, tweet_text):
                        return list_category.index(category)
                    else:
                        return 404
                elif re.search(sub_category, tweet_text):
                    return list_category.index(category)
        else:
            for sub_category in category:
                if re.search(sub_category, tweet_text):
                    return list_category.index(category)


def categorize_sentimen_pelayanan(list_output_sentimen, list_output_senti_cs, list_output_senti_galian,
                                  list_output_senti_kabel, list_output_senti_padam, list_output_senti_token,
                                  list_output_senti_gardu):
    for index, sentimen in list_output_sentimen:
        tweet = list_cleaned_tweets_testing[index]
        pelayanan = categorize_pelayanan(tweet, list_category_pelayanan)
        if pelayanan == 0:
            list_output_senti_cs.append(index)
        elif pelayanan == 1:
            list_output_senti_galian.append(index)
        elif pelayanan == 2:
            list_output_senti_kabel.append(index)
        elif pelayanan == 3:
            list_output_senti_padam.append(index)
        elif pelayanan == 4:
            list_output_senti_token.append(index)
        elif pelayanan == 5:
            list_output_senti_gardu.append(index)


def categorize_sentimen_pelayanan_daerah(list_output_sentimen_pelayanan, list_output_sentimen_pelayanan_kelurahan,
                                         list_output_sentimen_pelayanan_kecamatan):
    for index in list_output_sentimen_pelayanan:
        tweet = list_cleaned_tweets_testing[index]
        for daerah in list_daerah:
            jak = daerah.split(' | ')
            kelurahan = jak[0]
            kecamatan = jak[1]
            if re.search(kelurahan, tweet):
                list_output_sentimen_pelayanan_kelurahan.append((index, kelurahan))
                list_output_sentimen_pelayanan_kecamatan.append((index, kecamatan))
                break
            elif re.search(kecamatan, tweet):
                list_output_sentimen_pelayanan_kecamatan.append((index, kecamatan))
                break


def insert_output_pelayanan():
    for tweet in list_cleaned_tweets_testing:
        pelayanan = categorize_pelayanan(tweet, list_category_pelayanan)
        list_output_pelayanan.append(pelayanan)


def totalling_sentimen_pelayanan():
    for category in list_category_pelayanan:
        category_position = list_category_pelayanan.index(category)
        list_index_positif = [x for x, y in list_output_pelayanan_positif if y == category_position]
        list_index_negatif = [x for x, y in list_output_pelayanan_negatif if y == category_position]
        list_index_netral = [x for x, y in list_output_pelayanan_netral if y == category_position]
        list_total_pelayanan_positif_temp.append((list_label_pelayanan[category_position], len(list_index_positif),
                                                  list_index_positif))
        list_total_pelayanan_negatif_temp.append((list_label_pelayanan[category_position], len(list_index_negatif),
                                                  list_index_negatif))
        list_total_pelayanan_netral.append((list_label_pelayanan[category_position], len(list_index_netral),
                                            list_index_netral))


def totalling_sentimen_pelayanan_kelurahan(list_senti_pelayanan_kel, list_total_senti_pelayanan_kel):
    if len(list_senti_pelayanan_kel) != 0:
        for kelurahan in list_kelurahan:
            list_index = [x for x, y in list_senti_pelayanan_kel if y == kelurahan]
            if len(list_index) != 0:
                list_total_senti_pelayanan_kel.append(
                    (kelurahan.title().replace(' ', '\n'), len(list_index), list_index))
        list_total_senti_pelayanan_kel.sort(key=takeSecond, reverse=True)
        list_sort_label = getList(0, list_total_senti_pelayanan_kel[:10])
        list_sort_total = getList(1, list_total_senti_pelayanan_kel[:10])
        list_sort_index = getList(2, list_total_senti_pelayanan_kel[:10])
        list_index = np.arange(len(list_total_senti_pelayanan_kel[:10]))
        return list_sort_label, list_sort_total, list_sort_index, list_index
    else:
        return [], [], [], []


def totalling_sentimen_pelayanan_kecamatan(list_senti_pelayanan_kec, list_total_senti_pelayanan_kec):
    if len(list_senti_pelayanan_kec) != 0:
        for kecamatan in list_kecamatan:
            list_index = [x for x, y in list_senti_pelayanan_kec if y == kecamatan]
            if len(list_index) != 0:
                list_total_senti_pelayanan_kec.append(
                    (kecamatan.title().replace(' ', '\n'), len(list_index), list_index))
        list_total_senti_pelayanan_kec.sort(key=takeSecond, reverse=True)
        list_sort_label = getList(0, list_total_senti_pelayanan_kec[:10])
        list_sort_total = getList(1, list_total_senti_pelayanan_kec[:10])
        list_sort_index = getList(2, list_total_senti_pelayanan_kec[:10])
        list_index = np.arange(len(list_total_senti_pelayanan_kec[:10]))
        return list_sort_label, list_sort_total, list_sort_index, list_index
    else:
        return [], [], [], []


def sentimen_category(list_category_positif, list_category_negatif, list_category_netral, list_output_category):
    for sentimen, category in zip(list_output_sentimen, list_output_category):
        if 'positif' in sentimen:
            list_category_positif.append(category)
        elif 'negatif' in sentimen:
            list_category_negatif.append(category)
        else:
            list_category_netral.append(category)


def sentimen_category_index(list_output_sentimen, list_output_category, list_output_category_sentimen):
    for index, sentimen in list_output_sentimen:
        list_output_category_sentimen.append((index, list_output_category[index]))


def show_chart_1(title, xlabel, index, list_total, list_label):
    if len(list_total) != 0:
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(jumlah_tweet)
        rects = ax.bar(index, list_total, align='center')
        ax.set_xticklabels(list_label)
        ax.set_xticks(index)
        autolabel(rects, ax)
        plt.show()


def show_chart_2(title, xlabel, index, list_total_laporan, list_total_keluhan, list_total_dukungan, list_label):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(jumlah_tweet)
    rect_laporan = ax.bar(index, list_total_laporan, width_2, color='b', align='center')
    rect_keluhan = ax.bar(index + width_2, list_total_keluhan, width_2, color='r', align='center')
    rect_dukungan = ax.bar(index + width_2 + width_2, list_total_dukungan, width_2, color='g',
                           align='center')
    ax.set_xticklabels(list_label)
    ax.set_xticks(index + width_2 + width_2 / 2)
    ax.legend((rect_laporan[0], rect_keluhan[0], rect_dukungan[0]), (laporan, keluhan, dukungan))
    autolabel(rect_laporan, ax)
    autolabel(rect_keluhan, ax)
    autolabel(rect_dukungan, ax)
    plt.show()


def export_to_excel(list_total_category_sentimen, list_index_tweet_category_sentimen,
                    list_sort_label_category_sentimen, list_sort_total_category_sentimen,
                    sheet, sentimen, category):
    sheet.write(0, 0, 'Sentimen')
    sheet.write(0, 1, category)
    sheet.write(0, 2, 'Total')
    sheet.write(0, 3, 'Isi Tweets')
    total_list_temp = 0
    for index in range(len(list_total_category_sentimen[:10])):
        list_index_tweet = list_index_tweet_category_sentimen[index]
        total_list = len(list_index_tweet)
        if total_list != 0:
            if index == 0:
                sheet.write(index + 1, 0, sentimen)
                sheet.write(index + 1, 1, list_sort_label_category_sentimen[index])
                sheet.write(index + 1, 2, list_sort_total_category_sentimen[index])
                for index_tweet in range(total_list):
                    tweet = tweets_test[list_index_tweet[index_tweet]]
                    sheet.write(index_tweet + 1, 3, tweet)
            else:
                sheet.write(total_list_temp + 1, 0, sentimen)
                sheet.write(total_list_temp + 1, 1, list_sort_label_category_sentimen[index])
                sheet.write(total_list_temp + 1, 2, list_sort_total_category_sentimen[index])
                for index_tweet in range(total_list):
                    tweet = tweets_test[list_index_tweet[index_tweet]]
                    sheet.write(index_tweet + total_list_temp + 1, 3, tweet)
        total_list_temp = total_list_temp + total_list


def list_sorted(list_netral, list_category_temp, list_category):
    for label_net, total_net, list_index_net in list_netral:
        for label_other, total_other, list_index_other in list_category_temp:
            if label_net == label_other:
                list_category.append((label_other, total_other, list_index_other))


def getList(index, list_category):
    return [item[index] for item in list_category]


def takeSecond(element):
    return element[1]


def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.0 * height,
                '%d' % int(height),
                ha='center', va='bottom')


with open('tweets_training.txt', encoding='utf-8-sig') as file_train:
    labels_train = []
    tweets_train = []
    data_train = file_train.readlines()
    for data in data_train:
        text = data.split(' | ')
        labels_train.append(text[0])
        tweets_train.append(text[1].replace('\n', ' '))

file_train.close()

with open('tweets_testing.txt', encoding='utf-8-sig') as file_test:
    dates_test = []
    tweets_test = []
    data_test = file_test.readlines()
    for data in data_test:
        text = data.split(' | ')
        dates_test.append(text[0])
        tweets_test.append(text[1].replace('\n', ''))

file_test.close()

with open('normalization_words.txt', encoding='utf-8-sig') as file_normalization:
    list_not_normal_words = []
    list_normalized_words = []
    data_normalization = file_normalization.readlines()
    for data in data_normalization:
        text = data.split(' | ')
        list_not_normal_words.append(text[0])
        list_normalized_words.append(text[1].replace('\n', ' '))

file_normalization.close()

list_stop_words = StopWordRemoverFactory().get_stop_words()
stemming = StemmerFactory().create_stemmer()

# with open('list_cleaned_tweets.txt', mode='w', encoding='utf-8') as file_cleaned:
#     for tweet_train in tweets_train:
#         cleaned_tweet_train = text_preprocessing(tweet_train)
#         file_cleaned.writelines('%s\n' % cleaned_tweet_train)
#
#     for tweet_test in tweets_test:
#         cleaned_tweet_test = text_preprocessing(tweet_test)
#         file_cleaned.writelines('%s\n' % cleaned_tweet_test)
# file_cleaned.close()

list_cleaned_tweets = []

with open('list_cleaned_tweets.txt', encoding='utf-8') as file_cleaned_tweets:
    cleaned_tweets = file_cleaned_tweets.readlines()
    for tweet in cleaned_tweets:
        list_cleaned_tweets.append(tweet.replace('\n', ''))
file_cleaned_tweets.close()

# tfidf = TfidfVectorizer(lowercase=False, norm=None, smooth_idf=False)
# dataset = tfidf.fit_transform(list_cleaned_tweets)
# model_train = dataset[:500]
# model_test = dataset[500:]
# model = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
# model.fit(model_train, labels_train)
# list_prediksi = model.predict(model_test).tolist()

# with open('tweets_predicted_labels.txt', mode='w', encoding='utf-8') as file_cleaned_tweets:
#     for prediksi in list_prediksi:
#         file_cleaned_tweets.writelines('%s\n' % prediksi)
# file_cleaned_tweets.close()

i = 0
width = 0.45
width_2 = 0.3
positif = 'Positif'
negatif = 'Negatif'
netral = 'Netral'
keluhan = 'Keluhan'
dukungan = 'Dukungan'
laporan = 'laporan'
tanggal = 'Tanggal'
pelayanan = 'Pelayanan'
kotamadya = 'Kotamadya'
kelurahan_text = 'Kelurahan'
kecamatan_text = 'Kecamatan'
jumlah_tweet = 'Jumlah Tweet'
font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}
plt.rc('font', **font)
book = xlwt.Workbook()
sh_tanggal_netral = book.add_sheet('Tanggal Netral')
sh_tanggal_negatif = book.add_sheet('Tanggal Negatif')
sh_tanggal_positif = book.add_sheet('Tanggal Positif')
sh_pelayanan_netral = book.add_sheet('Pelayanan Netral')
sh_pelayanan_negatif = book.add_sheet('Pelayanan Negatif')
sh_pelayanan_positif = book.add_sheet('Pelayanan Positif')
sh_kotamadya_netral = book.add_sheet('Kotamadya Netral')
sh_kotamadya_negatif = book.add_sheet('Kotamadya Negatif')
sh_kotamadya_positif = book.add_sheet('Kotamadya Positif')
sh_kecamatan_neg_pdm_listrik = book.add_sheet('Kecamatan Pemadaman Listrik')
sh_kelurahan_neg_pdm_listrik = book.add_sheet('Kelurahan Pemadaman Listrik')
sh_kecamatan_neg_cs = book.add_sheet('Kecamatan Customer Service')
sh_kelurahan_neg_cs = book.add_sheet('Kelurahan Customer Service')

list_category_sentimen = ['positif', 'netral', 'negatif']
list_output_sentimen = []
list_cleaned_tweets_testing = list_cleaned_tweets[500:]
list_daerah = []
list_kelurahan = []
list_kecamatan = []

with open('tweets_predicted_labels.txt', encoding='utf-8') as file_cleaned_tweets:
    tweets = file_cleaned_tweets.readlines()
    for tweet in tweets:
        list_output_sentimen.append(tweet.replace('\n', ''))
file_cleaned_tweets.close()

with open('list_daerah.txt') as file_daerah:
    read_list_daerah = file_daerah.readlines()
    for daerah in read_list_daerah:
        daerah.replace('\n', '')
        jak = daerah.split(' | ')
        kelurahan = jak[0]
        kecamatan = jak[1]
        list_daerah.append(daerah)
        list_kelurahan.append(kelurahan)
        if kecamatan not in list_kecamatan:
            list_kecamatan.append(kecamatan)

list_label_sentimen = ['Positif', 'Negatif', 'Netral']
list_total_sentimen = []
list_output_positif = []
list_output_negatif = []
list_output_netral = []

for sentimen in list_output_sentimen:
    i = i + 1
    if 'positif' in sentimen:
        list_output_positif.append((i - 1, sentimen))
    elif 'negatif' in sentimen:
        list_output_negatif.append((i - 1, sentimen))
    else:
        list_output_netral.append((i - 1, sentimen))

totalling_sentimen()

list_total_sentimen.sort(key=takeSecond, reverse=True)
list_sort_label_sentimen = getList(0, list_total_sentimen)
list_sort_total_sentimen = getList(1, list_total_sentimen)
index_sentimen = np.arange(len(list_total_sentimen))

show_chart_1('Banyaknya sentimen terhadap PLN', 'Sentimen', index_sentimen, list_sort_total_sentimen,
             list_sort_label_sentimen)

list_label_tanggal = []
list_category_tanggal = []
insert_category_label()

list_output_tanggal = []
list_output_tanggal_positif = []
list_output_tanggal_negatif = []
list_output_tanggal_netral = []
insert_output_tanggal()

sentimen_category_index(list_output_positif, list_output_tanggal, list_output_tanggal_positif)
sentimen_category_index(list_output_negatif, list_output_tanggal, list_output_tanggal_negatif)
sentimen_category_index(list_output_netral, list_output_tanggal, list_output_tanggal_netral)

list_total_tanggal_netral = []
list_total_tanggal_negatif = []
list_total_tanggal_negatif_temp = []
list_total_tanggal_positif = []
list_total_tanggal_positif_temp = []

totalling_sentimen_tanggal()

list_total_tanggal_netral.sort(key=takeSecond, reverse=True)
list_sorted(list_total_tanggal_netral, list_total_tanggal_negatif_temp, list_total_tanggal_negatif)
list_sorted(list_total_tanggal_netral, list_total_tanggal_positif_temp, list_total_tanggal_positif)

list_sort_label_tanggal_netral = getList(0, list_total_tanggal_netral[:10])
list_sort_label_tanggal_negatif = getList(0, list_total_tanggal_negatif[:10])
list_sort_label_tanggal_positif = getList(0, list_total_tanggal_positif[:10])

list_sort_total_tanggal_netral = getList(1, list_total_tanggal_netral[:10])
list_sort_total_tanggal_negatif = getList(1, list_total_tanggal_negatif[:10])
list_sort_total_tanggal_positif = getList(1, list_total_tanggal_positif[:10])

list_index_tweet_tanggal_netral = getList(2, list_total_tanggal_netral[:10])
list_index_tweet_tanggal_negatif = getList(2, list_total_tanggal_negatif[:10])
list_index_tweet_tanggal_positif = getList(2, list_total_tanggal_positif[:10])
index_tanggal = np.arange(len(list_total_tanggal_netral[:10]))

export_to_excel(list_total_tanggal_netral, list_index_tweet_tanggal_netral, list_sort_label_tanggal_netral,
                list_sort_total_tanggal_netral, sh_tanggal_netral, netral, tanggal)
export_to_excel(list_total_tanggal_negatif, list_index_tweet_tanggal_negatif, list_sort_label_tanggal_negatif,
                list_sort_total_tanggal_negatif, sh_tanggal_negatif, negatif, tanggal)
export_to_excel(list_total_tanggal_positif, list_index_tweet_tanggal_positif, list_sort_label_tanggal_positif,
                list_sort_total_tanggal_positif, sh_tanggal_positif, positif, tanggal)

show_chart_2('10 urutan terbanyak laporan, dukungan, dan keluhan terhadap PLN berdasarkan bulan Februari', 'Tanggal',
             index_tanggal, list_sort_total_tanggal_netral, list_sort_total_tanggal_negatif,
             list_sort_total_tanggal_positif, list_sort_label_tanggal_netral)

customer_service = ['telepon', 'respon']
galian_kabel = ['gali', 'konstruksi', 'proyek']
kabel_tiang_listrik = ['kabel', 'tiang']
mati_listrik = ['mati', 'padam', 'nyala', 'listrik']
token_listrik = ['token', 'meter', 'spaning']
gardu_lisrik = ['gardu', 'trafo']

list_category_pelayanan = [customer_service, galian_kabel, kabel_tiang_listrik, mati_listrik, token_listrik,
                           gardu_lisrik]
list_label_pelayanan = ['Customer\nService', 'Galian\nKabel', 'Kabel pada\nTiang Listrik', 'Pemadaman\nListrik',
                        'Token/Meteran\nListrik', 'Gardu\nListrik']

list_output_pelayanan = []
list_output_pelayanan_netral = []
list_output_pelayanan_negatif = []
list_output_pelayanan_positif = []

insert_output_pelayanan()
sentimen_category_index(list_output_positif, list_output_pelayanan, list_output_pelayanan_positif)
sentimen_category_index(list_output_negatif, list_output_pelayanan, list_output_pelayanan_negatif)
sentimen_category_index(list_output_netral, list_output_pelayanan, list_output_pelayanan_netral)

list_total_pelayanan_netral = []
list_total_pelayanan_negatif_temp = []
list_total_pelayanan_negatif = []
list_total_pelayanan_positif_temp = []
list_total_pelayanan_positif = []

totalling_sentimen_pelayanan()

list_total_pelayanan_netral.sort(key=takeSecond, reverse=True)
list_sorted(list_total_pelayanan_netral, list_total_pelayanan_negatif_temp, list_total_pelayanan_negatif)
list_sorted(list_total_pelayanan_netral, list_total_pelayanan_positif_temp, list_total_pelayanan_positif)

list_sort_label_pelayanan_netral = getList(0, list_total_pelayanan_netral)
list_sort_label_pelayanan_negatif = getList(0, list_total_pelayanan_negatif)
list_sort_label_pelayanan_positif = getList(0, list_total_pelayanan_positif)

list_sort_total_pelayanan_netral = getList(1, list_total_pelayanan_netral)
list_sort_total_pelayanan_negatif = getList(1, list_total_pelayanan_negatif)
list_sort_total_pelayanan_positif = getList(1, list_total_pelayanan_positif)

list_index_tweet_pelayanan_netral = getList(2, list_total_pelayanan_netral)
list_index_tweet_pelayanan_negatif = getList(2, list_total_pelayanan_negatif)
list_index_tweet_pelayanan_positif = getList(2, list_total_pelayanan_positif)
index_pelayanan = np.arange(len(list_total_pelayanan_netral))

export_to_excel(list_total_pelayanan_netral, list_index_tweet_pelayanan_netral, list_sort_label_pelayanan_netral,
                list_sort_total_pelayanan_netral, sh_pelayanan_netral, netral, pelayanan)
export_to_excel(list_total_pelayanan_negatif, list_index_tweet_pelayanan_negatif, list_sort_label_pelayanan_negatif,
                list_sort_total_pelayanan_negatif, sh_pelayanan_negatif, negatif, pelayanan)
export_to_excel(list_total_pelayanan_positif, list_index_tweet_pelayanan_positif, list_sort_label_pelayanan_positif,
                list_sort_total_pelayanan_positif, sh_pelayanan_positif, positif, pelayanan)

show_chart_2('Banyaknya laporan, dukungan, dan keluhan terhadap PLN berdasarkan pelayanan', 'Pelayanan',
             index_pelayanan,
             list_sort_total_pelayanan_netral, list_sort_total_pelayanan_negatif, list_sort_total_pelayanan_positif,
             list_sort_label_pelayanan_netral)

list_category_kota = ['jakpus', 'jaktim', 'jakbar', 'jakut', 'jaksel']
list_label_kota = ['Jakarta\nPusat', 'Jakarta\nTimur', 'Jakarta\nBarat', 'Jakarta\nUtara', 'Jakarta\nSelatan']

list_output_kota = []
list_output_kota_positif = []
list_output_kota_negatif = []
list_output_kota_netral = []

categorize_kota()
sentimen_category_index(list_output_positif, list_output_kota, list_output_kota_positif)
sentimen_category_index(list_output_negatif, list_output_kota, list_output_kota_negatif)
sentimen_category_index(list_output_netral, list_output_kota, list_output_kota_netral)

list_total_kota = []
list_total_kota_netral = []
list_total_kota_negatif_temp = []
list_total_kota_negatif = []
list_total_kota_positif_temp = []
list_total_kota_positif = []

totalling_sentimen_kota()

list_total_kota_netral.sort(key=takeSecond, reverse=True)
list_sorted(list_total_kota_netral, list_total_kota_negatif_temp, list_total_kota_negatif)
list_sorted(list_total_kota_netral, list_total_kota_positif_temp, list_total_kota_positif)

list_sorted_label_kota_netral = getList(0, list_total_kota_netral)
list_sorted_label_kota_negatif = getList(0, list_total_kota_negatif)
list_sorted_label_kota_positif = getList(0, list_total_kota_positif)

list_sort_total_kota_netral = getList(1, list_total_kota_netral)
list_sort_total_kota_negatif = getList(1, list_total_kota_negatif)
list_sort_total_kota_positif = getList(1, list_total_kota_positif)

list_index_tweet_kota_netral = getList(2, list_total_kota_netral)
list_index_tweet_kota_negatif = getList(2, list_total_kota_negatif)
list_index_tweet_kota_positif = getList(2, list_total_kota_positif)

index_kota = np.arange(len(list_total_kota_netral))

export_to_excel(list_total_kota_netral, list_index_tweet_kota_netral, list_sorted_label_kota_netral,
                list_sort_total_kota_netral, sh_kotamadya_netral, netral, kotamadya)
export_to_excel(list_total_kota_negatif, list_index_tweet_kota_negatif, list_sorted_label_kota_negatif,
                list_sort_total_kota_negatif, sh_kotamadya_negatif, negatif, kotamadya)
export_to_excel(list_total_kota_positif, list_index_tweet_kota_positif, list_sorted_label_kota_positif,
                list_sort_total_kota_positif, sh_kotamadya_positif, positif, kotamadya)

show_chart_2('Banyaknya laporan, dukungan, dan keluhan terhadap PLN berdasarkan kotamadya', 'Kotamadya',
             index_kota, list_sort_total_kota_netral, list_sort_total_kota_negatif,
             list_sort_total_kota_positif, list_sorted_label_kota_netral)

list_output_negatif_padam = []
list_output_negatif_padam_kec = []
list_output_negatif_padam_kel = []

list_output_negatif_cs = []
list_output_negatif_cs_kec = []
list_output_negatif_cs_kel = []

list_output_negatif_kabel = []
list_output_negatif_kabel_kec = []
list_output_negatif_kabel_kel = []

list_output_negatif_gardu = []
list_output_negatif_gardu_kec = []
list_output_negatif_gardu_kel = []

list_output_negatif_galian = []
list_output_negatif_galian_kec = []
list_output_negatif_galian_kel = []

list_output_negatif_token = []
list_output_negatif_token_kec = []
list_output_negatif_token_kel = []

list_total_negatif_padam_kec = []
list_sort_label_neg_padam_kec = []
list_sort_total_neg_padam_kec = []
list_sort_index_neg_padam_kec = []
list_index_neg_padam_kec = []

list_total_negatif_cs_kec = []
list_sort_label_neg_cs_kec = []
list_sort_total_neg_cs_kec = []
list_sort_index_neg_cs_kec = []
list_index_neg_cs_kec = []

list_total_negatif_kabel_kec = []
list_sort_label_neg_kabel_kec = []
list_sort_total_neg_kabel_kec = []
list_sort_index_neg_kabel_kec = []
list_index_neg_kabel_kec = []

list_total_negatif_gardu_kec = []
list_sort_label_neg_gardu_kec = []
list_sort_total_neg_gardu_kec = []
list_sort_index_neg_gardu_kec = []
list_index_neg_gardu_kec = []

list_total_negatif_galian_kec = []
list_sort_label_neg_galian_kec = []
list_sort_total_neg_galian_kec = []
list_sort_index_neg_galian_kec = []
list_index_neg_galian_kec = []

list_total_negatif_token_kec = []
list_sort_label_neg_token_kec = []
list_sort_total_neg_token_kec = []
list_sort_index_neg_token_kec = []
list_index_neg_token_kec = []

list_total_negatif_padam_kel = []
list_sort_label_neg_padam_kel = []
list_sort_total_neg_padam_kel = []
list_sort_index_neg_padam_kel = []
list_index_neg_padam_kel = []

list_total_negatif_cs_kel = []
list_sort_label_neg_cs_kel = []
list_sort_total_neg_cs_kel = []
list_sort_index_neg_cs_kel = []
list_index_neg_cs_kel = []

list_total_negatif_kabel_kel = []
list_sort_label_neg_kabel_kel = []
list_sort_total_neg_kabel_kel = []
list_sort_index_neg_kabel_kel = []
list_index_neg_kabel_kel = []

list_total_negatif_gardu_kel = []
list_sort_label_neg_gardu_kel = []
list_sort_total_neg_gardu_kel = []
list_sort_index_neg_gardu_kel = []
list_index_neg_gardu_kel = []

list_total_negatif_galian_kel = []
list_sort_label_neg_galian_kel = []
list_sort_total_neg_galian_kel = []
list_sort_index_neg_galian_kel = []
list_index_neg_galian_kel = []

list_total_negatif_token_kel = []
list_sort_label_neg_token_kel = []
list_sort_total_neg_token_kel = []
list_sort_index_neg_token_kel = []
list_index_neg_token_kel = []

categorize_sentimen_pelayanan(list_output_negatif, list_output_negatif_cs, list_output_negatif_galian,
                              list_output_negatif_kabel, list_output_negatif_padam, list_output_negatif_token,
                              list_output_negatif_gardu)

categorize_sentimen_pelayanan_daerah(list_output_negatif_padam, list_output_negatif_padam_kel,
                                     list_output_negatif_padam_kec)
categorize_sentimen_pelayanan_daerah(list_output_negatif_cs, list_output_negatif_cs_kel,
                                     list_output_negatif_cs_kec)
categorize_sentimen_pelayanan_daerah(list_output_negatif_kabel, list_output_negatif_kabel_kel,
                                     list_output_negatif_kabel_kec)
categorize_sentimen_pelayanan_daerah(list_output_negatif_gardu, list_output_negatif_gardu_kel,
                                     list_output_negatif_gardu_kec)
categorize_sentimen_pelayanan_daerah(list_output_negatif_galian, list_output_negatif_galian_kel,
                                     list_output_negatif_galian_kec)
categorize_sentimen_pelayanan_daerah(list_output_negatif_token, list_output_negatif_token_kel,
                                     list_output_negatif_token_kec)

list_sort_label_neg_padam_kec, list_sort_total_neg_padam_kec, list_sort_index_neg_padam_kec, list_index_neg_padam_kec \
    = totalling_sentimen_pelayanan_kecamatan(list_output_negatif_padam_kec, list_total_negatif_padam_kec)
list_sort_label_neg_cs_kec, list_sort_total_neg_cs_kec, list_sort_index_neg_cs_kec, list_index_neg_cs_kec \
    = totalling_sentimen_pelayanan_kecamatan(list_output_negatif_cs_kec, list_total_negatif_cs_kec)
list_sort_label_neg_kabel_kec, list_sort_total_negatif_kabel_kec, list_sort_index_neg_kabel_kec, list_index_neg_kabel_kec \
    = totalling_sentimen_pelayanan_kecamatan(list_output_negatif_kabel_kec, list_total_negatif_kabel_kec)
list_sort_label_neg_gardu_kec, list_sort_total_neg_gardu_kec, list_sort_index_neg_gardu_kec, list_index_neg_gardu_kec \
    = totalling_sentimen_pelayanan_kecamatan(list_output_negatif_gardu_kec, list_total_negatif_gardu_kec)
list_sort_label_neg_galian_kec, list_sort_total_neg_galian_kec, list_sort_index_neg_galian_kec, list_index_neg_galian_kec \
    = totalling_sentimen_pelayanan_kecamatan(list_output_negatif_galian_kec, list_total_negatif_galian_kec)
list_sort_label_neg_token_kec, list_sort_total_neg_token_kec, list_sort_index_neg_token_kec, list_index_neg_token_kec \
    = totalling_sentimen_pelayanan_kecamatan(list_output_negatif_token_kec, list_total_negatif_token_kec)
list_sort_label_neg_padam_kel, list_sort_total_neg_padam_kel, list_sort_index_neg_padam_kel, list_index_neg_padam_kel \
    = totalling_sentimen_pelayanan_kelurahan(list_output_negatif_padam_kel, list_total_negatif_padam_kel)
list_sort_label_neg_cs_kel, list_sort_total_neg_cs_kel, list_sort_index_neg_cs_kel, list_index_neg_cs_kel \
    = totalling_sentimen_pelayanan_kelurahan(list_output_negatif_cs_kel, list_total_negatif_cs_kel)
list_sort_label_neg_kabel_kel, list_sort_total_negatif_kabel_kel, list_sort_index_neg_kabel_kel, list_index_neg_kabel_kel \
    = totalling_sentimen_pelayanan_kelurahan(list_output_negatif_kabel_kel, list_total_negatif_kabel_kel)
list_sort_label_neg_gardu_kel, list_sort_total_neg_gardu_kel, list_sort_index_neg_gardu_kel, list_index_neg_gardu_kel \
    = totalling_sentimen_pelayanan_kelurahan(list_output_negatif_gardu_kel, list_total_negatif_gardu_kel)
list_sort_label_neg_galian_kel, list_sort_total_neg_galian_kel, list_sort_index_neg_galian_kel, list_index_neg_galian_kel \
    = totalling_sentimen_pelayanan_kelurahan(list_output_negatif_galian_kel, list_total_negatif_galian_kel)
list_sort_label_neg_token_kel, list_sort_total_neg_token_kel, list_sort_index_neg_token_kel, list_index_neg_token_kel \
    = totalling_sentimen_pelayanan_kelurahan(list_output_negatif_token_kel, list_total_negatif_token_kel)

show_chart_1('10 urutan terbanyak keluhan pemadaman listrik PLN berdasarkan kecamatan di Jakarta',
             kecamatan_text, list_index_neg_padam_kec, list_sort_total_neg_padam_kec, list_sort_label_neg_padam_kec)
show_chart_1('5 urutan terbanyak keluhan customer service listrik PLN berdasarkan kecamatan di Jakarta',
             kecamatan_text, list_index_neg_cs_kec, list_sort_total_neg_cs_kec, list_sort_label_neg_cs_kec)
show_chart_1('Top 10 banyaknya keluhan kabel pada tiang listrik PLN berdasarkan kecamatan di Jakarta',
             kecamatan_text, list_index_neg_kabel_kec, list_sort_total_neg_kabel_kec, list_sort_label_neg_kabel_kec)
show_chart_1('Top 10 banyaknya keluhan gardu listrik PLN berdasarkan kecamatan di Jakarta',
             kecamatan_text, list_index_neg_gardu_kec, list_sort_total_neg_gardu_kec, list_sort_label_neg_gardu_kec)
show_chart_1('Top 10 banyaknya keluhan galian kabel listrik PLN berdasarkan kecamatan di Jakarta',
             kecamatan_text, list_index_neg_galian_kec, list_sort_total_neg_galian_kec, list_sort_label_neg_galian_kec)
show_chart_1('Top 10 banyaknya keluhan kabel pada token/meteran listrik PLN berdasarkan kecamatan di Jakarta',
             kecamatan_text, list_index_neg_token_kec, list_sort_total_neg_token_kec, list_sort_label_neg_token_kec)
show_chart_1('10 urutan terbanyak keluhan pemadaman listrik PLN berdasarkan kelurahan di Jakarta',
             kelurahan_text, list_index_neg_padam_kel, list_sort_total_neg_padam_kel, list_sort_label_neg_padam_kel)
show_chart_1('3 urutan terbanyak keluhan customer service listrik PLN berdasarkan kelurahan di Jakarta',
             kelurahan_text, list_index_neg_cs_kel, list_sort_total_neg_cs_kel, list_sort_label_neg_cs_kel)
show_chart_1('Top 10 banyaknya keluhan kabel pada tiang listrik PLN berdasarkan kelurahan di Jakarta',
             kelurahan_text, list_index_neg_kabel_kel, list_sort_total_neg_kabel_kel, list_sort_label_neg_kabel_kel)
show_chart_1('Top 10 banyaknya keluhan gardu listrik PLN berdasarkan kelurahan di Jakarta',
             kelurahan_text, list_index_neg_kabel_kel, list_sort_total_neg_kabel_kel, list_sort_label_neg_kabel_kel)
show_chart_1('Top 10 banyaknya keluhan galian kabel listrik PLN berdasarkan kelurahan di Jakarta',
             kelurahan_text, list_index_neg_galian_kel, list_sort_total_neg_galian_kel, list_sort_label_neg_galian_kel)
show_chart_1('Top 10 banyaknya keluhan kabel pada token/meteran listrik PLN berdasarkan kelurahan di Jakarta',
             kelurahan_text, list_index_neg_token_kel, list_sort_total_neg_token_kel, list_sort_label_neg_token_kel)

export_to_excel(list_total_negatif_padam_kec, list_sort_index_neg_padam_kec, list_sort_label_neg_padam_kec,
                list_sort_total_neg_padam_kec, sh_kecamatan_neg_pdm_listrik, negatif, kecamatan_text)
export_to_excel(list_total_negatif_cs_kec, list_sort_index_neg_cs_kec, list_sort_label_neg_cs_kec,
                list_sort_total_neg_cs_kec, sh_kecamatan_neg_cs, negatif, kecamatan_text)
export_to_excel(list_total_negatif_padam_kel, list_sort_index_neg_padam_kel, list_sort_label_neg_padam_kel,
                list_sort_total_neg_padam_kel, sh_kelurahan_neg_pdm_listrik, negatif, kelurahan_text)
export_to_excel(list_total_negatif_cs_kel, list_sort_index_neg_cs_kel, list_sort_label_neg_cs_kel,
                list_sort_total_neg_cs_kel, sh_kelurahan_neg_cs, negatif, kelurahan_text)

book.save('Kategori Analisis Sentimen.xls')
