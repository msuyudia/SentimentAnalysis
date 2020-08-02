import pandas as pd

labels_actual = []
labels_predicted = []

with open('tweets_actual_labels.txt', encoding='utf-8') as actual_file:
    labels_a = actual_file.readlines()
    for label in labels_a:
        labels_actual.append(label.replace('\n', ''))
actual_file.close()

with open('tweets_predicted_labels.txt', encoding='utf-8') as predicted_file:
    labels_p = predicted_file.readlines()
    for label in labels_p:
        labels_predicted.append(label.replace('\n', ''))
predicted_file.close()

list_true = []
list_false = []
net = 'netral'
neg = 'negatif'
pos = 'positif'
netpos = 'netpos'
netneg = 'netneg'
negnet = 'negnet'
negpos = 'negpos'
posnet = 'posnet'
posneg = 'posneg'
# netpos = predicted netral, but actually positif
# netneg = predicted netral, but actually negatif
# negnet = predicted negatif, but actually netral
# negpos = predicted negatif, but actually positif
# posnet = predicted positif, but actually netral
# posneg = predicted positif, but actually negatif

for i in range(len(labels_actual)):
    index = i - 1
    actual = labels_actual[index]
    predicted = labels_predicted[index]
    number = index + 1
    if predicted not in actual:
        list_false.append(predicted[:3] + actual[:3])
        print('%s Predicted %s, but actually is %s' % (number, predicted, actual))
    else:
        list_true.append(predicted)

total_net = list_true.count(net)
total_neg = list_true.count(neg)
total_pos = list_true.count(pos)
total_true = len(list_true)
total_netneg = list_false.count(netneg)
total_netpos = list_false.count(netpos)
total_negnet = list_false.count(negnet)
total_negpos = list_false.count(negpos)
total_posnet = list_false.count(posnet)
total_posneg = list_false.count(posneg)
total_false_net = total_netneg + total_netpos
total_false_neg = total_negnet + total_negpos
total_false_pos = total_posnet + total_posneg
total_false = len(list_false)
accuracy = total_true / (total_true + total_false)

df_confusion_matrix = pd.DataFrame({pos: [total_pos, total_netpos, total_negpos, ''],
                                    net: [total_posnet, total_net, total_negnet, ''],
                                    neg: [total_posneg, total_netneg, total_neg, ''],
                                    '< actual': ['', '', '', '']},
                                   [pos, net, neg, '^ predicted'])

df_accuracy = pd.DataFrame({'true': total_true,
                            'false': total_false,
                            'accuracy': accuracy},
                           ['total'])
print('\nTable Confusion Matrix')
print(df_confusion_matrix, '\n')
print('Table Accuracy')
print(df_accuracy)