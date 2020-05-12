import collections

import numpy as np

import util
import svm
import csv

from ps2.ps2.ps2.src.spam.svm import train_and_predict_svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. For normalization,
    you should convert everything to lowercase.

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.
    """

    # *** START CODE HERE ***
    # this is how the get_words function works:
    # it gets a single message and returns the lower case and each word in a list
    lower_case = message.lower()
    splitted = lower_case.split()

    return splitted
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least five messages.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    # make a list of all the words
    word_list = []
    dict = {}
    c = 0
    for row in messages:
        seperated = get_words(row)
        for i in seperated:
            word_list.append(i)
    for words in word_list:
        tobe = words in dict.values()
        if tobe == False:
            if word_list.count(words) > 4:
                dict[c] = words
                c += 1
    return dict
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message.
    Each row in the resulting array should correspond to each message
    and each column should correspond to a word of the vocabulary.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    Array = np.zeros([len(messages),len(word_dictionary)])
    keys = list(word_dictionary.keys())
    values = list(word_dictionary.values())
    for i in range(len(messages)):
        for words in get_words(messages[i]):
            condition = words in values
            if condition == True:
                n = get_words(messages[i]).count(words)
                j = keys[values.index(words)]
                Array[i,j] = n


    return Array
    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes model given a training matrix and labels.

    The function should return the state of that model.

    Feel free to use whatever datatype you wish for the state of the model.

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    # *** START CODE HERE ***
    n_words = np.shape(matrix)[1]
    n_label = np.shape(matrix)[0]
    phi_spam = np.zeros((1,n_words))
    phi_nonspam = np.zeros((1,n_words))
    zeros = 0
    ones = 0
    # count the number of spam(1) and not-spam(0) emails
    for i in labels:
        if i == 0:
            zeros += 1
        else:
            ones += 1
    phi_y = (1 + ones)/(n_words + n_label) #P(y=1)

    for i in range(n_label):
        if labels[i] == 0:
            for j in range(n_words):
                if matrix[i,j] != 0:
                    phi_nonspam[0,j] += 1/(2+zeros)
        if labels[i] == 1:
            for j in range(n_words):
                if matrix[i,j] != 0:
                    phi_spam[0,j] += 1/(2+ones)
    for j in range(n_words):
        phi_spam[0, j] += 1 / (2 + ones)
        phi_nonspam[0, j] += 1 / (2 + zeros)
    return phi_y, phi_spam,  phi_nonspam
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(phi_y, phi_spam,  phi_nonspam, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containg the predictions from the model
    """
    # *** START CODE HERE ***
    #zeros, ones, phi_spam, phi_nonspam = model
    n_words = np.shape(matrix)[1]
    n_label = np.shape(matrix)[0]
    phi = np.zeros((1,n_label))
    prob_log = np.zeros((1, n_label))
    pred = np.zeros((1, n_label))
    for i in range(n_label):
        mul1 = 1  # y=1
        mul2 = 1  # y=0
        summation = 0
        for j in range(n_words):
            if matrix[i,j] != 0:
                mul1 = mul1*phi_spam[0,j]
                mul2 = mul2*phi_nonspam[0,j]
                #summation += np.log(phi_spam[0,j])
                summation += np.log(phi_nonspam[0, j])
        #print(phi_y*mul1 + mul2*(1-phi_y))
        #prob_log[0,i] = summation + np.log(phi_y) - np.log(phi_y*mul1 + mul2*(1-phi_y))
        # probability that it is not spam
        prob_log[0, i] = summation + np.log(1-phi_y) - np.log(phi_y * mul1 + mul2 * (1 - phi_y))
        pred[0,i] = np.exp(prob_log[0,i])

    for i in range(n_label):
        if pred[0,i] < 0.5:
            pred[0,i] = 1
        else:
            pred[0,i] = 0


    return pred
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(phi_y, phi_spam,  phi_nonspam, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    N = len(dictionary)
    ratio = np.zeros((1,N))
    for j in range(N):
        ratio[0,j] = np.log(phi_spam[0,j]/phi_nonspam[0,j])
    sorted_ind = np.argsort(ratio)
    sorted_list = sorted_ind[0].tolist()
    word = []
    for i in sorted_list:
        word.append(dictionary[i])
    n = len(word)
    last_five = [word[n-1],word[n-2],word[n-3],word[n-4],word[n-5]]
    return last_five

    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    naive_bayes_accuracy = []
    for j in radius_to_consider:
        y = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, j)
        naive_bayes_accuracy.append(np.mean(y == val_labels))
    # *** END CODE HERE ***
    return radius_to_consider[np.argmax(naive_bayes_accuracy)]

def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')
    """"
    # Checking the code portions seperately
    dict = create_dictionary(train_messages)
    Arrray1 = transform_text(train_messages, dict)
    phi_y, phi_spam,  phi_nonspam = fit_naive_bayes_model(Arrray1, train_labels)

    Arrray = transform_text(test_messages, dict)

    word = get_top_five_naive_bayes_words(phi_y, phi_spam, phi_nonspam, dict)
    #print(word)
    """

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    phi_y, phi_spam,  phi_nonspam = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(phi_y, phi_spam, phi_nonspam, test_matrix)

    #np.savetxt('spam_naive_bayes_predictions', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(phi_y, phi_spam, phi_nonspam, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('spam_optimal_radius', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()
