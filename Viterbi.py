import re
import json
import os
import sys
from random import getrandbits
import numpy as np

data_set_file_name = "berp-POS-training.txt"
training_file_name = "training.txt"
unaltered_test_file_name = "unaltered-test.txt"
test_file_name = "test.txt"
output_file_name = "viterbi-output.txt"
unknown_symbol = "UNK"
prefixPattern = "(^[0-9])"
tagSeparator = "|"
my_hmm_name = "hmm"
start_state = "."


def readFile(filename,prefix_pattern):
    with open(filename) as f:
        content = f.readlines()
        #print('content size is:' + str(len(content)))
    if prefix_pattern:
        #print("File Lines is:" + str(len(content)))
        lines = []
        altered_line = ""
        for i in range(0, len(content)):
            #print("Content[i]: " + content[i])
            m = re.search(prefix_pattern,content[i])
            if m:
                #print("Group is:" + m.group())
                altered_line = altered_line + content[i].strip() +","
            else:
                lines.append(altered_line.strip(","))
                lines.append("\n")
                altered_line = ""
        return lines
    else:
        return content

def partitionData(content):
    '''
    Purpose: Partition the data read into testing and training data.
    :param content: List of strings
    :return: 2 lists.. the first a training set, the second a test set.
    '''
    # import os.path
    # if( (os.path.exists(training_file_name) and os.path.exists(test_file_name) and os.path.exists(unaltered_test_file_name)) ):
    #     test = readFile(test_file_name,"")
    #     train = readFile(training_file_name,"")
    #     #print('training size... '+str(len(train)) +"..... test size ...."+str(len(test)))
    #     return train,test
    # else:
    f1 = open(training_file_name, 'w')
    f2 = open(unaltered_test_file_name, 'w')
    f3 = open(test_file_name, 'w')
    unaltered_test = []
    training = []
    test = []
    for k in range (int(len(content)/10)):
        if content[k].strip():
            sentences = content[k].strip().split(',')
            for j in range(len(sentences)):
                l = sentences[j].strip() + '\n'
                unaltered_test.append(l)
                f2.write(sentences[j].strip() + '\n')
            unaltered_test.append('\n')
            f2.write('\n')
    i = 0
    while i < int(len(content)-len(content)/10):
        if content[i].strip():
            sentences = content[i].strip().split(',')
            for j in range(len(sentences)):
                l = sentences[j].strip() + "\n"
                training.append(l)
                f1.write(sentences[j].strip()+'\n')
            training.append('\n')
            f1.write('\n')
        i+=1

    for i in range(len(unaltered_test)):
        if unaltered_test[i].strip():
            sentences = unaltered_test[i].strip().split(',')
            for j in range(len(sentences)):
                line = sentences[j]
                parts = line.split('\t')
                if(len(parts)==3):
                    l = parts[0].strip()+'\t'+parts[1].strip()+'\n'
                    test.append(l)
                    f3.write(parts[0].strip()+'\t'+parts[1].strip()+'\n')
        else:
            test.append('\n')
            f3.write('\n')
    f1.close()
    f2.close()
    f3.close()
    return training,test

#######   Unigram  word and tag count #########
def UnigramCount(train):
    unigram_tag_count_dict = {}
    unigram_word_count_dict = {}
    total_no_of_words = 0
    for line in train:
        splitLine = line.split('\t')
        # Types and count of respective unigrams (tags)
        if len(splitLine) > 1:
            splitLine[1] = splitLine[1].rstrip()
            splitLine[2] = splitLine[2].rstrip()
            if splitLine[2] in unigram_tag_count_dict.keys():
                unigram_tag_count_dict[splitLine[2]] += 1
            else:
                unigram_tag_count_dict[splitLine[2]] = int(1)

            if splitLine[1] in unigram_word_count_dict.keys():
                unigram_word_count_dict[splitLine[1]] += 1
            else:
                unigram_word_count_dict[splitLine[1]] = int(1)
            total_no_of_words +=1

    return unigram_tag_count_dict, unigram_word_count_dict, total_no_of_words

#######   Unigram  word and tag count #########

def unigram_verification(unigrams, total_words, flag):
    unigram_prob = 0
    unigram_prob_dict = {}
    for key in unigrams:
        unigram_prob = (unigrams[key]/total_words)
        unigram_prob_dict[key] = unigram_prob

    if flag:
        print("Unigram Prob sum: " + str(unigram_prob))
        for key in unigrams.keys():
            print("Unigram key is:" + key + " ...... count is...... " + str(unigrams[key]))
    return unigram_prob_dict

def replaceWordWithUnkInTestFile(test, unigram_word_list):
    unk_word_index_dict = {}
    for i in range(len(test)):
        # Reading and splitting of the input data.
        line = test[i]
        splitLine = line.split('\t')

        # Types and count of respective unigrams (tags)
        if len(splitLine) > 1:
            splitLine[1] = splitLine[1].rstrip()

            if splitLine[1] not in unigram_word_list:
                unk_word_index_dict[i] = splitLine[1]
                test[i] = splitLine[0] + "\t"+ unknown_symbol+ "\n"
            else:
                test[i] = splitLine[0] + "\t" + splitLine[1] + "\n"
        else:
            test[i] = "\n"
        i += 1
    return test, unk_word_index_dict

def replaceWordWithUnkInTrainingFile(train, unigram_word_count_dict):
    for i in range(len(train)):
        # Reading and splitting of the input data.
        line = train[i]
        splitLine = line.split('\t')

        # Types and count of respective unigrams (tags)
        if len(splitLine) > 1:
            splitLine[1] = splitLine[1].rstrip()

            if unigram_word_count_dict[splitLine[1]] < 2:
                train[i] = splitLine[0].strip() + "\t" + unknown_symbol + "\t" + splitLine[2].rstrip() + "\n"
            else:
                train[i] = splitLine[0].strip() + "\t" + splitLine[1].strip() + "\t" + splitLine[2].rstrip() + "\n"
        else:
            train[i] = "\n"
        i += 1
    return train

def UnigramSequence(train):
    tag_seq = []
    word_seq = []
    for line in train:
        splitLine = line.split('\t')
        if len(splitLine) > 1:
            splitLine[1] = splitLine[1].rstrip()
            splitLine[2] = splitLine[2].rstrip()
            word_seq.append(splitLine[1])
            tag_seq.append(splitLine[2])
    return tag_seq, word_seq

def BigramTransitionMatrix(unique_tags, tag_sequence):
    transitionMatrix = {}
    ### initialising transition matrix ###
    for i in unique_tags:
        for j in unique_tags:
            transitionMatrix[i + tagSeparator + j] = 0

    ### Adding counts to the transition matrix ###
    for i in range(1, len(tag_sequence)):
        k1 = tag_sequence[i - 1]
        k2 = tag_sequence[i]
        fk = k2 + tagSeparator + k1  # currenttag-Prevtag ##P (Tag  given previous tag)
        if fk in transitionMatrix.keys():
            transitionMatrix[fk] += 1
    return transitionMatrix

def EmmissionMatrix(word_sequence, tag_sequence):
    print("check")
    word_given_tag_dict = {}
    for i in range(0, len(tag_sequence)):
        key = word_sequence[i] + tagSeparator + tag_sequence[i]
        if key in word_given_tag_dict.keys():
            word_given_tag_dict[key] += 1
        else:
            word_given_tag_dict[key] = 1

    return word_given_tag_dict

def addKBigramSmoothing(k, transitionMatrix, unigram_tag_count_dict):
    smoothed_bigram_counts = {}
    smoothed_bigram_probability = {}
    vocab_size = k * len(transitionMatrix.keys())
    for bigram_key in transitionMatrix.keys():
        smoothed_bigram_counts[bigram_key] = transitionMatrix[bigram_key] + k

    for bigram_key in smoothed_bigram_counts.keys():
        wiminus1 = bigram_key.split(tagSeparator)[1]
        smoothed_bigram_probability[bigram_key] = smoothed_bigram_counts[bigram_key] / (
                    unigram_tag_count_dict[wiminus1] + vocab_size)
    return smoothed_bigram_counts, smoothed_bigram_probability

def Convert_Smoothed_TransitionMatrix_Format(smoothed_transition_probabilty):
    _words = {}
    _tags = {}
    key_split = []

    for item in smoothed_transition_probabilty:
        arr = item.split("|")
        if (arr[0] not in _words):
            _words[arr[0]] = True

        if (arr[1] not in _tags):
            _tags[arr[1]] = True

    mat = dict()
    for tag in [*_tags]:
        mat[tag] = {}
        for w in [*_words]:
            mat[tag][w] = 0

    for key1, value1 in smoothed_transition_probabilty.items():
        arr = key1.split("|")
        mat[arr[1]][arr[0]] = value1
    return mat

def prob_EmissionMatrix(word_given_tag_dict, unigram_tag_count_dict):
    prob_word_given_tag_dict = {}
    for key, value in word_given_tag_dict.items():
        wiminus1 = key.split(tagSeparator)[1]
        # print("key:",key)
        # print("Value:%s, (unigram_tag_count_dict[wiminus1]) : %s "%(value,unigram_tag_count_dict[wiminus1]))
        if (wiminus1 is not None):
            prob_word_given_tag_dict[key] = value / (unigram_tag_count_dict[wiminus1])
        else:
            continue
    return prob_word_given_tag_dict


def Convert_EmmissionMatrix_Format(prob_word_given_tag_dict):
    _words = {}
    _tags = {}
    key_split = []

    for item in prob_word_given_tag_dict:
        arr = item.split("|")
        if (arr[0] not in _words):
            _words[arr[0]] = True

        if (arr[1] not in _tags):
            _tags[arr[1]] = True

    mat = dict()
    for tag in [*_tags]:
        mat[tag] = {}
        for w in [*_words]:
            mat[tag][w] = 0

    for key1, value1 in prob_word_given_tag_dict.items():
        arr = key1.split("|")
        mat[arr[1]][arr[0]] = value1
    return mat

def multiply(x, y):
    return x * y


def viterbi_algo(string_tokens, states, emission_prob, tag_transition_prob, start_of_string_state, end_of_string_state):
    T = len(string_tokens)
    N = len(states)

    viterbi = np.zeros((N + 3, T + 1))
    backpointer = np.zeros((N + 3, T + 1))

    for index, state in enumerate(states, 1):
        tag_trans = tag_transition_prob[start_of_string_state][state]
        emission = emission_prob[state][string_tokens[0]]
        viterbi[index, 1] = multiply(tag_trans, emission)
        backpointer[index, 1] = 0

    for o_i, o in enumerate(string_tokens[1:], 2):
        for s_i, s in enumerate(states, 1):
            for _s_i, _s in enumerate(states, 1):
                prev = viterbi[_s_i, o_i - 1]
                tag_trans = tag_transition_prob[_s][s]
                temp = multiply(prev, tag_trans)
                if temp > viterbi[s_i, o_i]:
                    viterbi[s_i, o_i] = temp
                    backpointer[s_i, o_i] = _s_i
            #print("emission_prob[%s][%s]:%s"%(s,o,emission_prob[s][o]))
            emission = emission_prob[s][o]
            viterbi[s_i, o_i] = multiply(viterbi[s_i, o_i], emission)

    for s_i, s in enumerate(states, 1):
        tag_trans = tag_transition_prob[s][end_of_string_state]
        temp = multiply(viterbi[s_i, T], tag_trans)

        if temp > viterbi[N + 1, T]:
            viterbi[N + 1, T] = temp
            backpointer[N + 1, T] = s_i

    ans = list(np.zeros((len(string_tokens) + 1,)))

    z_i = int(backpointer[N + 1, T])
    ans[T] = states[z_i - 1]

    for index in range(T, 1, -1):
        z_i = int(backpointer[z_i, index])
        ans[index - 1] = states[z_i - 1]

    ans = ans[1:]
    ans.append(end_of_string_state)
    return ans

if __name__ == '__main__':
    # step1 : read in the data file
    content = readFile(data_set_file_name,prefixPattern)
    #print("Content:",content)
    #a = np.arrayprint

    # step 2: partition the data
    train, test = partitionData(content);

    # print("Unigram tag count dict:",unigram_tag_count_dict)
    # print("train:",train)
    # print("test:",test)

    #step 3 : Unigram
    unigram_tag_count_dict, unigram_word_count_dict, total_no_of_words = UnigramCount(train)
    unigram_prob = unigram_verification(unigram_tag_count_dict,total_no_of_words,False)
    #print(unigram_prob)

    
    # step 4: UNK handling in test and training data

    train = replaceWordWithUnkInTrainingFile(train, unigram_word_count_dict)
    # print("Word_Count_Dict:",unigram_word_count_dict)
    # print("Train:", train)
    unigram_tag_count_dict, unigram_word_count_dict, total_no_of_words = UnigramCount(train)
    unigram_prob = unigram_verification(unigram_tag_count_dict, total_no_of_words, False)
    # test.append("3"+'\t'+"mikeTesting"+'\n')
    test, unk_word_index_dict = replaceWordWithUnkInTestFile(test, unigram_word_count_dict.keys())
    #print("TEST:",test)
    # unigram_word_count_dict["mikeTesting"] = 1
    # train.append("3" + '\t' + "mikeTesting" +'\t' + "check" + '\n')



    #Step 5 Transition and Emmission Matrices
    tag_sequence , word_sequence  = UnigramSequence(train)
    transitionMatrix = BigramTransitionMatrix(unigram_tag_count_dict.keys(), tag_sequence)
    #print("word_Sequence:",word_sequence)
    EmissionMatrix = EmmissionMatrix(word_sequence, tag_sequence)
    #print("EmissionMatrix:",EmissionMatrix)

    # Step 6 Probabilities of Matrices
    k = 0.7
    smoothedTransitionMatrix, prob_smoothed_transition = addKBigramSmoothing(k, transitionMatrix,
                                                                                   unigram_tag_count_dict)

    #print("EmissionMatrix:",prob_smoothed_transition)

    prob_EmissionMatrix = prob_EmissionMatrix(EmissionMatrix, unigram_tag_count_dict)
    #print(prob_EmissionMatrix)

    #Step 7 Converting Matrices to TransitionMatrix = {"tag1" : {'tag2':<count>,'tag3':<count>,'tag4':<count>}}
    # EmmissionMatrix = {"tag1" : {'word1':<count>,'word2':<count>,'word3':<count>}}
    prob_SmoothedTransitionMatrix = Convert_Smoothed_TransitionMatrix_Format(prob_smoothed_transition)
    prob_EmmissionMatrix = Convert_EmmissionMatrix_Format(prob_EmissionMatrix)

    # print("EmmissionMatrix:",prob_EmmissionMatrix)
    # print("TransitionMatrix:", prob_SmoothedTransitionMatrix)

    tokens = []
    output_lines = []
    sub_outputs = []
    START = "."
    END = "."
    for i, data in enumerate(test):
        line = test[i]
        parts = line.split('\t')
        if len(parts) > 1:
            tokens.append(parts[1].rstrip())
            sub_outputs.append(parts[0].rstrip() + "\t" + parts[1].rstrip())
        else:
            answer = viterbi_algo(tokens, [*prob_SmoothedTransitionMatrix], prob_EmmissionMatrix,
                                  prob_SmoothedTransitionMatrix, START, END)

            for j in range(0, len(sub_outputs)):
                sub_outputs[j] = sub_outputs[j] + '\t' + answer[j]
            output_lines.extend(sub_outputs)
            output_lines.append('\n')
            sub_outputs = []
            observations = []
            tokens = []

    f = open(output_file_name, 'w')
    for i in range(0, len(output_lines)):
        output = output_lines[i]
        if (i not in unk_word_index_dict.keys()):
                f.write(output)
                if output.strip():
                    f.write("\n")
        else:
            parts = output.split('\t')
            if (len(parts) == 3):
                unk_replaced_output = parts[0].rstrip() + str("\t") + unk_word_index_dict[i] + str('\t') + parts[2].strip() + '\n'
                f.write(unk_replaced_output)
    f.close()
