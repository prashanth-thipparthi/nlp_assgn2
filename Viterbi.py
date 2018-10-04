import math
import numpy as np
import re

training_file_name3 = "berp-POS-training_part1.txt"
training_file_name2 = "berp-POS-training_part2.txt"
training_file_name1 = "berp-POS-training_part3.txt"
#test_file_name = "test_local_part1.txt"
training_file_name = "berp-POS-trainingData1.txt"
test_file_name = "test_local_part1.txt"
output_file_name = "output1.txt"
unknown_symbol = "UNK"
tagSeparator = "|"
prefixPattern = "(^[0-9])"

#######   Word and Tag sequences from file #########

def UnigramSequence(file):
    tag_seq = []
    word_seq = []
    with open(file) as f:
        line = f.readline()
        while line:

            # Reading and splitting of the input data.
            line = f.readline()
            splitLine = line.split('\t')

            # Types and count of respective unigrams (tags)
            if len(splitLine) > 1:
                splitLine[1] = splitLine[1].rstrip()
                word_seq.append(splitLine[1])
                if len(splitLine) > 2:
                    splitLine[2] = splitLine[2].rstrip()
                    tag_seq.append(splitLine[2])
        if (len(tag_seq) > 0):
            return word_seq, tag_seq
        else:
            return word_seq


#######   Word and Tag sequences from file #########

#######   Unigram  word and tag count #########
def UnigramCount(file):
    unigram_tag_count_dict = {}
    unigram_word_count_dict = {}
    with open(file) as f:
        line = f.readline()
        unique_words = {}
        while line:

            # Reading and splitting of the input data.
            line = f.readline()
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

        return unigram_tag_count_dict, unigram_word_count_dict

    # print (unigram_tag_count_dict)
    # print (len(unigram_tag_count_dict))
    # print (unigram_word_count_dict)
    # print (len(unigram_word_count_dict))


#######   Unigram  word and tag count #########

#######   Unigram Tag Probability #############

def prob_UnigramTagDict(unigram_tag_count_dict):
    total_tag_count = 0
    prob_uni_tag_dict = {}
    for key in unigram_tag_count_dict.keys():
        total_tag_count = total_tag_count + unigram_tag_count_dict[key]
    for key, value in unigram_tag_count_dict.items():
        prob_uni_tag_dict[key] = (value / total_tag_count)

    return prob_uni_tag_dict


#######   Unigram Tag Probability #############

#######   BigramTransitionMatrix #############

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


#######   BigramTransitionMatrix #############


#######   Probability of BigramTransitionMatrix #############
# def prob_TransitionMatrix(transitionMatrix,unigram_tag_count_dict):
# 	total_transitiontags_count = 0
# 	prob_transitionMatrix = {}
# 	for key in transitionMatrix.keys():
# 		total_transitiontags_count = total_transitiontags_count + transitionMatrix[key]
# 	for key, value in transitionMatrix.items():
# 		#prob_transitionMatrix[key] = math.log10(value/total_transitiontags_count)

# 		prob_transitionMatrix[key] = (value/total_transitiontags_count)

# 	return prob_transitionMatrix

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


#######   Probability of BigramTransitionMatrix #############

#######   Probability of word_given_tag_dict #############
# def prob_word_given_tag_dict(word_given_tag_dict):
# 	total_word_given_tag_dict_count = 0
# 	prob_word_given_tag_dict = {}
# 	for key in word_given_tag_dict.keys():
# 		total_word_given_tag_dict_count = total_word_given_tag_dict_count + word_given_tag_dict[key]
# 	for key, value in word_given_tag_dict.items():
# 		#prob_transitionMatrix[key] = math.log10(value/total_transitiontags_count)
# 		prob_word_given_tag_dict[key] = (value/total_word_given_tag_dict_count)

# 	return prob_word_given_tag_dict
def prob_word_given_tag_dict(word_given_tag_dict, unigram_tag_count_dict):
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


#######   Probability of word_given_tag_dict #############

#######   Word Given Tag Dict #############
def WordGivenTag(word_sequence, tag_sequence):
    word_given_tag_dict = {}
    for i in range(0, len(tag_sequence)):
        key = word_sequence[i] + tagSeparator + tag_sequence[i]
        if key in word_given_tag_dict.keys():
            word_given_tag_dict[key] += 1
        else:
            word_given_tag_dict[key] = 1

    return word_given_tag_dict


#######   Word Given Tag Dict #############
'''
read File from FileName

'''

##
def readFile_LineByLine(filename):
    # lines = []
    with open(filename) as f:
        content = f.readlines()
        lines = []
        altered_line = ""
        for i in range(0, len(content)):
            lines.append(content[i].strip())
    f.close()
    return lines
##


def replaceWordWithUnkInTrainingFile(file, unigram_word_count_dict):
    newLines = []
    # print("unigram_word_count_dict:",unigram_word_count_dict['source'])
    with open(file) as f:
        line = f.readline()

        while line:
            line = f.readline()
            # Reading and splitting of the input data.
            splitLine = line.split('\t')

            # Types and count of respective unigrams (tags)
            if len(splitLine) > 1:
                splitLine[0] = splitLine[0].rstrip()
                splitLine[1] = splitLine[1].rstrip()
                splitLine[2] = splitLine[2].rstrip()
                if unigram_word_count_dict[splitLine[1]] < 2:
                    newLines.append(splitLine[0] + "\tUNK\t" + splitLine[2] + "\n")
                else:
                    newLines.append(splitLine[0] + "\t" + splitLine[1] + "\t" + splitLine[2] + "\n")
            else:
                newLines.append("\n")
    f.close()
    print("newLines.coumt():",len(newLines))
    return newLines


def replaceWordWithUnkInTestFile(file, training_word_sequence):
    unk_word_index_dict = {}
    newLines = []
    with open(file) as f:
        line = f.readline()
        i = 0
        while line:
            # Reading and splitting of the input data.
            line = f.readline()
            splitLine = line.split('\t')

            # Types and count of respective unigrams (tags)
            if len(splitLine) > 1:
                splitLine[1] = splitLine[1].rstrip()

                if splitLine[1] not in training_word_sequence:
                    unk_word_index_dict[i] = splitLine[1]
                    newLines.append(splitLine[0] + "\tUNK" + "\n")
                else:
                    newLines.append(splitLine[0] + "\t" + splitLine[1] + "\n")
            else:
                newLines.append("\n")
            i += 1
    f.close()
    return newLines, unk_word_index_dict


def writeLinesIntoFile(fileName, lines, ):
    finalWrite = open(fileName, 'w')
    for i in range(0, len(lines)):
        finalWrite.write(lines[i])
    finalWrite.close()


def replacingWordwithUNK(test_file_name, unigram_word_count_dict):
    # unk_word_index_dict = {}
    # training_word_sequence, tag_word_sequence = UnigramSequence(training_file_name)
    # print("training_word_sequence:",training_word_sequence)
    newLines, unk_word_index_dict = replaceWordWithUnkInTestFile(test_file_name, unigram_word_count_dict.keys())
    #writeLinesIntoFile("op.txt", newLines)
    writeLinesIntoFile(test_file_name, newLines)
    return unk_word_index_dict


def replacingUNKwithWord(file, unk_word_index_dict):
    newLines = []
    with open(file) as f:
        line = f.readline()
        i = 0
        while line:
            # Reading and splitting of the input data.
            line = f.readline()
            splitLine = line.split('\t')
            if len(splitLine) > 1:
                splitLine[0] = splitLine[0].rstrip()
                splitLine[1] = splitLine[1].rstrip()

                if str(i) in unk_word_index_dict.keys():
                    splitLine[1] = unk_word_index_dict[str(i)]
                    newLines.append(splitLine[0] + "\t" + splitLine[1] + "\n")
                else:
                    newLines.append(splitLine[0] + "\t" + splitLine[1] + "\n")
            else:
                newLines.append("\n")
        i += 1
    return newLines


# def WordGivenTag1(word_sequence, tag_sequence):
# 	dict1 = {}
# 	dict2={}

# 	for i in tag_sequence:
# 		if i in word_sequence:
# 			dict1[i] = dict2

def prob_word_given_tag_dict_1(prob_word_given_tag_dict):
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


def prob_word_given_tag_dict_1(prob_word_given_tag_dict):
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


def smoothed_transition_probabilty1(smoothed_transition_probabilty):
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

    unigram_tag_count_dict, unigram_word_count_dict = UnigramCount(training_file_name)

    newLines = replaceWordWithUnkInTrainingFile(training_file_name,unigram_word_count_dict)
    writeLinesIntoFile(training_file_name, newLines)

    unigram_tag_count_dict, unigram_word_count_dict = UnigramCount(training_file_name)

    unk_word_index_dict = replacingWordwithUNK(test_file_name, unigram_word_count_dict)

    word_sequence, tag_sequence = UnigramSequence(training_file_name)
    # print(word_sequence)

    unique_tags = list(unigram_tag_count_dict.keys())
    transitionMatrix = BigramTransitionMatrix(unique_tags, tag_sequence)

    k = 0.7
    smoothedTransitionMatrix, smoothed_transition_probabilty = addKBigramSmoothing(k, transitionMatrix,
                                                                                   unigram_tag_count_dict)
    smoothed_transition_probabilty1 = smoothed_transition_probabilty1(smoothed_transition_probabilty)
    #print(smoothed_transition_probabilty1)

    word_given_tag_dict = WordGivenTag(word_sequence, tag_sequence)
    # word_given_tag_dict1 = WordGivenTag1(word_sequence, tag_sequence)
    # print (word_given_tag_dict1)

    prob_word_given_tag_dict = prob_word_given_tag_dict(word_given_tag_dict, unigram_tag_count_dict)
    # print(prob_word_given_tag_dict)
    prob_word_given_tag_dict_1 = prob_word_given_tag_dict_1(prob_word_given_tag_dict)
    #print(prob_word_given_tag_dict_1)

    #prob_word_given_tag_dict_1 = prob_word_given_tag_dict_1(word_given_tag_dict,unigram_tag_count_dict)



    tokens = []
    output_lines = []
    sub_outputs = []
    START = "."
    END = "."
    test_d = readFile_LineByLine(test_file_name)
    print("test_d:",test_d)
    for i, data in enumerate(test_d):
        line = test_d[i]
        parts = line.split('\t')
        if len(parts) > 1:
            tokens.append(parts[1])
            sub_outputs.append(parts[0] + "\t" + parts[1])
        else:
            # for index, word in enumerate(tokens):
            #     if word not in unigram_word_count_dict.keys():
            #         tokens[index] = "UNK"
            #print("Tokens:",tokens)
            answer = viterbi_algo(tokens, [*smoothed_transition_probabilty1], prob_word_given_tag_dict_1,
                                  smoothed_transition_probabilty1, START, END)

            for j in range(0, len(sub_outputs)):
                sub_outputs[j] = sub_outputs[j] + '\t' + answer[j]
            output_lines.extend(sub_outputs)
            output_lines.append('\n')
            sub_outputs = []
            observations = []

    f = open(output_file_name, 'w')
    for i in range(0, len(output_lines)):
        # print( str(type(output_lines[i])))
        # print(output_lines[i])
        output = output_lines[i]
        if (i not in unk_word_index_dict.keys()):
                f.write(output)
                f.write("\n")
        else:
            parts = output.split('\t')
            if (len(parts) == 3):
                unk_replaced_output = parts[0] + str("\t") + unk_word_index_dict[i] + str('\t') + parts[2].strip() + '\n'
                #print(unk_replaced_output)
                f.write(unk_replaced_output)
            else:
                f.write(output)
                f.write("\n")
    f.close()
    # newLines = replacingUNKwithWord(test_file_name, unk_word_index_dict)
    # writeLinesIntoFile("opup.txt", newLines)