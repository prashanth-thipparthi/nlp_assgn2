import re
import json
import os
import sys
from random import getrandbits

data_set_file_name = "berp-POS-training_Slashed.txt"
training_file_name = "training.txt"
unaltered_test_file_name = "unaltered-test.txt"
test_file_name = "test.txt"
output_file_name = "viterbi-output.txt"
unknown_symbol = "UNK"
prefixPattern = "(^[0-9])"
bigram_word_separator = "|"
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
        print("1")
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
        print("2")
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
        print("3")
        if unaltered_test[i].strip():
            sentences = unaltered_test[i].strip().split(',')
            for j in range(len(sentences)):
                line = sentences[j]
                parts = line.split('\t')
                if(len(parts)==3):
                    l = parts[0].strip()+'\t'+parts[1].strip()+'\n'
                    test.append(l)
                    f3.write(parts[0].strip()+'\t'+parts[1].strip()+'\n')
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

    return unigram_tag_count_dict, unigram_word_count_dict

#######   Unigram  word and tag count #########
if __name__ == '__main__':
    # step1 : read in the data file
    content = readFile(data_set_file_name,prefixPattern)
    print("Content:",content)
    #a = np.arrayprint
    train, test = partitionData(content);
    unigram_tag_count_dict, unigram_word_count_dict = UnigramCount(train)
    print("Unigram tag count dict:",unigram_tag_count_dict)
    print("train:",train)
    print("test:",test)