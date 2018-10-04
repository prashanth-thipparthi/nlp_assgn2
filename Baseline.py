test_file = "assgn2-test-set.txt"
output_file_name = "output_2.txt"
outputs=[]

training_file_name = "berp-POS-training.txt"
# test_file_name = "test_local.txt"
# output_file_name = "output.txt"
unknown_symbol = "UNK"
training_dict ={}
#training_dict={}

def readFile(file):
	word_dict = {}
	with open(file) as f:
		line = f.readline()

		while line:
			line = f.readline()
			arr = line.split('\t')
			tag_dict = None
			# Making of word and tag counter
			if len(arr) > 1:
				arr[2] = arr[2].rstrip()
				if arr[1] in word_dict.keys(): 
					tag_dict = word_dict[arr[1]]
					if arr[2] in tag_dict.keys():
						tag_dict[arr[2]]+=1
						word_dict[arr[1]] = tag_dict

					else:
						tag_dict[arr[2]] = 1
						word_dict[arr[1]] = tag_dict 
				else:
					count = 1
					word_dict[arr[1]] = {arr[2]:count}
	#print(word_dict)

	# Max count of tag for a word.
	global training_dict
	for key, value in word_dict.items():
		maxVal = 0
		flag = True
		max_tag_dict = {}
		for key1, value1 in value.items():
			if(flag):
				maxVal = value1
				flag = False
				key2 = key1
			if(maxVal<value1):
				maxVal = value1
				key2 = key1
		max_tag_dict[key2]=maxVal				
		training_dict[key] = max_tag_dict 
	print ("training_dict: ",training_dict)

def readFile1(file1):

	f = open(file1, 'r')
	g = open(output_file_name, 'w')

	for line in f:
		if line.strip():
			arr1 = line.split('\t')
			word = arr1[1].rstrip()
			if(word in training_dict.keys()):
				inner_dict = training_dict[word]
				tag = list(inner_dict.keys())[0].strip()
			else:
				tag = unknown_symbol.strip()
			out=str(arr1[0]) + "\t" + str(word) + "\t" + str(tag) + "\n"
			outputs.append(out)
		else:
			outputs.append(line)
	for i in range(0,len(outputs)):
		g.write(outputs[i])

	f.close()
	g.close()


if __name__ == '__main__':
    # step1 : read in the data file
	readFile(training_file_name)
	readFile1(test_file)
	




# if __name__ == '__main__':
#     # step1 : read in the data file
# 	content = readFile1(test_file)

