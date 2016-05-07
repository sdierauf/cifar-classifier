#maps class ids to words


mapping = dict()
file_path = "../tiny-imagenet-200/words.txt"
#file_path = '/Users/lilstutzgrl/School/IBR/FinalProject/tinyimagenet-200/words.txt' 
#for line in open(file_path, 'r'):
lines = open(file_path, 'r').readlines() #list of strings
for line in lines:
	contents = line.split(' ') #line[i] is a string
	length = len(contents)
	mapping[contents[0]] = contents[1:length-1]
	print("" %contents)

print ("mapping" %mapping['n02124075'])

