
def get_split(text):
	# Delete '[SEP]'
	text = text.replace('[SEP]','')
	# print(len(text.split()))
	l_total = []
	l_parcial = []
	if len(text.split())//150 >0:
		n = len(text.split())//150
	else: 
		n = 1
	for w in range(n):
		if w == 0:
			l_parcial = text.split()[:200]
			l_total.append(" ".join(l_parcial))
		else:
			l_parcial = text.split()[w*150:w*150 + 200]
			l_total.append(" ".join(l_parcial))

	return l_total

def get_fixed_split(text):
	# Delete '[SEP]'
	text = text.replace('[SEP]','')
	# print(len(text.split()))
	l_total = []
	l_parcial = []
	if len(text.split())//200 >0:
		n = len(text.split())//200
	else: 
		n = 1
	for w in range(n):
		l_parcial = text.split()[w*200:w*200 + 200]
		l_total.append(" ".join(l_parcial))

	return l_total


def get_natural_split(text):
	# Use '[SEP]' to get natural split
	l_total = []
	l_total = text.split('[SEP]')
	return l_total


