#!/bin/bash
import random
import sys
if __name__ == "__main__":
	try:
		it=int(sys.argv[1])
	except IndexError:
		it=10000
	else:
		pass
	unchange_win = 0
	change_win = 0
	for i in range(it):
		choice=[1,2,3]
		prize = random.randint(1,3)
		choose_unchange = random.randint(1,3)
		'''if choose_unchange == prize:
			unchange_win+=1'''
		choice.remove(choose_unchange)
		if choice[0] == prize:
			choice.remove(choice[0])
		elif choice[1] == prize:
			choice.remove(choice[0])
		else:
			choice.remove(choice[random.randint(0,1)])
		if choose_unchange == prize :
			unchange_win=unchange_win+1
		else:
			change_win=change_win+1
	print "change win:"
	print change_win*1.0/it
	print "unchange win"
	print unchange_win*1.0/it