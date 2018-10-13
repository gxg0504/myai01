from random import choice


def game(change):
    #代表三个门
    choice_no = [1,2,3]
    #真正有奖的那个门的序号
    prize = choice(choice_no)
    # print('prize is',prize)
    your_choice = choice(choice_no)
    # print('yourchoice',your_choice)
    choice_no.pop(choice_no.index(your_choice))
    try:
        c = choice_no.copy()
        c.pop(c.index(prize))
        host_open = c[0]
    except:
        host_open=choice(choice_no)
    # print(len(choice_no))
    # print('host open',host_open)
    if change==1:
        choice_no.pop(choice_no.index(host_open))
        # print(len(choice_no))
        your_choice=choice_no[0]
        # print('your new choice',your_choice)
    return your_choice==prize

if __name__ == '__main__':
    times = 1000000
    win = 0
    for i in range(times):
        win+=game(0)
    print(win/times)
