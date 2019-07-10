from random import randint
import time

def task(todo , size):
    while(True):
        print("\n\nList remaining is : ")
        for i in range(1,size+1):
            print(todo[i])
        a = input("\n\nShould we proceed ? : ")
        a = a.lower()
        if a == 'y' or a == 'yes' or a == 'yeah' or a == 'yup' or a == 'yep' or a == 'sure'or a == 'of course' or a == 'go for it':
            pass
        else :
            break
        r = randint(1,size)
        size = size - 1
        remove = todo.pop(r)
        print('\n' + remove + ' is removed from the list')
        
        if(size == 1):
            print("\n\nYou should be " + todo[1] + " right now.")
            break
        else:
            pass

print("Enter your tasks : ")
l = [0]
i = 1
while(i != ''):
    i = input()
    l.append(i)

l.pop()
size = len(l) - 1
task(l , size)
time.sleep(20)
        
        
