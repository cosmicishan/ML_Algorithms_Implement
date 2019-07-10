import json
from difflib import get_close_matches
data = json.load(open('data.json'))

def translate(w):
    w = w.lower()
    if w in data:
        return data[w]
    elif len(get_close_matches(w,data.keys())) > 0:
        yn = input("Do you mean %s if yes press Y else N : " % get_close_matches(w,data.keys())[0])
        if yn == 'y' and 'Y':
            return data[get_close_matches(w,data.keys())[0]]
        elif yn == 'n' and 'N':
            return 'This word doen\'t exist check the word twice.'
        else:
            return 'Give proper input'
    else:
        return 'This word doen\'t exist check the word twice.'

#print("The dictionary contain {} words of english language ".format(len(data)))        
sign = True
while(sign):    
    word = input("\nEnter the word you want to see the meaning : ")

    output = translate(word)

    if type(output) == list:
        for i in output:
            print(i)
    else :
        print(output)

    check = input("\n\nDo you want to get meaning of another word if yes press Y else N : ")

    if check == 'y' and 'Y':
        pass
    elif check == 'n' and 'N':
         print("\nHave a Nice day")
         sign = False
         break    
    else:
        while(True):
            print("\nEnter the proper commad")
            check = input("Do you want to get meaning of another word if yes press Y else N : ")
            if check == 'y' and 'Y':
                break
            elif check == 'n' and 'N':
                print("\nHave a Nice day")
                sign = False
                break
            else:
                pass
       
