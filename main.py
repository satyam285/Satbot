from tkinter import *
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open("intents.json") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)


def chat(inp):
    while True:
        if inp.lower() == "quit":
            break
        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        return random.choice(responses)



def Response(s):
    s = s[:-1]
    result = chat(s)
    result = result + "\n"
    chat_area.tag_configure("yellow",justify="left",foreground="yellow", font = ("Arial" , 10, "bold" ,"italic"),)
    chat_area.configure(state='normal')
    chat_area.insert(END, "Bot : ","yellow")
    chat_area.configure(fg="white",font = ("Arial" , 10))
    chat_area.insert(END, result)
    chat_area.configure(state='disabled')
    chat_area.see(END)

def Query():
    result = text_area.get("1.0","end")
    text_area.delete("1.0",END)
    # print(result)
    chat_area.tag_configure("right",justify="right")
    chat_area.tag_configure("green",justify="right",foreground="light green", font = ("Arial" , 10, "bold" ,"italic"),)
    chat_area.configure(state='normal')
    chat_area.insert(END, "You : ","green")
    chat_area.configure(fg="white",font = ("Arial" , 10))
    chat_area.insert(END, result,"right")
    chat_area.configure(state='disabled')
    chat_area.see(END)
    Response(result)

    # result = text_area.get("1.0","end -1c")
    # widget_width = 0
    # widget_height = float(text_area.index(END))-1
    # for line in result.split("\n"):
    #     if len(line) > widget_width:
    #         widget_width = len(line)+1

    # text_area.delete("1.0",END)

    # t = Text(chat_area,bd = 2,height = widget_height, width = widget_width , bg = 'light grey')
    # t.pack(anchor=E)
    # t.configure(state='normal')
    # t.insert(END, result)
    # t.configure(state='disabled')
    # chat_area.create_window(window = t)

    # Response(result)
    
root = Tk()
root.title('Satyam Bot')
root.geometry('400x500')

chat_area = Text(root, bd = 1, bg = 'grey', width = 20, height = 10,cursor = "arrow", state='disabled')
chat_area.place(x = 5 , y = 5 , width = 375, height = 440)

text_area = Text(root, bd = 1, bg = 'light grey', width = 15, height = 2)
text_area.place(x = 5 , y = 448 , width = 320, height = 47)

send_button = Button(root,text = 'Send',command=Query, bg = 'green' ,fg = 'white', activebackground = 'light green' , height = 5 ,width = 12, font = ('Arial' ,10) )
send_button.place(x = 330 , y = 448 , width = 65 , height = 47)

scrollbar = Scrollbar(root, command = chat_area.yview)
scrollbar.place(x = 382 , y = 5 ,height = 440 , width = 15)

chat_area.configure(yscrollcommand=scrollbar.set)
root.mainloop()
