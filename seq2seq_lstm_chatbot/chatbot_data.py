import os
import re
import yaml
import requests, zipfile, io
from corpora_tools import *

def cornell_download(url):
    if not os.path.exists("cornell movie-dialogs corpus"):
        r = requests.get(url) 
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall()
    return

def kaggle_download(url):
    if not os.path.exists("chatbot_nlp/data"):
        r = requests.get(url) 
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall()
    return

# movie_conversations 텍스트에서 발화(utterance) 리스트를 얻음, re.sub의 패턴에 raw string 명시하도록 r 추가
def read_conversations():
    filename = "cornell movie-dialogs corpus/movie_conversations.txt"
    with open(filename, "r", encoding="ISO-8859-1") as fh:
        conversations_chunks = [line.split(" +++$+++ ") for line in fh]
    return [re.sub(r'[\[\]\']', '', el[3].strip()).split(", ") for el in conversations_chunks]

# 라인 번호와 그 라인의 대화를 반환
def read_lines():
    filename = "cornell movie-dialogs corpus/movie_lines.txt"
    with open(filename, "r", encoding="ISO-8859-1") as fh:
        line_chunks = [line.split(" +++$+++ ") for line in fh]
    return {line[0]: line[-1].strip() for line in line_chunks}

# 토큰화 정렬과정, 두 개의 발화로 구성된 튜플을 포함한 생성기(generators)를 반환, 발화는 space 기준으로 토큰화
def get_tokenized_sequencial_sentences(list_of_lines, line_text):
    for line in list_of_lines:
        for i in range(len(line) - 1):
            yield (line_text[line[i]].split(" "), line_text[line[i+1]].split(" "))

def kaggle_tokenized():
    dir_path = 'chatbot_nlp/data'
    files_list = os.listdir(dir_path + os.sep)

    questions = list()
    answers = list()

    ## 답변이 2개 이상일 경우 쪼개기
    for filepath in files_list:
        stream = open( dir_path + os.sep + filepath , 'rb')
        docs = yaml.safe_load(stream)
        conversations = docs['conversations']
        for con in conversations:
            if len( con ) > 2 :
                replies = con[ 1 : ]
                for rep in replies:
                    question_tokens = str(con[0]).split(" ")
                    questions.append(question_tokens)
                    rep_tokens = str(rep).split(" ")
                    answers.append( rep_tokens )
            elif len( con )> 1:
                question_tokens = str(con[0]).split(" ")
                questions.append(question_tokens)
                answer_tokens = str(con[1]).split(" ")
                answers.append( answer_tokens )

    return questions, answers

def retrieve_corpora_from_zip():
    cornell_download("http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip")
    conversations = read_conversations()
    lines = read_lines()
    questions, answers = zip(*list(get_tokenized_sequencial_sentences(conversations, lines)))

    # dataset 추가
    kaggle_download('https://github.com/shubham0204/Dataset_Archives/blob/master/chatbot_nlp.zip?raw=true')
    sen1, sen2 = kaggle_tokenized()
    questions = questions + tuple(sen1)
    answers = answers + tuple(sen2)

    return questions, answers
