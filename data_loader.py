# -*- coding: utf-8 -*-
import json
from os.path import isfile, join
import re
import numpy as np
import pickle
import h5py
import jieba

def prepare_training_data(data_dir='Data'):
    qa_json_file = join(data_dir, 'FM-CH-QA.json')
    qa_data_file = join(data_dir, 'qa_data_file.pkl')
    vocab_file = join(data_dir, 'vocab_file.pkl')
    
    if isfile(qa_data_file):
        with open(qa_data_file) as f:
             data = pickle.load(f)
        return data
    
    print("Loading Data")
    with open(qa_json_file) as f:
        qa = json.loads(f.read())
    
    print("train", len(qa['train']))
    print("val", len(qa['val']))
    
    # 合并所有数据
    all_data = qa['train'] + qa['val']
    # 找出前1000个备用答案
    answer_vocab = make_answer_vocab(all_data)
    # 建立问题词典
    question_vocab, max_question_length = make_questions_vocab(all_data, answer_vocab)
   
    print "Max Question Length", max_question_length
    
    training_data = []
    for i,content in enumerate(qa['train']):
        ans = content["Answer"]
        if ans in answer_vocab:
            training_data.append({
                'image_id' : content['image_id'],
                'question' : np.zeros(max_question_length),
                'answer' : answer_vocab[ans]
                })
            question_words = jieba.lcut(content['Question'])
            base = max_question_length - len(question_words)
            for i in range(0, len(question_words)):
                # 在training_data中添加数据
                training_data[-1]['question'][base + i] = question_vocab[ question_words[i] ]
    
    print "Training Data", len(training_data)
    val_data = []
    for i,content in enumerate(qa['val']):
        ans = content["Answer"]
        if ans in answer_vocab:
            val_data.append({
                'image_id' : content['image_id'],
                'question' : np.zeros(max_question_length),
                'answer' : answer_vocab[ans]
                })
            question_words = jieba.lcut(content['Question'])
            base = max_question_length - len(question_words)
            for i in range(0, len(question_words)):
                val_data[-1]['question'][base + i] = question_vocab[ question_words[i] ]
                
    print "Validation Data", len(val_data)
    data = {
        'training' : training_data,
        'validation' : val_data,
        'answer_vocab' : answer_vocab,
        'question_vocab' : question_vocab,
        'max_question_length' : max_question_length
        }

    print "Saving qa_data"
    with open(qa_data_file, 'wb') as f:
        pickle.dump(data, f)

    with open(vocab_file, 'wb') as f:
        vocab_data = {
            'answer_vocab' : data['answer_vocab'],
            'question_vocab' : data['question_vocab'],
            'max_question_length' : data['max_question_length']
            }
        pickle.dump(vocab_data, f)

    return data
	
def load_questions_answers(data_dir = 'Data'):
    qa_data_file = join(data_dir, 'qa_data_file.pkl')
    if isfile(qa_data_file):
        with open(qa_data_file) as f:
            data = pickle.load(f)
            return data

def get_question_answer_vocab( data_dir = 'Data'):
    vocab_file = join(data_dir, 'vocab_file.pkl')
    vocab_data = pickle.load(open(vocab_file))
    return vocab_data

def make_answer_vocab(qa):
    top_n = 1000
    answer_frequency = {} 
    for annotation in qa:
        answer = annotation['Answer']
        if answer in answer_frequency:
            answer_frequency[answer] += 1
        else:
            answer_frequency[answer] = 1
    answer_frequency_tuples = [ (-frequency, answer) for answer, frequency in answer_frequency.iteritems()]
    answer_frequency_tuples.sort()
    answer_frequency_tuples = answer_frequency_tuples[0:top_n-1]

    answer_vocab = {}
    for i, ans_freq in enumerate(answer_frequency_tuples):
        ans = ans_freq[1]
        answer_vocab[ans] = i
        
    answer_vocab['UNK'] = top_n-1

    return answer_vocab


def make_questions_vocab(qa, answer_vocab):
    # 用于英文分词
    # word_regex = re.compile(r'\w+')
    question_frequency = {}
    
    max_question_length = 0
    for i, content in enumerate(qa):
        ans = content['Answer']
        count = 0
        if ans in answer_vocab:
            # 分词 
            question_words = jieba.lcut(content['Question'])
            for qw in question_words:
                if qw in question_frequency:
                    question_frequency[qw] += 1
                else:
                    question_frequency[qw] = 1
                count += 1
        if count > max_question_length:
            max_question_length = count

    qw_freq_threhold = 0
    qw_tuples = [ (-frequency, qw) for qw, frequency in question_frequency.iteritems()]
	 # qw_tuples.sort()
     
    qw_vocab = {}
    for i, qw_freq in enumerate(qw_tuples):
        frequency = -qw_freq[0]
        qw = qw_freq[1]
        # print frequency, qw
        if frequency > qw_freq_threhold:
			# +1 for accounting the zero padding for batc training
            qw_vocab[qw] = i + 1
        else:
            break

    qw_vocab['UNK'] = len(qw_vocab) + 1

    return qw_vocab, max_question_length


def load_fc7_features(data_dir, split):
    fc7_features = None
    image_id_list = None
    with h5py.File( join( data_dir, (split + '_fc7.h5')),'r') as hf:
        fc7_features = np.array(hf.get('fc7_features'))
    with h5py.File( join( data_dir, (split + '_image_id_list.h5')),'r') as hf:
        image_id_list = np.array(hf.get('image_id_list'))
    return fc7_features, image_id_list

if __name__ == '__main__':
    prepare_training_data()