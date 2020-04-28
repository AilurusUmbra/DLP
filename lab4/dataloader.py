import json


class DataTrans:
    def __init__(self):
        self.char2idx=self.build_char2idx()
        self.idx2char=self.build_idx2char()
        self.MAX_LENGTH=0  # max length of the training data word(contain 'EOS')

    def build_char2idx(self):
        dictionary={'SOS':0,'EOS':1,'UNK':2}
        dictionary.update([(chr(i+97),i+3) for i in range(0,26)])
        return dictionary

    def build_idx2char(self):
        dictionary={0:'SOS',1:'EOS',2:'UNK'}
        dictionary.update([(i+3,chr(i+97)) for i in range(0,26)])
        return dictionary

    def seq2idx(self,sequence,add_eos=True):
        indices=[]
        for c in sequence:
            indices.append(self.char2idx[c])
        if add_eos:
            indices.append(self.char2idx['EOS'])
        self.MAX_LENGTH = max(self.MAX_LENGTH, len(indices))
        return indices

    def idx2seq(self,indices):
        re=""
        for i in indices:
            re+=self.idx2char[i]
        return re

    def build_training_set(self,path):
        int_list=[]
        str_list=[]
        with open(path,'r') as file:
            dict_list=json.load(file)
            for dict in dict_list:
                target=self.seq2idx(dict['target'])
                for input in dict['input']:
                    int_list.append([self.seq2idx(input,add_eos=True),target])
                    str_list.append([input,dict['target']])
        return int_list,str_list
