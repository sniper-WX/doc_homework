# -*- coding:utf-8 -*-
import random

if __name__ == '__main__':
    raw_file_name = '../data/data_nr.txt'
    train_file_name = '../data/train_data.txt'
    test_file_name = '../data/test_data.txt'
    train_out_file = open(train_file_name,'w')
    test_out_file = open(test_file_name,'w')
    test_ratio = 2#最大10
    with open(raw_file_name,'r') as in_file:
        i = 1
        for line in in_file:
            i += 1
            if i == 1000 :
                break
            random_int = random.randint(1, 10)
            if random_int <= test_ratio:
                out_file = test_out_file
            else:
                out_file = train_out_file

            words = line.strip().split()
            for word in words:
                if word.find('/nr') != -1:
                    word,nr_flag = word.split('/')
                    out_file.write('%s %s\n'%(word,'B'))
                else:
                    out_file.write('%s %s\n'%(word,'O'))
