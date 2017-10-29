# -*- coding: utf-8 -*-
import sys


class ExpData(object):
    def __init__(self):
        self.__data = []
        self.__train_data = []
        self.__test_data = []
        self.__tag_set = set()

    def add_exp_data(self, record):
        self.__data.append(record)

    def add_tag(self, tag):
        self.__tag_set.add(tag)

    def data_cutting(self, train_ratio, test_ratio):
        choice = 'train'
        train_cnt = 0
        test_cnt = 0
        self.__test_data = []
        self.__train_data = []

        for data_item in self.__data:
            if choice == 'train':
                self.__train_data.append(data_item)
                train_cnt += 1
                if train_cnt > train_ratio:
                    train_cnt = 0
                    choice = 'test'
            elif choice == 'test':
                self.__test_data.append(data_item)
                test_cnt += 1
                if test_cnt > test_ratio:
                    test_cnt = 0
                    choice = 'train'
            else:
                print('-----------erro logic---------------')
        print('apply data_cutting with %s train data, %s test data' % (len(self.__train_data), len(self.__test_data)))

    def get_train_data(self):
        return self.__train_data

    def get_test_data(self):
        return self.__test_data

    def get_all_tags(self):
        return self.__tag_set

class DataIO(object):
    def __init__(self):
        self.POS_Tag_SPLIT = '/'

    def load_raw_data(self, file_path):
        exp_data = ExpData()
        with open(file_path, encoding='utf-8') as raw_file:
            i = 1
            try:
                for line in raw_file:
                    word_strings = line.strip().split()
                    exp_record = []
                    i += 1
                    for word_string in word_strings:
                        items = word_string.strip().rsplit(self.POS_Tag_SPLIT, maxsplit=1)
                        try:
                            word, pos_tag = items
                        except Exception as e:
                            word, pos_tag = ',', 'w'
                        if pos_tag.startswith('u'):
                            pos_tag = 'u'
                        print('at line %s : word=%s , pos_tag = %s' % (i, word, pos_tag))
                        exp_data.add_tag(pos_tag)
                        word_item = {}
                        word_item['word'] = word
                        word_item['pos_tag'] = pos_tag
                        exp_record.append(word_item)
                    exp_data.add_exp_data(exp_record)
            except Exception as e:
                print(e)
        return exp_data

    def persit_data(self, target_data, file_path):
        with open(file_path, 'w', encoding='utf-8') as out_file:
            for data_item in target_data:
                out_file.write('%s\n' % data_item)


if __name__ == '__main__':
    io_obj = DataIO()
    exp_data = io_obj.load_raw_data('C:\\Users\\Administrator\\Desktop\\raw_data.txt')
    exp_data.apply_cross_validate(8, 2)
    io_obj.persit_data(exp_data.get_train_data(), '../data/train_data.txt')
    io_obj.persit_data(exp_data.get_test_data(), '../data/test_data.txt')
    io_obj.persit_data(exp_data.get_test_data(), '../data/tags_in_use.txt')

