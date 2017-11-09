import numpy as  np


class HMMModel(object):
    def __init__(self):
        self.__hiden_state_transfer= {}
        self.__hiden_observe_transfer = {}
        self.__init_hiden = {}
        self.__observes = set()
        self.__hiden_states = set()


    def train(self, train_data, pos_tags):
        init_state_cnt = {}#π
        hiden_observe_cnt = {}#B
        hiden_transfer_cnt = {}#A
        #初始化初始状态概率
        self.__hiden_states = set()
        self.__observes = set()
        for pos_tag in pos_tags:
            self.__init_hiden[pos_tag] = float(0)
            self.__hiden_state_transfer[pos_tag] = {}
            self.__hiden_observe_transfer[pos_tag] = {}
            self.__hiden_states.add(pos_tags)

            hiden_transfer_cnt[pos_tag] = {}
            hiden_observe_cnt[pos_tag] = {}
            for pos_tag_1 in pos_tags:
                hiden_transfer_cnt[pos_tag][pos_tag_1] = 0
                self.__hiden_state_transfer[pos_tag][pos_tag_1] = float(0)

        record_cnt = 0
        for train_record in train_data:
            record_cnt += 1
            is_start = True
            last_pos_tag = ''
            for word in train_record:
                word_str = word['word']
                pos_tag = word['pos_tag']
                if is_start == True:
                    is_start = False
                    # 统计初始状态
                    if pos_tag not in init_state_cnt:
                        init_state_cnt[pos_tag] = 1
                    else:
                        init_state_cnt[pos_tag] += 1
                    last_pos_tag = pos_tag
                    continue

                #统计隐藏状态到观测状态的转移概率
                if word_str not in hiden_observe_cnt[pos_tag]:
                    hiden_observe_cnt[pos_tag][word_str] = 1
                    self.__hiden_observe_transfer[pos_tag][word_str] = float(0)
                else:
                    hiden_observe_cnt[pos_tag][word_str] += 1

                # 统计隐藏状态之间的转换
                hiden_transfer_cnt[last_pos_tag][pos_tag] += 1

                # 添加至观测状态集合
                self.__observes.add(word_str)

            if record_cnt % 10000 == 0:
                print('Trained %s records'% record_cnt)
        print("--------------------finished counting----------------------")
        #计算初始概率
        print("--------------------compute init state_cnt ----------------------")
        total_count = float(0)
        for state in init_state_cnt:
            total_count += init_state_cnt[state]
        for state in init_state_cnt:
            self.__init_hiden[state] = init_state_cnt[state]/total_count
            print(' %s::%s' % (state, self.__init_hiden[state]))

        #计算隐藏到观测状态概率
        print("--------------------compute hiden_observe_cnt ----------------------")
        for pos_tag in hiden_observe_cnt:
            pos_total_count = 0
            print('pos_tag:%s ===>'%pos_tag)
            for word_str in hiden_observe_cnt[pos_tag]:
                pos_total_count += hiden_observe_cnt[pos_tag][word_str]
            for word_str in hiden_observe_cnt[pos_tag]:
                current_word_cnt = hiden_observe_cnt[pos_tag][word_str]
                self.__hiden_observe_transfer[pos_tag][word_str] = float(current_word_cnt)/pos_total_count
                print(' %s::%s' % (word_str, self.__hiden_observe_transfer[pos_tag][word_str]),end='')
            print("")

        # 计算隐藏状态转移概率
        print("--------------------compute hiden_transfer_cnt ----------------------")
        for pos_tag_from in hiden_transfer_cnt:
            cur_pos_total_cnt = 0
            print('pos_tag_from:%s ===>'%pos_tag_from)
            for pos_tag_to in hiden_transfer_cnt[pos_tag_from]:
                cur_pos_total_cnt += hiden_transfer_cnt[pos_tag_from][pos_tag_to]
            if cur_pos_total_cnt == 0:
                continue
            for pos_tag_to in hiden_transfer_cnt[pos_tag_from]:
                current_pos_cnt = hiden_transfer_cnt[pos_tag_from][pos_tag_to]
                self.__hiden_state_transfer[pos_tag_from][pos_tag_to] = float(current_pos_cnt)/cur_pos_total_cnt
                print(' %s::%s'%(pos_tag_to, self.__hiden_state_transfer[pos_tag_from][pos_tag_to]),end='')
            print("")






    def test(self,test_data):

        pass

    def __viterbi_search(self,words_seq):
        seq_length = len(words_seq)
        states_max_probs = [{}]*seq_length
        states_path = [{}]*seq_length

        

