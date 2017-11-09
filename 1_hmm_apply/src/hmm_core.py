import numpy as  np



class HMMModel(object):
    def __init__(self):
        self.__hiden_state_transfer= {}
        self.__hiden_observe_transfer = {}
        self.__init_hiden = {}
        self.__observes = set()
        self.__hiden_states = set()
    def get_parameters(self):
        return (self.__hiden_state_transfer,self.__hiden_observe_transfer,self.__init_hiden,self.__observes,self.__hiden_states)

    def get_hiden_state_transfer(self):
        return self.__hiden_state_transfer

    def get_hiden_observe_transfer(self):
        return self.__hiden_observe_transfer

    def get_init_hiden(self):
        return self.__init_hiden

    def get_observes(self):
        return self.__observes

    def get_hiden_states(self):
        return self.__hiden_states

    def set_hiden_state_transfer(self,hiden_state_transfer):
        self.__hiden_state_transfer = hiden_state_transfer

    def set_hiden_observe_transfer(self,hiden_observe_transfer):
        self.__hiden_observe_transfer = hiden_observe_transfer

    def set_init_hiden(self, init_hiden):
        self.__init_hiden = init_hiden

    def set_observes(self,observes):
        self.__observes = observes

    def set_hiden_states(self, hiden_states):
        self.__hiden_states = hiden_states

    def train(self, train_data, pos_tags=None):
        #初始化初始两种状态概率
        self.__hiden_states = set()
        self.__observes = set()

        if pos_tags is None:
            pos_tags = set()
            for train_record in train_data:
                for word in train_record:
                    word_str = word['word']
                    pos_tag = word['pos_tag']
                    #添加至隐藏状态集合
                    pos_tags.add(pos_tag)
                    # 添加至观测状态集合
                    self.__observes.add(word_str)
        init_state_cnt = {}#π
        hiden_observe_cnt = {}#B
        hiden_transfer_cnt = {}#A

        for pos_tag in pos_tags:
            self.__init_hiden[pos_tag] = float(0)
            self.__hiden_state_transfer[pos_tag] = {}
            self.__hiden_observe_transfer[pos_tag] = {}
            self.__hiden_states.add(pos_tag)

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

                # 统计隐藏状态到观测状态的转移概率
                if word_str not in hiden_observe_cnt[pos_tag]:
                    hiden_observe_cnt[pos_tag][word_str] = 1
                    self.__hiden_observe_transfer[pos_tag][word_str] = float(0)
                else:
                    hiden_observe_cnt[pos_tag][word_str] += 1

                if is_start == True:
                    is_start = False
                    # 统计初始状态
                    if pos_tag not in init_state_cnt:
                        init_state_cnt[pos_tag] = 1
                    else:
                        init_state_cnt[pos_tag] += 1
                else:
                    # 统计隐藏状态之间的转换
                    hiden_transfer_cnt[last_pos_tag][pos_tag] += 1
                last_pos_tag = pos_tag

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

        total_cnt = 0
        correct_cnt = 0
        i=1
        for record in test_data:
            words = []
            pos_tags= []
            for word_item in record:
                word = word_item['word']
                pos_tag = word_item['pos_tag']
                words.append(word)
                pos_tags.append(pos_tag)
            predicted_tags = self.__viterbi_search(words)
            print('=====test record %s start======'%i)
            print('origin :%s' % pos_tags)
            print('predict:%s' % predicted_tags)
            print('raw record:%s'%record)
            print('=====test record %s end======'%i)
            for pos_tag,predicted_tag in zip(pos_tags, predicted_tags):
                total_cnt += 1
                if pos_tag == predicted_tag:
                    correct_cnt += 1

            if i%50==0:
                print('current total counts:%s' % total_cnt)
                print('current correct count:%s' % correct_cnt)
                print('current correct rate:%s' % (float(correct_cnt) / float(total_cnt)))
            i+=1

        print ('--------------------test finished------------------------')
        print('total counts:%s'%total_cnt)
        print('correct count:%s'%correct_cnt)
        print('correct rate:%s'%(float(correct_cnt)/float(total_cnt)))
        pass

    def __viterbi_search(self,words_seq):
        seq_length = len(words_seq)
        if seq_length == 0 :
            return []
        states_max_probs = []
        states_path = []
        for i in range(0,seq_length):
            states_max_probs.append({})
            states_path.append({})

        first_word = words_seq[0]
        for hiden_state in self.__hiden_states:
            if first_word not in self.__hiden_observe_transfer[hiden_state]:
                h_o_transfer_prob = self.__get_simulate_HOtrans_prob(first_word)
            else:
                h_o_transfer_prob = self.__hiden_observe_transfer[hiden_state][first_word]
            states_max_probs[0][hiden_state] = self.__init_hiden[hiden_state]*h_o_transfer_prob
            states_path[0][hiden_state] = 'Source'

        resize = 1
        for i in range(1, seq_length):
            cur_word = words_seq[i]
            max_max_prob = 0
            for hiden_state in self.__hiden_states:
                max_source = ''
                max_prob = 0
                for hiden_state_from in self.__hiden_states:
                    if cur_word not in self.__hiden_observe_transfer[hiden_state]:
                        h_o_transfer_prob = self.__get_simulate_HOtrans_prob(cur_word)
                    else:
                        h_o_transfer_prob = self.__hiden_observe_transfer[hiden_state][cur_word]
                    cur_prob = states_max_probs[i-1][hiden_state_from] * self.__hiden_state_transfer[hiden_state_from][hiden_state]*h_o_transfer_prob
                    if cur_prob >= max_prob:
                        max_prob = cur_prob
                        max_source = hiden_state_from
                #print('max——prob：：%s -> %s ::%s++++++++++max--source: %s'%(hiden_state,cur_word,max_prob,max_source))
                max_prob *= resize
                states_max_probs[i][hiden_state] = max_prob
                states_path[i][hiden_state] = max_source
                if max_prob > max_max_prob:
                    max_max_prob = max_prob
            #print('max_max_prob of word(%s):%s'%(cur_word,max_max_prob))
            if max_max_prob > 10000:
                resize = 0.0001
            elif max_max_prob < 0.0001:
                resize = 10000
            else:
                resize = 1
        max_prob = 0
        max_source = ''
        for hiden_state in self.__hiden_states:
            cur_prob = states_max_probs[seq_length-1][hiden_state]
            if cur_prob >= max_prob:
                max_prob =  cur_prob
                max_source = hiden_state

        best_path = [max_source]
        for i in range(seq_length-1,0,-1):
            cur_state = best_path[0]
            best_path.insert(0,states_path[i][cur_state])

        return best_path

    def __get_simulate_HOtrans_prob(self, cur_word):
        if cur_word not in self.__observes:
            #print('%s 未登录'%cur_word)
            return 1
        else:
            return 0

