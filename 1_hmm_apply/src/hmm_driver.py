from dataIO import DataIO
from hmm_core import HMMModel

if __name__ == '__main__':
    option = 'testing'
    #option = 'cut_data'
    model = HMMModel()
    io_obj = DataIO()
    if option == 'training':
        #exp_data = io_obj.load_raw_data('../data/raw_data.txt')
        print('--------------------loading data------------------------')
        train_data = io_obj.load_formed_data('../data/train_data.txt').get_all_data()

        print('--------------------start training------------------------')
        model.train(train_data)
        io_obj.persist_model(model,'../data/model.txt')


    elif option == 'cut_data':
        #exp_data = io_obj.load_raw_data('../data/raw_test_data.txt')
        exp_data = io_obj.load_raw_data('../data/raw_data.txt')
        exp_data.data_cutting(8,2)
        io_obj.persit_data(exp_data.get_train_data(), '../data/train_data.txt')
        io_obj.persit_data(exp_data.get_test_data(), '../data/test_data.txt')
        io_obj.persit_data(exp_data.get_all_tags(), '../data/tags_in_use.txt')


    elif option == 'testing':
        print('--------------------loading data------------------------')
        #test_data = io_obj.load_formed_data('../data/test_data_error.txt').get_all_data()
        test_data = io_obj.load_formed_data('../data/test_data.txt').get_all_data()
        print('--------------------start testing------------------------')
        model = io_obj.load_model('../data/model.txt')
        model.test(test_data)