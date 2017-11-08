from dataIO import DataIO
from hmm_core import HMMModel

if __name__ == '__main__':
    option = 'training'
    model = HMMModel()
    if option == 'training':
        io_obj = DataIO()
        #exp_data = io_obj.load_raw_data('../data/raw_data.txt')
        exp_data = io_obj.load_raw_data('../data/raw_test_data.txt')

        exp_data.data_cutting(8,2)
        io_obj.persit_data(exp_data.get_train_data(),'../data/train_data.txt')
        io_obj.persit_data(exp_data.get_test_data(), '../data/test_data.txt')
        io_obj.persit_data(exp_data.get_all_tags(), '../data/tags_in_use.txt')
        model.train(exp_data.get_train_data(), exp_data.get_all_tags())

    elif option == 'testing':
        pass