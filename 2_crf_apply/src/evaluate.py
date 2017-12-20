
if __name__ == '__main__':
    result_file_name= 'C:/Users/Administrator/Desktop/CRF++-0.58/CRF++-0.58/result.txt'
    with open(result_file_name,'r',encoding='utf-8') as result_file:
        num_mark_nr= 0
        num_predict_nr = 0
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        i = 0
        for line in result_file:
            # print('line:%s'%i)
            i += 1
            items = line.strip().split()
            if len(items) < 3:
                continue
            mark_status = items[-2]
            predict_status = items[-1]
            if mark_status == 'B' :
                if predict_status == 'B':
                    TP += 1
                else:
                    FN += 1
            else:
                if predict_status == 'B':
                    FP += 1
                else:
                    TN += 1
        precision = (float(TP)/(TP+FP))
        recall = (float(TP)/(TP+FN))
        print('precision :%s'%precision)
        print('recall:%s'%recall)
        print('F1 value:%s'%(precision*recall*2/(precision + recall)))


