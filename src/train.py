import tensorflow as tf
import numpy as np
from model import ourmodel


def train(args, data_info, show_loss):
    train_data = data_info[0]
    eval_data = data_info[1]
    test_data = data_info[2]
    n_entity = data_info[3]
    nn = data_info[4]
    ripple_set = data_info[5]
    model = ourmodel(args, n_entity, nn)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(args.n_epoch):
            np.random.shuffle(train_data)
            start = 0
            while start < train_data.shape[0]:
                _, loss = model.train(
                    sess, get_feed_dict(args, model, train_data, ripple_set, start, start + args.batch_size))
                start += args.batch_size
                if show_loss:
                    print('%.1f%% %.4f' % (start / train_data.shape[0] * 100, loss))


            f = open('re.txt', 'a')
            f.write('\n'+'epoch '+str(step)+'\n')
            f.close()
            f = open('re.txt', 'a')
            f.write('\n'+'train ' + '\n')
            f.close()

            #print('epoch '+str(step))
            #print('train')
            train_auc, train_acc = evaluation(sess, args, model, train_data, ripple_set, args.batch_size)

            f = open('re.txt', 'a')
            f.write('\n' + 'eval ' + '\n')
            f.close()

            #print('eval')
            eval_auc, eval_acc = evaluation(sess, args, model, eval_data, ripple_set, args.batch_size)

            f = open('re.txt', 'a')
            f.write('\n' + 'test ' + '\n')
            f.close()

            #print('test')
            test_auc, test_acc = evaluation(sess, args, model, test_data, ripple_set, args.batch_size)
            f = open('re.txt', 'a')
            f.write('\n')
            f.write('epoch %d    auc_train: %.4f  acc_train: %.4f    auc_eval: %.4f  acc_eval: %.4f    auc_test: %.4f  acc_test: %.4f'
                  % (step, train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc))
            f.write('\n')
            f.close()
            print('epoch %d    auc_train: %.4f  acc_train: %.4f    auc_eval: %.4f  acc_eval: %.4f    auc_test: %.4f  acc_test: %.4f'
                  % (step, train_auc, train_acc, eval_auc, eval_acc, test_auc, test_acc))


def get_feed_dict(args, model, data, ripple_set, start, end):
    feed_dict = dict()
    feed_dict[model.items] = data[start:end, 1]
    feed_dict[model.labels] = data[start:end, 2]
    for i in range(args.n_hop):
        feed_dict[model.memories_h[i]] = [ripple_set[user][i][0] for user in data[start:end, 0]]
        feed_dict[model.memories_r[i]] = [ripple_set[user][i][1] for user in data[start:end, 0]]
        feed_dict[model.memories_t[i]] = [ripple_set[user][i][2] for user in data[start:end, 0]]
    return feed_dict


def evaluation(sess, args, model, data, ripple_set, batch_size):
    start = 0
    auc_list = []
    acc_list = []
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    while start < data.shape[0]:
        auc, acc ,TP0,FP0,TN0 ,FN0 = model.eval(sess, get_feed_dict(args, model, data, ripple_set, start, start + batch_size))
        TP+= TP0
        FP+=FP0
        TN+=TN0
        FN+=FN0
        auc_list.append(auc)
        acc_list.append(acc)
        start += batch_size
    Precise = float(TP) / float(TP + FP)
    Recall = float(TP) / float(TP + FN)
    f1 = (2 * Precise * Recall) / (Precise + Recall)
    f = open('re.txt', 'a')
    f.write('precise ' + str(Precise))
    f.write('\n')
    f.write('recall ' + str(Recall))
    f.write('\n')
    f.write('f1 ' + str(f1))
    f.write('\n')
    f.close()
    #print('precise ' + str(Precise))
    #print('recall ' + str(Recall))
    #print('f1 ' + str(f1))
    return float(np.mean(auc_list)), float(np.mean(acc_list))
