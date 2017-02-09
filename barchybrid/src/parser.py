from optparse import OptionParser
from arc_hybrid import ArcHybridLSTM
import pickle, utils, os, time, sys

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="conll_train", help="Annotated CONLL train file", metavar="FILE", default="../data/PTB_SD_3_3_0/train.conll")
    parser.add_option("--dev", dest="conll_dev", help="Annotated CONLL dev file", metavar="FILE", default="../data/PTB_SD_3_3_0/dev.conll")
    parser.add_option("--test", dest="conll_test", help="Annotated CONLL test file", metavar="FILE", default="../data/PTB_SD_3_3_0/test.conll")
    parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="params.pickle")
    parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE")
    parser.add_option("--load_model", dest="load_model", help="Load model file", metavar="FILE", default="")
    parser.add_option("--load_params", dest="load_params", help="Load params file", metavar="FILE", default="")
    parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE", default="barchybrid.model")
    parser.add_option("--wembedding", type="int", dest="wembedding_dims", default=100)
    parser.add_option("--pembedding", type="int", dest="pembedding_dims", default=25)
    parser.add_option("--rembedding", type="int", dest="rembedding_dims", default=25)
    parser.add_option("--epochs", type="string", dest="epochs", default=30)
    parser.add_option("--hidden", type="int", dest="hidden_units", default=100)
    parser.add_option("--hidden2", type="int", dest="hidden2_units", default=0)
    parser.add_option("--k", type="int", dest="window", default=3)
    parser.add_option("--lr", type="float", dest="learning_rate", default=0.1)
    parser.add_option("--outdir", type="string", dest="output", default="results")
    parser.add_option("--activation", type="string", dest="activation", default="tanh")
    parser.add_option("--lstmlayers", type="int", dest="lstm_layers", default=2)
    parser.add_option("--lstmdims", type="int", dest="lstm_dims", default=200)
    parser.add_option("--cnn-seed", type="int", dest="seed", default=7)
    parser.add_option("--disableoracle", action="store_false", dest="oracle", default=True)
    parser.add_option("--disableblstm", action="store_false", dest="blstmFlag", default=True)
    parser.add_option("--bibi-lstm", action="store_true", dest="bibiFlag", default=False)
    parser.add_option("--usehead", action="store_true", dest="headFlag", default=False)
    parser.add_option("--userlmost", action="store_true", dest="rlFlag", default=False)
    parser.add_option("--userl", action="store_true", dest="rlMostFlag", default=False)
    parser.add_option("--predict", action="store_true", dest="predictFlag", default=False)
    parser.add_option("--feat_pos", action="store_true", dest="POSFlag", default=False)
    parser.add_option("--feat_cap", action="store_true", dest="CAPFlag", default=False)
    parser.add_option("--feat_lem", action="store_true", dest="LEMFlag", default=False)
    parser.add_option("--feat_suf", action="store_true", dest="SUFFlag", default=False)
    parser.add_option("--feat_sym", action="store_true", dest="symFlag", default=False)
    parser.add_option("--feat_wn", action="store_true", dest="wnFlag", default=False)
    parser.add_option("--lem_spacy", action="store_true", dest="spacyFlag", default=False)
    parser.add_option("--feat_sentiment", action="store_true", dest="sentimentFlag", default=False)
    parser.add_option("--feat_brown", action="store_true", dest="BrownFlag", default=False)
    parser.add_option("--cnn-mem", type="int", dest="cnn_mem", default=512)

    (options, args) = parser.parse_args()
    print 'Using external embedding:', options.external_embedding

    if not options.predictFlag:
        if not (options.rlFlag or options.rlMostFlag or options.headFlag):
            print 'You must use either --userlmost or --userl or --usehead (you can use multiple)'
            sys.exit()

        if options.load_model != "":
            with open(options.load_params, 'r') as paramsfp:
                words, w2i, pos, rels, stored_opt = pickle.load(paramsfp)

            stored_opt.external_embedding = options.external_embedding

            parser = ArcHybridLSTM(words, pos, rels, w2i, stored_opt)
            parser.Load(options.load_model)

        else:
            print 'Preparing vocab'
            words, w2i, pos, rels = utils.vocab(options.conll_train)

            with open(os.path.join(options.output, options.params), 'w') as paramsfp:
                pickle.dump((words, w2i, pos, rels, options), paramsfp)
            print 'Finished collecting vocab'

            print 'Initializing blstm arc hybrid:'
            parser = ArcHybridLSTM(words, pos, rels, w2i, options)

        for i, (epoch, train) in enumerate(zip(options.epochs.split(','), options.conll_train.split(',')), 1):
            for iepoch in range(1, int(epoch)+1):
                print 'Starting epoch', iepoch
                parser.Train(train)
                devpath = os.path.join(options.output, 'dev_epoch_' + str(i) + '_' + str(iepoch) + '.conll')
                utils.write_conll(devpath, parser.Predict(options.conll_dev))
                os.system('perl src/utils/eval.pl -g ' + options.conll_dev + ' -s ' + devpath  + ' > ' + devpath + '.txt &')
                print 'Finished predicting dev'
                parser.Save(os.path.join(options.output, options.model + '_' + str(i) + '_' + str(iepoch)))
    else:
        with open(options.params, 'r') as paramsfp:
            words, w2i, pos, rels, stored_opt = pickle.load(paramsfp)

        stored_opt.external_embedding = options.external_embedding

        parser = ArcHybridLSTM(words, pos, rels, w2i, stored_opt)
        parser.Load(options.model)
        tespath = os.path.join(options.output, 'test_pred.conll')
        ts = time.time()
        pred = parser.Predict(options.conll_test)
        te = time.time()
        utils.write_conll(tespath, pred)
        os.system('perl src/utils/eval.pl -g ' + options.conll_test + ' -s ' + tespath  + ' > ' + tespath + '.txt &')
        print 'Finished predicting test',te-ts

