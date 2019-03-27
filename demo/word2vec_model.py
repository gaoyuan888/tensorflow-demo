import logging
import os.path
import sys
import multiprocessing

from gensim.models import Word2Vec#pip install gensim
from gensim.models.word2vec import LineSentence
if __name__ == '__main__':
    
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))
    # check and process input arguments
    if len(sys.argv) < 4:
        print (globals()['__doc__'] % locals())
        sys.exit(1)
    #inp分好词的文件如2800g.txt, outp1输出的模型文件名，outp2格式化保存词向量模型的文件（一般用不到，但执行语句必须包含这一项）
    inp, outp1, outp2 = sys.argv[1:4]
    #https://blog.csdn.net/qq_35273499/article/details/79098689
    #https://blog.csdn.net/jerr__y/article/details/52967351
    model = Word2Vec(LineSentence(inp), size=400, window=5, min_count=5, workers=multiprocessing.cpu_count())
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)
#python word2vec_model.py 2800g.txt 2800a.model 2800a.vector
#opencc -i wiki_texts.txt -o test.txt -c t2s.json