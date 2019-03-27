import jieba#pip install jieba
import jieba.analyse
import codecs#打开文件
def cut_words(sentence):
    #print sentence
    return " ".join(jieba.cut(sentence)).encode('utf-8')
f=codecs.open('2800.txt','r',encoding="utf8")
target = codecs.open("2800g.txt", 'w',encoding="utf8")
print ('open files')
line_num=1
line = f.readline()
while line:
    print('---- processing ', line_num, ' article----------------')
    line_seg = " ".join(jieba.cut(line))
    target.writelines(line_seg)
    line_num = line_num + 1
    line = f.readline()
f.close()
target.close()

#while line:
#    curr = []
#    for oneline in line:
#        #print(oneline)
#        curr.append(oneline)
#    after_cut = map(cut_words, curr)
#    target.writelines(after_cut)
#    print ('saved',line_num,'articles')
#    exit()
#    line = f.readline1()
#f.close()
#target.close()

# python Testjieba.py