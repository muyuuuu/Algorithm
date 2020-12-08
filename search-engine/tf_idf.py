import math
from Inverted_index import data_preprocess

def run():
    file_name = 'data.txt'

    data = data_preprocess()

    # word_dict 返回数据结构 
    # article 返回文档数量
    word_dict, article_num = data.process()

    # 只是保存数据
    data.save_data()

    # 记录结果
    tf_article = {}

    # 先初始化为0
    for article in range(1, article_num):
        tf_article[str(article)] = 0
    
    article = 1
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip('\n')
            words = line.split(' ')
            # 每个网页的词汇量
            length = len(words)
            temp = {}
            for word in words:
                # 去除单词两边的空格
                word.strip()
                # 开始记录词频
                # 单词转小写，防止因大小写无法区分而导致搜索失败
                if word not in temp.keys():
                    temp[word.lower()] = 1
                else:
                    temp[word.lower()] += 1
            # 计算 TF-IDF
            for word in temp.keys():
                if word in word_dict.keys():
                    temp[word] = (temp[word] / length) * math.log10(article_num / len(word_dict[word]) + 1)
                else:
                    temp[word] = (temp[word] / length) * math.log10(article_num)
            # 保留值最大的 4 个
            tf_article[str(article)] = sorted(temp.items(), key=lambda item: item[1], reverse=True)[:4]
            # 进入下一个网页
            article += 1

    return tf_article, article_num