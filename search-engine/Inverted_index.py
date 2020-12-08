import json

class data_preprocess:
    
    def __init__(self):
        self.file_name = 'data.txt'
        self.outfile = 'data.json'
        self.word_index = 1
        self.index_dict = {}
        self.word_num = 0

    def process(self):
        with open (self.file_name) as f:
            # 按行算文本
            contents = f.readlines()
            for sentence in contents:
                # 去除行末的空格 防止算入单词
                sentence = sentence.strip('\n')
                # 按空格分隔为单词
                words = sentence.split(' ')
                for word in words:
                    word.strip()
                    # 单词作为字典的key append内容为第几个网页 
                    # 比如 and 单词在 第 1，3，5，6，7，8 个网页中出现过
                    # 就是 key 为 and ，值为 [1, 3, 5, 6, 7, 8] 的列表
                    self.index_dict.setdefault(word.lower(), []).append(self.word_index)
                # 因为数据库内的一行内容记作一个网页 所以读完一行后网页个数加一，作为下一个网页的标记
                self.word_index += 1
        # 返回字典 和 网页数量
        return (self.index_dict, self.word_index)

    def save_data(self):
        with open(self.outfile, 'w') as f:
            json.dump(self.index_dict, f, indent = 4) 
