import tf_idf
import numpy as np

# all_page 是所有网页的TF-IDF字典
# page_num 表示有几个网页
all_page, page_num = tf_idf.run()

query = "Iron man"

file_name = 'data.txt'

# 记录每篇文章的关键词
words1 = []
# 记录查询的关键词
words2 = []
# 查询与关键词汇总的词汇表
words = []
count = {}
# 关键词的词向量
vec1 = []
# 查询内容的词向量
vec2 = []
result = {}

# 记录查询内容
for word in query.split(' '):
    words2.append(word.lower())

for per in range(1, page_num):
    # 添加第 per 篇文章的词汇
    for word in all_page[str(per)]:
        words1.append(word[0].lower())
    words.extend(words1)
    words.extend(words2)
    # 记录词频
    for word in words:
        if word not in count:
            count[word] = 1
        else:
            count[word] += 1
    # 生成词向量
    for word in words:
        if word in words1:
            vec1.append(count[word])
        else:
            vec1.append(0)
        if word in words2:
            vec2.append(count[word])
        else:
            vec2.append(0)
    # 计算相似性
    result[str(per)] = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    words1 = []
    words = []
    vec1 = []
    vec2 = []
    count = {}

# 最终的展示结果
show_page = []
result_page = sorted(result.items(), key=lambda item: item[1], reverse=True)[:3]
for page in result_page:
    # 相似性大于0.5才收录
    # print(page[1])
    # if page[1] > 0.05:
        # 第【0】 项表示索引，即第几个网页
    # 但是只展示前三个结果也不错
    show_page.append(int(page[0]))

if len(show_page) == 0:
    print('Nothing...')
else:
    row = 1
    with open (file_name) as f:
        lines = f.readlines()
        for line in lines:
            # 如果当前的第row个网页在展示结果中 那么展示网页内容
            if row in show_page:
                print(line.strip('\n'))
            row += 1
