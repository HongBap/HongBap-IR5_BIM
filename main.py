import math
import re
import time

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# Xử lý file doc và query
def remove_stopwords(documents):
    filtered_documents = []
    for doc in documents:
        words = word_tokenize(doc) # tách vb thành các từ riêng biệt
        filtered_words = [word for word in words if word.lower() not in stop_words]
        filtered_documents.append(' '.join(filtered_words))
    return filtered_documents


def read_file_and_modify(filename):
    with open(filename, 'r') as f:
        text = f.read()
    clean_text = re.sub(r'\n', '', text)
    clean_text = re.sub(r'\d+', '', clean_text)
    clean_text = clean_text.strip().split("/")
    clean_text = [doc.strip().lower() for doc in clean_text]
    return clean_text

#tạo chỉ mục ngược
def create_inverted_index(docs):
    '''
    Khởi tạo Inverted Index từ các docs
    Inverted Index sẽ có dạng {'từ khóa': [tài liệu 1, tài liệu 2, ...]}
    '''
    inverted_index = {}
    for i in range(0, len(docs), 1):
        words = docs[i].lower().split()
        for word in words:
            if word not in inverted_index:
                inverted_index[word] = []
            if (i + 1) not in inverted_index[word]:
                inverted_index[word].append(i + 1)
    return inverted_index


def get_df(inverted_index, term):
    '''
    trả về số lượng doc chứ từ khóa term tong inverted_index
    '''
    if term in inverted_index:
        return len(inverted_index[term])
    return 0


def intersection_d_q(doc, query):
    '''
    tìm và trả về danh sách các từ chung (giao) giữa doc và query.
    '''
    return list(set(doc.split()) & set(query.split()))

def preweight(inverted_index, docs):
    '''
    Tính toán trọng số BIM ban đầu dựa trên công thức:
    c(t) = log((N-df + 0.5)/(df+0.5))
    '''
    N = len(docs) - 1
    weight_i_index = []
    for term in inverted_index:
        term_df = get_df(inverted_index, term)
        c_t = math.log((N - term_df + 0.5) / (term_df + 0.5))
        weight_i_index.append({term: inverted_index[term], "c": c_t})
    return weight_i_index

def find_term(weighten_inverted_index, term):
    '''
    Tìm từ trong inverted index, trả về index của term đó,
    trả về -1 nếu không tìm được`
    '''
    for index in range(len(weighten_inverted_index)):
        item_term = list(weighten_inverted_index[index].keys())[0]
        if (item_term == term):
            return index
    return -1

def query_BIM(weighten_inverted_index, query, docs):
    '''
    Tính toán RSV là tổng các weight term trong querry
    '''
    docs_computed_RSV = []
    for doc_index in range(len(docs)):
        positive_terms = intersection_d_q(docs[doc_index], query)
        if (len(positive_terms) == 0):
            docs_computed_RSV.append({"doc_id": doc_index + 1, "rsv": 0})
        else:
            rsv = 0.0
            for term in positive_terms:
                term_index = find_term(weighten_inverted_index, term)
                rsv += weighten_inverted_index[term_index]['c']
            docs_computed_RSV.append({"doc_id": doc_index + 1, "rsv": rsv})
    return docs_computed_RSV


def sort_by_RSV(evaluated_list):
    return sorted(evaluated_list, key=lambda x: -x['rsv'])

if __name__ == "__main__":
    start_time = time.time()
    docs = read_file_and_modify("doc-text")
    queries = read_file_and_modify("query-text")
    # phần tử cuối là rỗng, nên bỏ
    queries = queries[:-1]
    # Filter stop words
    docs = remove_stopwords(docs)
    queries = remove_stopwords(queries)

    inverted_index = create_inverted_index(docs)  # {'từ khóa': [tài liệu 1, tài liệu 2, ...]}
    weighten_inverted_index = preweight(inverted_index,docs)
    results = []
    for query_idx, query in enumerate(queries):
        evaluated_list = query_BIM(weighten_inverted_index, query, docs)
        sorted_evaluated_list = sort_by_RSV(evaluated_list)
        results.append({"query_id": query_idx + 1, "doc_list": sorted_evaluated_list[:5]})
    print(results)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Thời gian chạy chương trình: {total_time} giây")
