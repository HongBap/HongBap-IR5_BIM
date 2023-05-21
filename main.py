'''
N19DCCN067 - Ngô Sơn Hồng
N19dccn070 - Lê Quang Hùng
N19dccn112 - Nguyễn Thị Huỳnh My
'''
import math
import re
import time
from nltk import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

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
    clean_text = text.replace('\n', '')
    # Xoá khoảng trắng ở đầu và cuối
    clean_text = clean_text.strip()
    # Xoá khoảng cách thừa giữa các từ
    words = clean_text.split()
    clean_text = ' '.join(words)
    # Xoá số
    clean_text = re.sub(r'\d+', '', clean_text)
    # Tách văn bản giữa các tài liệu
    documents = clean_text.split("/")
    clean_documents = [doc.strip().lower() for doc in documents]
    # Xoá stop words
    clean_documents = remove_stopwords(clean_documents)
    clean_documents.pop()
    return clean_documents

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

def get_len_Vi(relevant_docs, term, docs):
    '''
    Tính |Vi|
    '''
    Vi = []
    for doc in relevant_docs:
        print(doc)
        doc_id = doc['doc_id']
        list = docs[doc_id - 1].split()
        if term in list:
            Vi.append(doc_id)
    return len(Vi)

def preweight(inverted_index, docs):
    '''
    Tính toán trọng số BIM ban đầu với công thức:
        c(t) = log((N-df + 0.5)/(df+0.5))
    '''
    N = len(docs)
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

def compute_RSV(weighten_inverted_index, query, docs):
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

def update_rsv(doc_id, new_rsv, relevant_docs):
    for doc in relevant_docs:
        if doc['doc_id'] == doc_id:
            doc['rsv'] = new_rsv
            break

def compute_RSV_after_estimate_pi(relevant_docs, weighten_inverted_index, query, docs):
    doc_ids = [doc['doc_id'] for doc in relevant_docs]
    for doc_id in doc_ids:
        rsv = 0  # Khởi tạo RSV cho mỗi tài liệu
        positive_terms = intersection_d_q(docs[doc_id - 1], query)
        if len(positive_terms) > 0:
            for term in positive_terms:
                term_index = find_term(weighten_inverted_index, term)
                if term_index is not None:
                    rsv += weighten_inverted_index[term_index]['c']
            update_rsv(doc_id, rsv, relevant_docs)

def estimate_ci(relevant_docs, query, weighten_inverted_index, inverted_index, docs):
    '''
    relevant_docs: tập văn bản phù hợp
    |Vi|: số lượng văn bản trong relevant_docs chứa term
    '''
    V = [doc['doc_id'] for doc in relevant_docs]
    N = len(docs)
    len_V = len(V)
    for term in query.split():
        len_Vi = get_len_Vi(relevant_docs, term, docs)
        if len_Vi <= 0: # weight = 0
            continue
        else: # xi = qi = 1: từ xuất hiện trong query và top văn bản đã chọn
            ni = get_df(inverted_index, term) # số văn bản chứa term
            pi = (len_Vi + 0.5) / (len_V + 1)
            ri = (ni - len_Vi + 0.5) / (N - len_V + 1)
            c_i = math.log((pi * (1 - ri)) / (ri * (1 - pi)))

        index = find_term(weighten_inverted_index, term)
        weighten_inverted_index[index]['c'] = c_i

def sort_by_RSV(evaluated_list):
    return sorted(evaluated_list, key=lambda x: -x['rsv'])

def get_top_rsv(weighten_inverted_index, queries, docs):
    top_rsv = 5
    result = []
    for query_idx, query in enumerate(queries):
        evaluated_list = compute_RSV(weighten_inverted_index, query, docs)
        sorted_evaluated_list = sort_by_RSV(evaluated_list)
        result.append({"query_id": query_idx + 1, "doc_list": sorted_evaluated_list[:top_rsv]})
    return result


if __name__ == "__main__":
    start_time = time.time()
    docs = read_file_and_modify("doc-text")
    queries = read_file_and_modify("query-text")

    # 1. Tạo chỉ mục ngược
    inverted_index = create_inverted_index(docs)
    weighten_inverted_index = preweight(inverted_index,docs)

    # 2. Tìm top 5 văn bản có rsv cao nhất cho mỗi truy vấn
    top_doc_rsv_of_queries = get_top_rsv(weighten_inverted_index, queries, docs)

    max_iterations = 10
    previous_result = []  # Kết quả trước đó
    for i, query_result in enumerate(top_doc_rsv_of_queries):
        query_id = query_result['query_id']
        top_docs = query_result['doc_list'][:5]

        relevant_docs = top_docs
        relevant_docs_copy = relevant_docs.copy()

        if len(top_docs) == 0:
            break
        for j in range(max_iterations):
            estimate_ci(relevant_docs, queries[i], weighten_inverted_index, inverted_index, docs)
            compute_RSV_after_estimate_pi(relevant_docs, weighten_inverted_index, queries[i], docs)
            relevant_docs_copy.sort(key=lambda x: x['rsv'], reverse=True)
            doc_ids_before = [doc['doc_id'] for doc in relevant_docs]
            doc_ids_after = [doc['doc_id'] for doc in relevant_docs_copy]
            if doc_ids_after == doc_ids_before:
                print("query ", i + 1)
                print(relevant_docs_copy)
                print('\n')
                break
            elif j == max_iterations - 1:
                print("query ", i + 1)
                print(relevant_docs)


    end_time = time.time()
    total_time = end_time - start_time
    print(f"Thời gian chạy chương trình: {total_time} giây")
