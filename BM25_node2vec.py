# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

data = pd.read_csv('./interst_freq_gamble_porn.csv')
print('Missing：\n', data.isnull().sum())


#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from numpy import linalg as LA
#import sklearn.preprocessing as prep
#import math
#import seaborn as sns

# 商品消费分布图
# =============================================================================
# fig = plt.figure(figsize=(16, 9))
# for i, col in enumerate(list(data.columns)[1:]):
#     plt.subplot(321+i)
#     q95 = np.percentile(data[col], 95)
#     sns.distplot(data[data[col] < q95][col])
# plt.show()
# =============================================================================

#features=data[['aa_count','aa_amount','ab_count','ab_amount','ac_count', 'ac_amount', 'ad_count', 'ad_amount', 'ae_count', 'ae_amount', 'af_count', 'af_amount', 'ag_count', 'ag_amount', 'am_count', 'am_amount', 'ai_count', 'ai_amount', 'aj_count', 'aj_amount', 'ak_count', 'ak_amount',  'an_count', 'an_amount', 'ao_count', 'ao_amount', 'ap_count', 'ap_amount', 'aq_count', 'aq_amount', 'as_count', 'as_amount', 'at_count', 'at_amount', 'au_count', 'au_amount', 'aw_count', 'aw_amount', 'av_count', 'av_amount']]
features=data[['aa01','aa04','aa05','aa06','aa07','aa08','aa09','aa10','aa11',
 'aa12',  'aa14', 'aa15', 'aa17', 'aa18',
'aa19',
 'aa20',
 'ab01',
 'ab02',
 'ab03',
 'ac01',
 'ac02',
'ac05',
'ad01',
'ad02',
'ad03',
'ad04',
'ad05',
'ad06',
'aa01',
'aa02',
'aa03',
'af01',
 'ag01',
 'ag02',
 'ag03',
 'ag04',
 'ai01',
 'ai02',
 'ai03',
 'aj01',
 'aj04',
 'aj05',
 'aj07',
 'ak01',
 'ak02',
 'ak03',
 'ak04','ak05',
 'ak06',
 'ak07',
 'ak08',
 'al01',
 'al02',
 'al05',
 'al06',
 'al08',
 'al11',
 'al15',
'am01',
'an01',
 'ao01',
 'ap01',
 'ap02',
 'aq01',
 'as09',
 'at01',
 'at02',
 'at04',
 'at05',
 'at06',
 'au01',
 'av01',
 'av02',
 'av03',
 'av04',
 'av05',
 'gamble_flag',
 'porn_flag']]

#mapmean=['商贸/超市/综合零售','便利店/杂货店','自动售卖机','百货商城/免税商店/折扣商店','户外/运动/健身器材/安防','鲜花/盆栽/室内装饰品','书籍/音像/文具/乐器','交通工具/配件/改装','贵重珠宝/首饰/钟表','宠物/宠物食品/饲料','数码家电/办公设备或服务','母婴用品/儿童玩具','家装建材/家居家纺','服饰/箱包/饰品','钟表/眼镜','化妆品/护肤品','电子器件/机械设备','日用品/个人护理','保健/医疗器材','大学校园服务/校园卡','幼中小学校','教育/培训/考试缴费/学费/','保健药/药店/中草药','私立/民营医院/诊所','保健信息咨询平台','挂号平台/医保/社保报销','公立医院','游戏/电竞','电影票/剧场票','健身/运动/美容美发/按摩推拿','酒吧/夜总会','KTV/网吧','在线增值服务（红钻/绿钻）/网络虚拟服务（在线视频/音乐/直播）','在线图书','阅读（知识经济）','客运站（汽车/船运）','地铁票','火车票','城市交通/高速收费/etc','旅游平台','wifi旅游服务','航空公司/机票代理','景点/门票','旅馆','酒店','度假区/别墅','招待所/联络部','装饰/设计/景观美化/园艺服务','婚庆/摄影','家政/维修','邮政/快递物流服务/O2O货运物流','政务办理（交警罚款/出入境/税务）','咨询/法律咨询/金融咨询','人才中介机构/招聘/猎头','兽医服务','安防/保安/侦探/其他服务','职业社交/婚介/交友','移动充电/共享服务','同城O2O服务平台','住房服务（租售）','综合生活服务信息平台','兼职或求职','广告/会展/活动策划/培训；文物经典/艺术收藏品','手机充值','公司业务','理财通','证券/期货/p2p','保险','信贷产品还款（不含信用卡）','甜品冷饮/面包/食品','茶馆/咖啡馆/西餐吧','快餐小吃/自助餐/摊贩','旅游饭店','美食城','围餐/餐厅（中西）','公益/慈善/众筹','加油站（粤通卡充值/加油卡）','车辆4S/保养/养护/维修/器件购买','汽车充电/充气/代驾','货车服务/洗车服务','汽车资讯平台']
#features=data[['aa_count','aa_amount','ab_count','ab_amount','ac_count', 'ac_amount', 'ad_count', 'ad_amount', 'ae_count', 'ae_amount', 'af_count', 'af_amount', 'ag_count', 'ag_amount', 'am_count', 'am_amount', 'ai_count', 'ai_amount', 'aj_count', 'aj_amount', 'ak_count', 'ak_amount', 'al_count', 'al_amount', 'an_count', 'an_amount', 'ao_count', 'ao_amount', 'ap_count', 'ap_amount', 'aq_count', 'aq_amount', 'as_count', 'as_amount', 'at_count', 'at_amount', 'au_count', 'au_amount', 'aw_count', 'aw_amount', 'av_count', 'av_amount']]
# 1:线下零售 2：教育培训 3：医疗卫生 4：生活缴费 5：电商团购 6：游戏 7：休闲娱乐 8：文化艺术 9：网络虚拟服务 10：交通出行 11：票务商旅酒店 12：生活服务 13：手机充值 14：公司其他业务 15：金融理财 16：金融保险 17：金融借贷 18：餐饮 19：慈善 20：其他 21：汽车服务
# 计算每一列的平均值
#meandata = np.mean(features, axis=0)  
# 均值归一化

mapmean=['商贸/超市/综合零售','百货商城/免税商店/折扣商店','户外/运动/健身器材/安防','鲜花/盆栽/室内装饰品','书籍/音像/文具/乐器','交通工具/配件/改装','贵重珠宝/首饰/钟表','宠物/宠物食品/饲料','数码家电/办公设备或服务','母婴用品/儿童玩具','服饰/箱包/饰品','钟表/眼镜','化妆品/护肤品','电子器件/机械设备',
'熟食', '生鲜果蔬',
'大学校园服务/校园卡','幼中小学校','教育/培训/考试缴费/学费/','保健药/药店/中草药','私立/民营医院/诊所','公立医院', 
'水电煤缴费',
'有线电视缴费',
'物业管理费',
'电信/宽带/话费缴费',
'停车缴费',
'自助缴费机',
'团购平台（如拼多多）',
'在线商城',
'外卖平台（如美团）',
'游戏/电竞','电影票/剧场票','健身/运动/美容美发/按摩推拿','酒吧/夜总会','KTV/网吧','在线增值服务（红钻/绿钻）/网络虚拟服务（在线视频/音乐/直播）','在线图书','阅读（知识经济）','客运站（汽车/船运）','租车共享车','火车票','城市交通/高速收费/etc','旅游平台','wifi旅游服务','航空公司/机票代理','景点/门票','旅馆','酒店','度假区/别墅','招待所/联络部','装饰/设计/景观美化/园艺服务','婚庆/摄影','政务办理（交警罚款/出入境/税务）','咨询/法律咨询/金融咨询','兽医服务','职业社交/婚介/交友','综合生活服务信息平台',
'手机充值',
'广告/会展/活动策划/培训；文物经典/艺术收藏品','腾讯充值','理财通','证券/期货/p2p','保险','信贷产品还款（不含信用卡）','甜品冷饮/面包/食品','茶馆/咖啡馆/西餐吧','旅游饭店','美食城','围餐/餐厅（中西）','公益/慈善/众筹','加油站（粤通卡充值/加油卡）','车辆4S/保养/养护/维修/器件购买','汽车充电/充气/代驾','货车服务/洗车服务','汽车资讯平台']


features=features.values
features=features.astype(float)
#features = prep.normalize(features,norm='l2',axis=0)
#features=np.real(features)
columnssize=np.size(features,1)
rowsize=np.size(features,0)
storefeatures=np.copy(features)
newfeatures=[]


for i in range(rowsize):
    if np.count_nonzero(features[i,0:columnssize-2])>4:
        newfeatures.append(features[i,:].tolist())


dnfeat_temp=np.array(newfeatures)
columnssize=np.size(newfeatures,1)
rowsize=np.size(newfeatures,0)
dnfeat=dnfeat_temp[:,0:columnssize-2]
dnfeat_label=dnfeat_temp[:,columnssize-2]
dnfeat_label_porn=dnfeat_temp[:,columnssize-1]
columnssize=np.size(dnfeat,1)
rowsize=np.size(dnfeat,0)
storednfeat=np.copy(dnfeat)


def TF_IDF(X):
    X_rows=np.size(X,0)
    X_columns=np.size(X,1)
    Y=np.empty([X_rows,X_columns])
    Y=Y.astype(float)
    IDF_value=[]
    for j in range(X_columns):
        IDF_value.append(np.log(X_rows/(np.size(X[:,j].nonzero())+1))) 
    for i in range(X_rows):
        user_ttl_cnt=np.sum(X[i,:])
        for j in range(X_columns):
               Y[i,j]=X[i,j]*IDF_value[j]/user_ttl_cnt
    return Y


def TFIDF_IDF(X):
    X_rows=np.size(X,0)
    X_columns=np.size(X,1)
    Y=np.empty([X_rows,X_columns])
    Y=Y.astype(float)
    IDF_value=[]
    for j in range(X_columns):
        IDF_value.append(np.log(X_rows/(np.size(X[:,j].nonzero())+1))) 
    for i in range(X_rows):
        user_ttl_cnt=np.sum(X[i,:])
        User_IDF=0
        for j in range(X_columns):
               Y[i,j]=X[i,j]*IDF_value[j]/user_ttl_cnt
               User_IDF+=Y[i,j]
        Y[i,:]=Y[i,:]/User_IDF
    return Y

def BM25(X,k,b):
    X_rows=np.size(X,0)
    X_columns=np.size(X,1)
    Y=np.empty([X_rows,X_columns])
    Y=Y.astype(float)
    IDF_value=[]
    AVG_length=np.mean(np.sum(X,1))
    for j in range(X_columns):
        IDF_value.append(np.log(X_rows/(np.size(X[:,j].nonzero())+1))) 
    for i in range(X_rows):
        user_ttl_cnt=np.sum(X[i,:])
        for j in range(X_columns):
            tf=X[i,j]/user_ttl_cnt
            Y[i,j]=IDF_value[j]*(tf*(k+1))/ (tf + k * (1 - b + b * (user_ttl_cnt/AVG_length)))
    return Y


   
dnfeat=BM25(dnfeat,1.6,0.72)
#xindex =  dnfeat.nonzero()
#totaldata=dnfeat[xindex]
#n_bins=20
#fig = plt.figure()
#axs=fig.add_subplot(1,1,1)
# We can set the number of bins with the `bins` kwarg
#axs.hist(totaldata, bins=n_bins)
#plt.show()


import networkx as nx


graph = nx.Graph()
formatted_positions = set()

def matrix2graph(matrix_input, graph):
    length_of_user=np.size(matrix_input,0)
    length_of_sp=np.size(matrix_input,1)
          
    # Build the graph
    for i in range(length_of_user):
        for j in range(length_of_sp):
            if matrix_input[i,j]>0.0001:
                graph.add_edge(i+1, length_of_user+j+1, weight=matrix_input[i,j])
             
    return graph

graph=matrix2graph(dnfeat,graph)

import edges
from node2vec import Node2Vec



# =============================================================================
# import argparse
# import node2vec
# from gensim.models import Word2Vec
# =============================================================================

# =============================================================================
# def parse_args():
# 	'''
# 	Parses the node2vec arguments.
# 	'''
# 	parser = argparse.ArgumentParser(description="Run node2vec.")
# 
# 	parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
# 	                    help='Input graph path')
# 
# 	parser.add_argument('--output', nargs='?', default='emb/karate.emb',
# 	                    help='Embeddings path')
# 
# 	parser.add_argument('--dimensions', type=int, default=128,
# 	                    help='Number of dimensions. Default is 128.')
# 
# 	parser.add_argument('--walk-length', type=int, default=80,
# 	                    help='Length of walk per source. Default is 80.')
# 
# 	parser.add_argument('--num-walks', type=int, default=10,
# 	                    help='Number of walks per source. Default is 10.')
# 
# 	parser.add_argument('--window-size', type=int, default=10,
#                     	help='Context size for optimization. Default is 10.')
# 
# 	parser.add_argument('--iter', default=1, type=int,
#                       help='Number of epochs in SGD')
# 
# 	parser.add_argument('--workers', type=int, default=8,
# 	                    help='Number of parallel workers. Default is 8.')
# 
# 	parser.add_argument('--p', type=float, default=1,
# 	                    help='Return hyperparameter. Default is 1.')
# 
# 	parser.add_argument('--q', type=float, default=1,
# 	                    help='Inout hyperparameter. Default is 1.')
# 
# 	parser.add_argument('--weighted', dest='weighted', action='store_true',
# 	                    help='Boolean specifying (un)weighted. Default is unweighted.')
# 	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
# 	parser.set_defaults(weighted=True)
# 
# 	parser.add_argument('--directed', dest='directed', action='store_true',
# 	                    help='Graph is (un)directed. Default is undirected.')
# 	parser.add_argument('--undirected', dest='undirected', action='store_false')
# 	parser.set_defaults(directed=False)
# 
# 	return parser.parse_args()
# 
# def learn_embeddings(walks):
# 	'''
# 	Learn embeddings by optimizing the Skipgram objective using SGD.
# 	'''
# 	walks = [map(str, walk) for walk in walks]
# 	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
# 	model.save_word2vec_format(args.output)
# 	
# 	return
# 
# 
# args = parse_args()
# #nx_G = read_graph()
# G = node2vec.Graph(graph, args.directed, args.p, args.q)
# G.preprocess_transition_probs()
# walks = G.simulate_walks(args.num_walks, args.walk_length)
# learn_embeddings(walks)
# =============================================================================




nvc = Node2Vec(graph, dimensions=20, walk_length=16, num_walks=100, workers=2)

# Embed nodes
model = nvc.fit(window=10, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)

# =============================================================================
# # Look for most similar nodes
# model.wv.most_similar('2')  # Output node names are always strings
# 
# # Save embeddings for later use
# model.wv.save_word2vec_format('./EMBEDDING_FILENAME')
# 
# # Save model for later use
# model.save('./EMBEDDING_MODEL_FILENAME')
# =============================================================================
