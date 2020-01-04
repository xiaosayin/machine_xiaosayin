# machine_xiaosayin
Test_Single.py  为load weights 单独预测，生成csv文件  
One_second_Average.py 将单次预测的结果平均，集成学习，生成csv文件  
Densenet.py  为跑出好结果的network 和 train，数据增强也包含在其中  
Densesharp.py  为尝试的densesharp网络，同样进行了数据增强  
Sequential.py 为最初尝试的keras线性模型，两个简单的3D卷积层  
acc703.12.h5  load该权重，预测集精度能达到70.337%  
acc7058.06.h5  load该权重，预测集精度能达到70.58%    
acc703.csv   acc7058.csv 分别对应上述两个权重的预测结果（第二列）  
Average7037058.csv 为最终预测的结果  

# 需要修改路径的地方
Data.py 读数据，13，27，42行分别修改成对应训练集，测试集，和训练集label的路径  
One_second_Average.py  将要平均的csv文件的路径修改，也可添加更多的csv进行集成学习  
One_second_Average.py  170行，读test的路径需要修改成对应路径  
Densnet.py 和 Densesharp.py 保存权重的路径可以自定义修改  


# 参考代码
https://github.com/duducheng/DenseSharp   

