%% 极限学习机在分类问题中的应用研究

%% 清空环境变量
%clear all
clc
warning off

load abalone  %%% 导入数据集  行表示样本 列表示属性+标签  （最后一列为标签） ****** 需变化数据集名称  
DATA_Origin=abalone;  %%%% 数据集统一置换为DATA_Origin   ****** 需变化数据集名称

Num_Train=120;  %%% 确定训练集数目    ****** 需改变训练数据集中的样本数目

[DATA_m,DATA_N]=size(DATA_Origin);  %%% 计算数据集的行列数.  DATA_m为样本数目；DATA_N为属性数+标签（最后一列）

Data_label_data=DATA_Origin(:,end);  %%% 提取出最后一列的所有元素，待计算类别数目
Data_label=unique(Data_label_data);  %%% 确定数据集的类别名称
Num_Label=length(Data_label); %%% 确定数据集的类别数目

%%%%% 按类别提出数据
for label_i=1:Num_Label
    Number_sequence=find(Data_label_data==Data_label(label_i)); %%% 确定类别在原数据集中的位置序号序列
    DATA_LABEL{label_i}=DATA_Origin(Number_sequence',:);
end

%%%%% 训练集中数据样本是按类别“均匀随机选取”方式确定，
Train_data=[];Test_data=[];  %%% 初始数据集设为空集

%%%%%% 计算各类别的样本数目
for labelii=1:Num_Label %%%% 训练样本均匀随机选取时按向下取整的数据量，防止溢出
    DATA_Label_i=DATA_LABEL{labelii};  %%%% 将结构体中的数据提出出来，为数组数据
    number_sample(labelii)=size(DATA_Label_i,1);%%%%% 计算该类别的样本总数目
    rate_sample(labelii)=number_sample(labelii)/DATA_m;  %%% 计算类别样本在总样本中的比例
    Numb_label_i(labelii)=floor(rate_sample(labelii)*Num_Train); %%% 计算训练集中各类样本的数量（四舍五入）    
end
Numb_label_i(1)=Num_Train-sum(Numb_label_i(2:end));  %%% 防止未必整数，故将多余的训练集作为第一类

for label_i=1:Num_Label
    DATA_Label_i=DATA_LABEL{label_i};  %%%% 将结构体中的数据提出出来，为数组数据
    a=randperm(size(DATA_Label_i,1));  %%%%%% 随机排序第i组便签数据，并选定其中部分数据组成训练集
    Train_data=[Train_data;DATA_Label_i(a(1:Numb_label_i(label_i))',:)];  %%% 按类别组成训练集
    Test_data=[Test_data;DATA_Label_i(a(Numb_label_i(label_i)+1:end)',:)];  %%% 按类别组成测试集
end
% 生成训练集
Fact_train = Train_data(:,1:end-1)';  %%%% 训练集属性
Label_train = Train_data(:,end)';  %%%% 训练集标签
% 生成测试集
Fact_test = Test_data(:,1:end-1)';  %%%% 测试集属性
Label_test = Test_data(:,end)'; %%%% 测试集标签

NetNo=100;FunNo=1;
switch FunNo
    case 1
       FunName= 'sig';
    case 2
       FunName= 'sin';
    case 3
       FunName= 'hardlim';
end

tic

%% ELM创建/训练
[IW,B,LW,TF,TYPE] = elmtrain(Fact_train,Label_train,NetNo,FunName,1);

%% ELM仿真测试
T_sim_1 = elmpredict(Fact_train,IW,B,LW,TF,TYPE);
T_sim_2 = elmpredict(Fact_test,IW,B,LW,TF,TYPE);

toc

subplot(121)
plot(1:size(Label_train,2),Label_train,'-b*')
hold on
plot(1:size(Label_train,2),T_sim_1,'-ro')
legend('训练集原标签','训练集预测标签')

subplot(122)
plot(1:size(Label_test,2),Label_test,'-b*')
hold on
plot(1:size(Label_test,2),T_sim_2,'-ro')
legend('测试集原标签','测试集预测标签')

%% 结果对比
result_1 = [Label_train' T_sim_1'];
result_2 = [Label_test' T_sim_2'];
% 训练集正确率
k1 = length(find(Label_train == T_sim_1));
n1 = length(Label_train);
Accuracy_1 = k1 / n1 * 100;
disp(['训练集正确率Accuracy = ' num2str(Accuracy_1) '%(' num2str(k1) '/' num2str(n1) ')'])
% 测试集正确率
k2 = length(find(Label_test == T_sim_2));
n2 = length(Label_test);
Accuracy_2 = k2 / n2 * 100;
disp(['测试集正确率Accuracy = ' num2str(Accuracy_2) '%(' num2str(k2) '/' num2str(n2) ')'])

%% 显示
count_B = length(find(Label_train == 1));
count_M = length(find(Label_train == 2));
rate_B = count_B / 500;
rate_M = count_M / 500;
total_B = length(find(DATA_Origin(:,2) == 1));
total_M = length(find(DATA_Origin(:,2) == 2));
number_B = length(find(Label_test == 1));
number_M = length(find(Label_test == 2));
number_B_sim = length(find(T_sim_2 == 1 & Label_test == 1));
number_M_sim = length(find(T_sim_2 == 2 & Label_test == 2));
% disp(['病例总数：' num2str(569)...
%       '  良性：' num2str(total_B)...
%       '  恶性：' num2str(total_M)]);
% disp(['训练集病例总数：' num2str(500)...
%       '  良性：' num2str(count_B)...
%       '  恶性：' num2str(count_M)]);
% disp(['测试集病例总数：' num2str(69)...
%       '  良性：' num2str(number_B)...
%       '  恶性：' num2str(number_M)]);
% disp(['良性乳腺肿瘤确诊：' num2str(number_B_sim)...
%       '  误诊：' num2str(number_B - number_B_sim)...
%       '  确诊率p1=' num2str(number_B_sim/number_B*100) '%']);
% disp(['恶性乳腺肿瘤确诊：' num2str(number_M_sim)...
%       '  误诊：' num2str(number_M - number_M_sim)...
%       '  确诊率p2=' num2str(number_M_sim/number_M*100) '%']);  



R = [];
for i = 50:50:500
    %% ELM创建/训练
    [IW,B,LW,TF,TYPE] = elmtrain(Fact_train,Label_train,i,'sig',1);
    
    %% ELM仿真测试
    T_sim_1 = elmpredict(Fact_train,IW,B,LW,TF,TYPE);
    T_sim_2 = elmpredict(Fact_test,IW,B,LW,TF,TYPE);
    
    %% 结果对比
    result_1 = [Label_train' T_sim_1'];
    result_2 = [Label_test' T_sim_2'];
    % 训练集正确率
    k1 = length(find(Label_train == T_sim_1));
    n1 = length(Label_train);
    Accuracy_1 = k1 / n1 * 100;
%     disp(['训练集正确率Accuracy = ' num2str(Accuracy_1) '%(' num2str(k1) '/' num2str(n1) ')'])
    % 测试集正确率
    k2 = length(find(Label_test == T_sim_2));
    n2 = length(Label_test);
    Accuracy_2 = k2 / n2 * 100;
%     disp(['测试集正确率Accuracy = ' num2str(Accuracy_2) '%(' num2str(k2) '/' num2str(n2) ')'])
    R = [R;Accuracy_1 Accuracy_2];
end
  
figure
plot(50:50:500,R(:,2),'b:o')
xlabel('隐含层神经元个数')
ylabel('测试集预测正确率（%）')
title('隐含层神经元个数对ELM性能的影响')


