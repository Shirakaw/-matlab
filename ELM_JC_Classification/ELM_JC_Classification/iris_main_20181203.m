%% ����ѧϰ���ڷ��������е�Ӧ���о�

%% ��ջ�������
%clear all
clc
warning off

load abalone  %%% �������ݼ�  �б�ʾ���� �б�ʾ����+��ǩ  �����һ��Ϊ��ǩ�� ****** ��仯���ݼ�����  
DATA_Origin=abalone;  %%%% ���ݼ�ͳһ�û�ΪDATA_Origin   ****** ��仯���ݼ�����

Num_Train=120;  %%% ȷ��ѵ������Ŀ    ****** ��ı�ѵ�����ݼ��е�������Ŀ

[DATA_m,DATA_N]=size(DATA_Origin);  %%% �������ݼ���������.  DATA_mΪ������Ŀ��DATA_NΪ������+��ǩ�����һ�У�

Data_label_data=DATA_Origin(:,end);  %%% ��ȡ�����һ�е�����Ԫ�أ������������Ŀ
Data_label=unique(Data_label_data);  %%% ȷ�����ݼ����������
Num_Label=length(Data_label); %%% ȷ�����ݼ��������Ŀ

%%%%% ������������
for label_i=1:Num_Label
    Number_sequence=find(Data_label_data==Data_label(label_i)); %%% ȷ�������ԭ���ݼ��е�λ���������
    DATA_LABEL{label_i}=DATA_Origin(Number_sequence',:);
end

%%%%% ѵ���������������ǰ���𡰾������ѡȡ����ʽȷ����
Train_data=[];Test_data=[];  %%% ��ʼ���ݼ���Ϊ�ռ�

%%%%%% ���������������Ŀ
for labelii=1:Num_Label %%%% ѵ�������������ѡȡʱ������ȡ��������������ֹ���
    DATA_Label_i=DATA_LABEL{labelii};  %%%% ���ṹ���е��������������Ϊ��������
    number_sample(labelii)=size(DATA_Label_i,1);%%%%% �����������������Ŀ
    rate_sample(labelii)=number_sample(labelii)/DATA_m;  %%% ��������������������еı���
    Numb_label_i(labelii)=floor(rate_sample(labelii)*Num_Train); %%% ����ѵ�����и����������������������룩    
end
Numb_label_i(1)=Num_Train-sum(Numb_label_i(2:end));  %%% ��ֹδ���������ʽ������ѵ������Ϊ��һ��

for label_i=1:Num_Label
    DATA_Label_i=DATA_LABEL{label_i};  %%%% ���ṹ���е��������������Ϊ��������
    a=randperm(size(DATA_Label_i,1));  %%%%%% ��������i���ǩ���ݣ���ѡ�����в����������ѵ����
    Train_data=[Train_data;DATA_Label_i(a(1:Numb_label_i(label_i))',:)];  %%% ��������ѵ����
    Test_data=[Test_data;DATA_Label_i(a(Numb_label_i(label_i)+1:end)',:)];  %%% �������ɲ��Լ�
end
% ����ѵ����
Fact_train = Train_data(:,1:end-1)';  %%%% ѵ��������
Label_train = Train_data(:,end)';  %%%% ѵ������ǩ
% ���ɲ��Լ�
Fact_test = Test_data(:,1:end-1)';  %%%% ���Լ�����
Label_test = Test_data(:,end)'; %%%% ���Լ���ǩ

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

%% ELM����/ѵ��
[IW,B,LW,TF,TYPE] = elmtrain(Fact_train,Label_train,NetNo,FunName,1);

%% ELM�������
T_sim_1 = elmpredict(Fact_train,IW,B,LW,TF,TYPE);
T_sim_2 = elmpredict(Fact_test,IW,B,LW,TF,TYPE);

toc

subplot(121)
plot(1:size(Label_train,2),Label_train,'-b*')
hold on
plot(1:size(Label_train,2),T_sim_1,'-ro')
legend('ѵ����ԭ��ǩ','ѵ����Ԥ���ǩ')

subplot(122)
plot(1:size(Label_test,2),Label_test,'-b*')
hold on
plot(1:size(Label_test,2),T_sim_2,'-ro')
legend('���Լ�ԭ��ǩ','���Լ�Ԥ���ǩ')

%% ����Ա�
result_1 = [Label_train' T_sim_1'];
result_2 = [Label_test' T_sim_2'];
% ѵ������ȷ��
k1 = length(find(Label_train == T_sim_1));
n1 = length(Label_train);
Accuracy_1 = k1 / n1 * 100;
disp(['ѵ������ȷ��Accuracy = ' num2str(Accuracy_1) '%(' num2str(k1) '/' num2str(n1) ')'])
% ���Լ���ȷ��
k2 = length(find(Label_test == T_sim_2));
n2 = length(Label_test);
Accuracy_2 = k2 / n2 * 100;
disp(['���Լ���ȷ��Accuracy = ' num2str(Accuracy_2) '%(' num2str(k2) '/' num2str(n2) ')'])

%% ��ʾ
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
% disp(['����������' num2str(569)...
%       '  ���ԣ�' num2str(total_B)...
%       '  ���ԣ�' num2str(total_M)]);
% disp(['ѵ��������������' num2str(500)...
%       '  ���ԣ�' num2str(count_B)...
%       '  ���ԣ�' num2str(count_M)]);
% disp(['���Լ�����������' num2str(69)...
%       '  ���ԣ�' num2str(number_B)...
%       '  ���ԣ�' num2str(number_M)]);
% disp(['������������ȷ�' num2str(number_B_sim)...
%       '  ���' num2str(number_B - number_B_sim)...
%       '  ȷ����p1=' num2str(number_B_sim/number_B*100) '%']);
% disp(['������������ȷ�' num2str(number_M_sim)...
%       '  ���' num2str(number_M - number_M_sim)...
%       '  ȷ����p2=' num2str(number_M_sim/number_M*100) '%']);  



R = [];
for i = 50:50:500
    %% ELM����/ѵ��
    [IW,B,LW,TF,TYPE] = elmtrain(Fact_train,Label_train,i,'sig',1);
    
    %% ELM�������
    T_sim_1 = elmpredict(Fact_train,IW,B,LW,TF,TYPE);
    T_sim_2 = elmpredict(Fact_test,IW,B,LW,TF,TYPE);
    
    %% ����Ա�
    result_1 = [Label_train' T_sim_1'];
    result_2 = [Label_test' T_sim_2'];
    % ѵ������ȷ��
    k1 = length(find(Label_train == T_sim_1));
    n1 = length(Label_train);
    Accuracy_1 = k1 / n1 * 100;
%     disp(['ѵ������ȷ��Accuracy = ' num2str(Accuracy_1) '%(' num2str(k1) '/' num2str(n1) ')'])
    % ���Լ���ȷ��
    k2 = length(find(Label_test == T_sim_2));
    n2 = length(Label_test);
    Accuracy_2 = k2 / n2 * 100;
%     disp(['���Լ���ȷ��Accuracy = ' num2str(Accuracy_2) '%(' num2str(k2) '/' num2str(n2) ')'])
    R = [R;Accuracy_1 Accuracy_2];
end
  
figure
plot(50:50:500,R(:,2),'b:o')
xlabel('��������Ԫ����')
ylabel('���Լ�Ԥ����ȷ�ʣ�%��')
title('��������Ԫ������ELM���ܵ�Ӱ��')


