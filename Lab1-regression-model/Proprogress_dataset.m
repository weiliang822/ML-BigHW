clear;
clc;

data_name = 'abalone';% 数据集名

n_entradas= 8; % 属性数
n_clases= 3; % 分类数
n_fich= 1; % 文件数，含有训练和测试集
fich{1}= 'C:\Users\86189\Desktop\abalone\abalone.data';% 训练数据路径
n_patrons(1)= 2954; % 训练集数据量
n_patrons(2)= 1253;   % 测试数据量
n_patrons(3)= 4177;  % 数据集总数据量

n_max= max(n_patrons);
x = zeros(n_fich, n_max, n_entradas); % 属性数据
cl= zeros(n_fich, n_max);             % 标签


for i_fich = 1:n_fich
    f=fopen(fich{i_fich}, 'r');
    if -1==f
        error('打开数据文件出错 %s\n', fich{i_fich});
    end

    for i=1:n_patrons(3) % 将总数据集数据分为测试数据集和训练数据集
        fprintf('%5.1f%%\r', 100*i/n_patrons(3)); % 显示进度
        t = fscanf(f, '%c', 1); % 读取一个字符数据
        switch t % 将对应字符替换为数字
               case 'M'
                   x(1,i,1)=-1;
               case 'F'
                   x(1,i,1)=0;
               case 'I'
                   x(1,i,1)=1;
        end

        for j=2:n_entradas
            fscanf(f,'%c',1); % 中间有分隔符，后移1个位置
            x(1,i,j) = fscanf(f,'%f', 1);% 依次读取这一行所有属性
        end

         fscanf(f,'%c',1); 
         cl(1,i) = fscanf(f,'%i', 1); % 读取最后的标记值
         fscanf(f,'%c',1);

     end
     fclose(f);
end


%% 处理完成，保存文件
fprintf('现在保存数据文件...\n')
dir_path=['./预处理完成/',data_name];
if exist('./预处理完成/','dir')==0   %该文件夹不存在，则直接创建
    mkdir('./预处理完成/');
end
data_train =  squeeze(x(1,1:n_patrons(1),:)); % 数据
label_train = squeeze(cl(1,1:n_patrons(1)))';% 标签
dataSet_train = [label_train, data_train];
saveData(dataSet_train,[dir_path,'_train']); % 保存文件至文件夹

data_test =  squeeze(x(1,n_patrons(1)+1:n_patrons(3),:)); % 数据
label_test = squeeze(cl(1,n_patrons(1)+1:n_patrons(3)))';% 标签
dataSet_test = [label_test,data_test];
saveData(dataSet_test,[dir_path,'_test']);

fprintf('预处理完成\n')


%% 子函数，用于保存txt/data/mat三种类型文件
function saveData(DataSet,fileName)
% DataSet:整理好的数据集
% fileName：数据集的名字

%% Data为整理好的数据集矩阵
mat_name = [fileName,'.mat'];
save(mat_name, 'DataSet')  % 保存.mat文件
data_name = [fileName,'.data'];
save(data_name,'DataSet','-ASCII'); % 保存data文件

% 保存txt文件
txt_name = [fileName,'.txt'];
f=fopen(txt_name,'w');
[m,n]=size(DataSet);
for i=1:m
    for j=1:n
        if j==n
            if i~=m
                fprintf(f,'%g \n',DataSet(i,j));
            else
                fprintf(f,'%g',DataSet(i,j));
            end
        else
            fprintf(f,'%g,',DataSet(i,j));
        end
    end
end
fclose(f);

end