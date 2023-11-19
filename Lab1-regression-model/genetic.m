% 1. 读取训练数据集和测试数据集
trainData = csvread('abalone_train.csv'); % 从train_data.csv文件中读取训练数据
testData = csvread('abalone_test.csv'); % 从test_data.csv文件中读取测试数据

% 2. 提取标签值和参数值
trainLabels = trainData(:, 1); % 第一列是标签值
trainFeatures = trainData(:, 2:end); % 后面的列是参数值

testLabels = testData(:, 1); % 第一列是标签值
testFeatures = testData(:, 2:end); % 后面的列是参数值

% 3. 定义线性回归模型函数
linearRegressionModel = @(x, data) x(1) + sum(x(2:end) .* data, 2);

% 4. 定义遗传算法参数和优化函数
options = optimoptions('ga', 'MaxGenerations', 1000, 'PopulationSize', 100);
fitnessFunction = @(x) sum((linearRegressionModel(x, trainFeatures) - trainLabels).^2);

% 5. 使用遗传算法进行优化
initialGuess = rand(size(trainFeatures, 2) + 1, 1); % 随机初始化参数
bestParams = ga(fitnessFunction, length(initialGuess), [], [], [], [], [], [], [], options);

% 6. 使用最优参数进行预测
predictedLabels = linearRegressionModel(bestParams, testFeatures);

% 7. 计算模型的准确度
mse = mean((predictedLabels - testLabels).^2); % 均方误差
accuracy = 1 - mse / var(testLabels); % 准确度

% 8. 输出结果
fprintf('测试准确度: %.2f%%\n', accuracy * 100);
fprintf('最佳参数结果:\n');
disp(bestParams);

% 9. 绘制结果图和曲线拟合
figure;
scatter(testLabels, predictedLabels);
hold on;
plot(testLabels, testLabels, 'r'); % 红线表示理论值与预测值一致的情况
xlabel('实际值');
ylabel('预测值');
title('线性回归模型拟合结果');
legend('预测值', '一致线');

% 10. 输出误差值
fprintf('均方误差(MSE): %.4f\n', mse);