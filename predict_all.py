import os
import json

import torch
from PIL import Image
from torchvision import transforms
from model import resnet101, resnet50, resnet34
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, transform=None, prefix='E:/archive/crop'):
        self.dataframe = dataframe
        self.transform = transform
        self.prefix = prefix

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = os.path.join(self.prefix, self.dataframe.iloc[idx, 0])  # 添加前缀路径
        label = self.dataframe.iloc[idx, 1]

        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label

def main(image_path, model, json_path, output_file):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 数据转换
    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载测试集
    df = pd.read_table(image_path, sep='\t')
    # 划分训练集和测试集
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    test_dataset = CustomDataset(test_df, transform=data_transform)

    # 创建文件写入对象
    file_write = open(output_file, 'w', encoding='utf-8')

    # 预测并写入结果
    for i in tqdm(range(len(test_dataset))):
        img, label = test_dataset[i]
        img = torch.unsqueeze(img, dim=0).to(device)

        with torch.no_grad():
            output = torch.squeeze(model(img)).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).item()
            score = predict[predict_cla].item()

            file_name = os.path.basename(test_df.iloc[i, 0])
            predict_label = label_map_inverse[str(predict_cla)]

            file_write.write(f"{file_name}\t{str(label)}\t{predict_cla}\t{predict_label}\n")

    file_write.close()

if __name__ == '__main__':
    # 参数设置
    image_path = 'E:/archive/code/image_path_0401.txt'
    weights_path = 'E:/archive/code/Resnet/model_weight/ResNet-34-0108.pth'
    json_path = 'E:/archive/code/Resnet/class_indices.json'
    output_file = 'E:/archive/code/Resnet/output/ResNet_test_data_res_0402.txt'

    # 创建模型
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet34(num_classes=47).to(device)

    # 加载模型权重
    assert os.path.exists(weights_path), f"File '{weights_path}' dose not exist."
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 读取标签映射
    with open(json_path, 'r') as f:
        label_map_inverse = json.load(f)

    # 划分训练集和测试集
    df = pd.read_table(image_path, sep='\t', header=None)
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)

    # 将测试集路径写入文件
    test_image_path = 'E:/archive/code/Resnet/test_image_path.txt'
    test_df.to_csv(test_image_path, sep='\t', index=False)

    # 进行预测
    main(test_image_path, model, json_path, output_file)

    # 计算准确率和混淆矩阵
    df = pd.read_table(output_file, sep='\t', header=None)
    df.columns = ['file_name', 'label', 'score', 'predict']

    label = df['label']
    predict = df['score']

    # print('The classification report is:\n')
    report_dict = classification_report(label,predict)
    # print(report_dict)


    label = df['label']
    predict = df['score']

    # 计算准确率
    accuracy = accuracy_score(label, predict)

    # 计算加权的精确率、召回率、F1 分数
    precision = precision_score(label, predict, average='weighted')
    recall = recall_score(label, predict, average='weighted')
    f1 = f1_score(label, predict, average='weighted')

    print(f"Accuracy: {accuracy}")
    print(f"Weighted Precision: {precision}")
    print(f"Weighted Recall: {recall}")
    print(f"Weighted F1 Score: {f1}")

    # confusion_data = confusion_matrix(label, predict)
    # print(confusion_data)
    #
    # fig, ax = plot_confusion_matrix(confusion_data)
    # plt.savefig('./confusion_data_resnet34.png', dpi=300)
    # plt.show()
