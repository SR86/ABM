"""
数据投毒攻击实验 (Data Poisoning Attack Experiment)
===================================================
本实验演示了标签翻转攻击(Label Flipping Attack)对机器学习模型的影响。
通过污染一定比例的训练数据标签，观察模型性能的变化。

实验环境: Python 3.x, scikit-learn, numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DataPoisoningExperiment:
    """数据投毒攻击实验类"""
    
    def __init__(self, dataset_name='iris'):
        """
        初始化实验
        
        Args:
            dataset_name: 数据集名称 ('iris' 或 'digits')
        """
        self.dataset_name = dataset_name
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_original = None
        self.num_classes = None
        
        self._load_data()
    
    def _load_data(self):
        """加载数据集"""
        if self.dataset_name == 'iris':
            data = load_iris()
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                data.data, data.target, test_size=0.3, random_state=42, stratify=data.target
            )
        elif self.dataset_name == 'digits':
            data = load_digits()
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                data.data, data.target, test_size=0.3, random_state=42, stratify=data.target
            )
        else:
            raise ValueError(f"不支持的数据集: {self.dataset_name}")
        
        self.y_train_original = self.y_train.copy()
        self.num_classes = len(np.unique(self.y_train))
        print(f"数据集: {self.dataset_name}")
        print(f"训练样本数: {len(self.y_train)}, 测试样本数: {len(self.y_test)}")
        print(f"类别数: {self.num_classes}")
        print("-" * 50)
    
    def poison_labels(self, poison_ratio=0.1, target_class=None, source_class=None):
        """
        执行标签翻转攻击
        
        Args:
            poison_ratio: 投毒比例 (0.0 - 1.0)
            target_class: 目标类别（攻击者希望模型将样本分类到的类别）
            source_class: 源类别（被篡改的样本原本属于的类别）
                         如果为 None，则随机选择非目标类别的样本进行投毒
        
        Returns:
            poisoned_indices: 被投毒的样本索引
            num_poisoned: 投毒样本数量
        """
        y_poisoned = self.y_train_original.copy()
        
        if target_class is None:
            target_class = 0  # 默认目标类别为 0
        
        # 找出所有非目标类别的样本
        non_target_indices = np.where(self.y_train_original != target_class)[0]
        
        # 计算要投毒的样本数量
        num_poisoned = int(len(non_target_indices) * poison_ratio)
        
        # 随机选择要投毒的样本
        np.random.seed(42)
        poisoned_indices = np.random.choice(non_target_indices, num_poisoned, replace=False)
        
        # 如果指定了源类别，则只投毒属于该类别的样本
        if source_class is not None:
            source_indices = np.where(self.y_train_original == source_class)[0]
            poisoned_indices = np.intersect1d(poisoned_indices, source_indices)
            num_poisoned = len(poisoned_indices)
        
        # 执行标签翻转
        y_poisoned[poisoned_indices] = target_class
        
        print(f"投毒比例: {poison_ratio*100:.1f}%")
        print(f"投毒样本数: {num_poisoned}")
        print(f"目标类别: {target_class}")
        if source_class is not None:
            print(f"源类别: {source_class}")
        
        return y_poisoned, poisoned_indices
    
    def train_and_evaluate(self, model, X_train, y_train, X_test, y_test, model_name="Model"):
        """
        训练模型并评估性能
        
        Args:
            model: 分类器模型
            X_train, y_train: 训练数据
            X_test, y_test: 测试数据
            model_name: 模型名称
            
        Returns:
            accuracy: 测试准确率
            y_pred: 预测结果
        """
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy, y_pred
    
    def run_experiment(self, poison_ratios=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3]):
        """
        运行完整实验，测试不同投毒比例下的模型性能
        
        Args:
            poison_ratios: 投毒比例列表
        """
        models = {
            'SVM': SVC(kernel='rbf', random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
        }
        
        results = {model_name: [] for model_name in models.keys()}
        
        print("\n开始实验...")
        print("=" * 70)
        
        for ratio in poison_ratios:
            print(f"\n投毒比例: {ratio*100:.1f}%")
            print("-" * 70)
            
            # 生成投毒数据
            y_poisoned, _ = self.poison_labels(poison_ratio=ratio, target_class=0)
            
            # 测试每个模型
            for model_name, model_template in models.items():
                # 克隆模型
                model = model_template.__class__(**model_template.get_params())
                
                # 训练并评估
                acc, _ = self.train_and_evaluate(
                    model, self.X_train, y_poisoned, self.X_test, self.y_test, model_name
                )
                
                results[model_name].append(acc)
                print(f"  {model_name:20s}: 准确率 = {acc:.4f}")
        
        return results, poison_ratios
    
    def plot_results(self, results, poison_ratios):
        """
        可视化实验结果
        
        Args:
            results: 实验结果字典
            poison_ratios: 投毒比例列表
        """
        plt.figure(figsize=(12, 8))
        
        colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']
        markers = ['o', 's', '^', 'D']
        
        for (model_name, accuracies), color, marker in zip(results.items(), colors, markers):
            plt.plot(
                [r * 100 for r in poison_ratios], 
                accuracies, 
                marker=marker, 
                linewidth=2, 
                markersize=8,
                label=model_name,
                color=color
            )
        
        plt.xlabel('投毒比例 (%)', fontsize=12, fontweight='bold')
        plt.ylabel('测试准确率', fontsize=12, fontweight='bold')
        plt.title(f'数据投毒攻击对模型性能的影响 ({self.dataset_name.upper()} 数据集)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.legend(fontsize=10, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks([r * 100 for r in poison_ratios])
        plt.ylim(0, 1.05)
        plt.tight_layout()
        
        # 保存图像
        save_path = f'c:\\项目文档\\poisoning_results_{self.dataset_name}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n结果图已保存至: {save_path}")
        
        plt.show()
    
    def detailed_analysis(self, poison_ratio=0.2):
        """
        对特定投毒比例进行详细分析
        
        Args:
            poison_ratio: 投毒比例
        """
        print(f"\n{'='*70}")
        print(f"详细分析 - 投毒比例: {poison_ratio*100:.1f}%")
        print(f"{'='*70}")
        
        # 生成投毒数据
        y_poisoned, poisoned_indices = self.poison_labels(
            poison_ratio=poison_ratio, target_class=0, source_class=1
        )
        
        # 使用 SVM 进行详细分析
        model = SVC(kernel='rbf', random_state=42)
        
        # 清洁数据训练
        model_clean = SVC(kernel='rbf', random_state=42)
        acc_clean, y_pred_clean = self.train_and_evaluate(
            model_clean, self.X_train, self.y_train_original, 
            self.X_test, self.y_test, "Clean Model"
        )
        
        # 投毒数据训练
        acc_poisoned, y_pred_poisoned = self.train_and_evaluate(
            model, self.X_train, y_poisoned, 
            self.X_test, self.y_test, "Poisoned Model"
        )
        
        print(f"\n清洁模型准确率: {acc_clean:.4f}")
        print(f"投毒模型准确率: {acc_poisoned:.4f}")
        print(f"性能下降: {(acc_clean - acc_poisoned)*100:.2f}%")
        
        print("\n清洁模型分类报告:")
        print(classification_report(self.y_test, y_pred_clean, digits=4))
        
        print("\n投毒模型分类报告:")
        print(classification_report(self.y_test, y_pred_poisoned, digits=4))
        
        # 分析被投毒样本的影响
        print(f"\n被投毒样本数: {len(poisoned_indices)}")
        print(f"被投毒样本原始标签分布:")
        unique, counts = np.unique(self.y_train_original[poisoned_indices], return_counts=True)
        for label, count in zip(unique, counts):
            print(f"  类别 {label}: {count} 个样本")


def main():
    """主函数"""
    print("=" * 70)
    print("数据投毒攻击实验 (Data Poisoning Attack Experiment)")
    print("=" * 70)
    print()
    
    # 实验 1: Iris 数据集
    print("\n[实验 1] Iris 数据集")
    print("=" * 70)
    exp_iris = DataPoisoningExperiment(dataset_name='iris')
    results_iris, ratios_iris = exp_iris.run_experiment(
        poison_ratios=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    )
    exp_iris.plot_results(results_iris, ratios_iris)
    exp_iris.detailed_analysis(poison_ratio=0.2)
    
    # 实验 2: Digits 数据集
    print("\n\n[实验 2] Digits 数据集")
    print("=" * 70)
    exp_digits = DataPoisoningExperiment(dataset_name='digits')
    results_digits, ratios_digits = exp_digits.run_experiment(
        poison_ratios=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    )
    exp_digits.plot_results(results_digits, ratios_digits)
    exp_digits.detailed_analysis(poison_ratio=0.2)
    
    print("\n" + "=" * 70)
    print("实验完成！")
    print("=" * 70)
    print("\n实验总结:")
    print("1. 随着投毒比例增加，模型性能逐渐下降")
    print("2. 不同模型对投毒攻击的鲁棒性不同")
    print("3. 标签翻转攻击可以有效降低分类器的准确率")
    print("4. 多类别任务中，投毒攻击的影响更加显著")


if __name__ == "__main__":
    main()
