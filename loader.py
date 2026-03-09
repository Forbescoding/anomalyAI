import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
class DataLoaderAnalyzer:
    def __init__(self):
        self.data=None
        self.X=None
        self.y=None
        self.feature_names=None
        self.scaler=StandardScaler()
        self.report={}
    def load_data(self, filepath):
        print("\n" + "="*70)
        print(f"ЗАГРУЗКА ДАННЫХ: {os.path.basename(filepath)}")
        print("="*70)
        if not os.path.exists(filepath):
            print(f"Файл не найден: {filepath}")
            return False
        ext=os.path.splitext(filepath)[1].lower()
        try:
            if ext=='.csv':
                self.data=pd.read_csv(filepath)
                print("CSV файл загружен")
            elif ext in ['.xlsx', '.xls']:
                self.data=pd.read_excel(filepath)
                print("Excel файл загружен")   
            elif ext=='.npy':
                array=np.load(filepath)
                self.data=pd.DataFrame(array)
                print("NumPy файл загружен") 
            elif ext=='.txt' or ext=='.data':
                try:
                    self.data=pd.read_csv(filepath, sep=',')
                except:
                    try:
                        self.data=pd.read_csv(filepath, sep='\s+')
                    except:
                        self.data=pd.read_csv(filepath, sep='\t')
                print("Текстовый файл загружен")
            elif ext=='.json':
                self.data=pd.read_json(filepath)
                print("JSON файл загружен")
            elif ext=='.pkl' or '.pkl.gz' in filepath:
                self.data=pd.read_pickle(filepath)
                print("Pickle файл загружен")
            else:
                print(f"Неподдерживаемый формат: {ext}")
                return False
            print(f"  Размер данных: {self.data.shape}")
            print(f"  Колонки: {list(self.data.columns)[:10]}...")
            return True
        except Exception as e:
            print(f"Ошибка загрузки: {e}")
            return False
    def detect_label_column(self):
        if self.data is None:
            return None
        label_names=['label', 'class', 'target', 'attack', 'type', 'y', 
                       'Label', 'CLASS', 'Attack', 'attack_type', 'outcome',
                       'класс', 'метка', 'тип']
        for col in self.data.columns:
            if col.lower() in [l.lower() for l in label_names]:
                print(f"Найдена колонка с метками: '{col}'")
                return col
        for col in self.data.columns:
            unique_vals=self.data[col].nunique()
            if 2<=unique_vals<=10 and len(self.data)>100:
                print(f"Возможная колонка с метками: '{col}' ({unique_vals} уникальных значений)")
                return col
        print("Не удалось автоматически определить колонку с метками")
        return None
    def prepare_data(self, label_col=None, test_size=0.3):
        if self.data is None:
            print("Сначала загрузите данные!")
            return False
        print("\n" + "=" * 70)
        print("ПОДГОТОВКА ДАННЫХ")
        print("=" * 70)
        if label_col is None:
            label_col=self.detect_label_column()
            if label_col is None:
                print("Не удалось определить колонку с метками")
                print("Укажите её вручную: prepare_data(label_col='имя_колонки')")
                return False
        if label_col not in self.data.columns:
            print(f"Колонка '{label_col}' не найдена")
            print(f"Доступные колонки: {list(self.data.columns)}")
            return False
        self.y=self.data[label_col].values
        unique_labels=np.unique(self.y)
        print(f"\nУникальные метки: {unique_labels}")
        if len(unique_labels)==2:
            label_mapping={unique_labels[0]: 0, unique_labels[1]: 1}
            self.y_binary=np.array([label_mapping[x] for x in self.y])
            print(f"Бинарные метки: {unique_labels[0]} -> 0, {unique_labels[1]} -> 1")
        else:
            normal_class=unique_labels[0]
            self.y_binary=np.array([0 if x==normal_class else 1 for x in self.y])
            print(f"Многоклассовые метки преобразованы в бинарные")
            print(f"Норма: {normal_class}, Аномалии: все остальные")
        self.feature_names=[col for col in self.data.columns if col!=label_col]
        self.X=self.data[self.feature_names].copy()
        
        categorical_cols=self.X.select_dtypes(include=['object']).columns
        if len(categorical_cols)>0:
            print(f"Найдены категориальные признаки: {list(categorical_cols)}")
            for col in categorical_cols:
                le=LabelEncoder()
                self.X[col]=le.fit_transform(self.X[col].astype(str))
            print("Категориальные признаки закодированы")  
        if self.X.isnull().sum().sum()>0:
            print(f"\nНайдены пропущенные значения, заполняется...")
            self.X=self.X.fillna(self.X.mean())
        self.X_scaled=self.scaler.fit_transform(self.X)
        # разделение на треньку и тест
        self.X_train, self.X_test, self.y_train, self.y_test=train_test_split(
            self.X_scaled, self.y_binary, test_size=test_size, 
            random_state=42, stratify=self.y_binary
        )
        print(f"\nРЕЗУЛЬТАТ ПОДГОТОВКИ:")
        print(f" Признаков: {self.X.shape[1]}")
        print(f" Всего записей: {len(self.X)}")
        print(f" Нормальный трафик: {(self.y_binary == 0).sum()} ({(self.y_binary == 0).mean()*100:.1f}%)")
        print(f" Аномальный трафик: {(self.y_binary == 1).sum()} ({(self.y_binary == 1).mean()*100:.1f}%)")
        print(f"\n Обучающая выборка: {len(self.X_train)} записей")
        print(f" Тестовая выборка: {len(self.X_test)} записей")
        return True
    def analyze_data(self):
        if self.X is None:
            print("Сначала подготовьте данные!")
            return
        print("\n" + "="*70)
        print("АНАЛИЗ ДАННЫХ")
        print("="*70)
        print("\n1. РАСПРЕДЕЛЕНИЕ КЛАССОВ:")
        unique, counts = np.unique(self.y_binary, return_counts=True)
        for label, count in zip(unique, counts):
            class_name = "Нормальный" if label==0 else "Аномальный"
            print(f" {class_name}: {count} записей ({count/len(self.y_binary)*100:.1f}%)")
        imbalance = abs(counts[0]-counts[1])/len(self.y_binary)
        print(f"\n   Дисбаланс классов: {imbalance:.3f}")
        if imbalance<0.2:
            print("Данные сбалансированы")
        else:
            print("Сильный дисбаланс классов")
        print("\n2. СТАТИСТИКА ПРИЗНАКОВ:")
        stats = pd.DataFrame({
            'Признак': self.feature_names[:10],
            'Среднее': np.mean(self.X, axis=0)[:10].round(3),
            'Стд отклонение': np.std(self.X, axis=0)[:10].round(3),
            'Мин': np.min(self.X, axis=0)[:10].round(3),
            'Макс': np.max(self.X, axis=0)[:10].round(3)
        })
        print(stats.to_string(index=False))
        print(f"   ... и еще {len(self.feature_names)-10} признаков")
        print("\n3. КОРРЕЛЯЦИЯ С ЦЕЛЕВОЙ ПЕРЕМЕННОЙ (топ 10):")
        correlations=[]
        for i, col in enumerate(self.feature_names):
            corr=np.corrcoef(self.X[:, i], self.y_binary)[0,1]
            correlations.append((col, abs(corr), corr))
        correlations.sort(key=lambda x: x[1], reverse=True)
        for col, abs_corr, corr in correlations[:10]:
            direction="положительная" if corr>0 else "отрицательная"
            print(f" {col}: {corr:.3f} ({direction})")
    def plot_results(self):
        #vizualka
        if self.X is None:
            print("Сначала подготовьте данные!")
            return
        print("\nСоздание графиков...")
        fig=plt.figure(figsize=(20, 15))
        ax1=plt.subplot(3, 3, 1)
        labels=['Нормальный', 'Аномальный']
        counts=[(self.y_binary==0).sum(),(self.y_binary==1).sum()]
        colors=['green', 'red']
        bars=ax1.bar(labels, counts, color=colors)
        ax1.set_title('Распределение классов', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Количество записей')
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                    f'{count}\n({count/len(self.y_binary)*100:.1f}%)',
                    ha='center', va='bottom', fontsize=11)
        #krugovaya, dopisat
        ax2=plt.subplot(3, 3, 2)
        ax2.pie(counts, labels=labels, autopct='%1.1f%%', 
                colors=colors, startangle=90)
        ax2.set_title('Соотношение классов', fontsize=14, fontweight='bold')
        
        ax3=plt.subplot(3, 3, 3)
        n_features=min(15, self.X.shape[1])
        corr_matrix=np.corrcoef(self.X[:, :n_features].T)
        im=ax3.imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        ax3.set_title('Корреляция признаков', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Признаки')
        ax3.set_ylabel('Признаки')
        plt.colorbar(im, ax=ax3)
        
        ax4=plt.subplot(3, 3, 4)
        for i in range(min(3, self.X.shape[1])):
            ax4.hist(self.X[self.y_binary == 0, i], alpha=0.5, 
                    label=f'{self.feature_names[i]} (норма)', bins=30)
            ax4.hist(self.X[self.y_binary == 1, i], alpha=0.5, 
                    label=f'{self.feature_names[i]} (аном)', bins=30)
        ax4.set_title('Распределение признаков', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Значение')
        ax4.set_ylabel('Частота')
        ax4.legend(fontsize=8)
        
        ax5=plt.subplot(3, 3, 5)
        data_to_plot=[self.X[self.y_binary==0, i] for i in range(min(5, self.X.shape[1]))]
        data_to_plot+=[self.X[self.y_binary==1, i] for i in range(min(5, self.X.shape[1]))]
        labels_plot=[f'{self.feature_names[i]}\n(норма)' for i in range(min(5, self.X.shape[1]))]
        labels_plot+=[f'{self.feature_names[i]}\n(аном)' for i in range(min(5, self.X.shape[1]))]
        ax5.boxplot(data_to_plot, labels=labels_plot)
        ax5.set_title('Box-plot признаков', fontsize=14, fontweight='bold')
        ax5.tick_params(axis='x', rotation=45, labelsize=8)
        
        ax6=plt.subplot(3, 3, 6)
        try:
            from sklearn.decomposition import PCA
            pca=PCA(n_components=2)
            X_pca=pca.fit_transform(self.X_scaled[:1000])
            y_sample=self.y_binary[:1000]
            ax6.scatter(X_pca[y_sample==0, 0], X_pca[y_sample==0, 1], 
                       c='green', alpha=0.5, label='Норма', s=10)
            ax6.scatter(X_pca[y_sample==1, 0], X_pca[y_sample==1, 1], 
                       c='red', alpha=0.5, label='Аномалия', s=10)
            ax6.set_title('PCA визуализация', fontsize=14, fontweight='bold')
            ax6.set_xlabel('PC1')
            ax6.set_ylabel('PC2')
            ax6.legend()
        except:
            ax6.text(0.5, 0.5, 'PCA недоступна', ha='center', va='center')
        
        ax7 = plt.subplot(3, 3, 7)
        ax7.axis('off')
        info_text = f"""
        ИНФОРМАЦИЯ О ДАННЫХ:
        
        Всего записей: {len(self.X)}
        Признаков: {self.X.shape[1]}
        
        Норма: {counts[0]} ({counts[0]/len(self.y_binary)*100:.1f}%)
        Аномалии: {counts[1]} ({counts[1]/len(self.y_binary)*100:.1f}%)
        
        Типы признаков:
        Числовые: {self.X.select_dtypes(include=[np.number]).shape[1]}
        Категориальные: {self.X.select_dtypes(include=['object']).shape[1]}
        
        Пропуски: {self.X.isnull().sum().sum()}
        """
        ax7.text(0.1, 0.9, info_text, fontsize=12, va='top', family='monospace')
        ax8=plt.subplot(3, 3, 8)
        top_corrs=correlations[:10]
        features=[c[0][:15]+'...' if len(c[0])>15 else c[0] for c in top_corrs]
        corr_values=[c[2] for c in top_corrs]
        colors_corr=['red' if x < 0 else 'green' for x in corr_values]
        ax8.barh(range(len(features)), corr_values, color=colors_corr)
        ax8.set_yticks(range(len(features)))
        ax8.set_yticklabels(features, fontsize=8)
        ax8.set_xlabel('Корреляция с целевой переменной')
        ax8.set_title('Топ-10 корреляций', fontsize=14, fontweight='bold')
        ax8.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        quality_score=0
        quality_score+=min(1.0, counts[0] / counts[1])*20 if counts[1]>0 else 20 
        quality_score+=min(100, self.X.shape[1])/100*20 
        quality_score+=min(10000, len(self.X))/10000*30  
        quality_score+=(1-self.X.isnull().sum().sum()/self.X.size)*30
        quality_score=min(100, quality_score)
        ax9.text(0.5, 0.9, 'ОЦЕНКА КАЧЕСТВА ДАННЫХ', 
                fontsize=14, fontweight='bold', ha='center')
        sizes=[quality_score, 100-quality_score]
        colors_quality=['green', 'lightgray']
        ax9.pie(sizes, colors=colors_quality, startangle=90,
               wedgeprops={'width': 0.3})
        ax9.text(0.5, 0.5, f'{quality_score:.0f}%', 
                fontsize=24, fontweight='bold', ha='center')
        
        if quality_score>=80:
            ax9.text(0.5, 0.2, 'Отличные данные!', 
                    fontsize=12, color='green', ha='center')
        elif quality_score>=60:
            ax9.text(0.5, 0.2, 'Хорошие данные', 
                    fontsize=12, color='orange', ha='center')
        else:
            ax9.text(0.5, 0.2, 'Требуют улучшения', 
                    fontsize=12, color='red', ha='center')
        plt.suptitle(f'АНАЛИЗ ДАННЫХ: {os.path.basename(self.current_file)}', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        filename = f'data_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"График сохранен: {filename}")
        plt.show()
    def save_processed_data(self, prefix='processed'):
        if self.X_train is None:
            print("Нет обработанных данных для сохранения")
            return
        timestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        np.save(f'{prefix}_X_train_{timestamp}.npy', self.X_train)
        np.save(f'{prefix}_X_test_{timestamp}.npy', self.X_test)
        np.save(f'{prefix}_y_train_{timestamp}.npy', self.y_train)
        np.save(f'{prefix}_y_test_{timestamp}.npy', self.y_test)
        import joblib
        joblib.dump(self.feature_names, f'{prefix}_features_{timestamp}.pkl')
        joblib.dump(self.scaler, f'{prefix}_scaler_{timestamp}.pkl')
        print(f"\n Данные сохранены с префиксом: {prefix}_{timestamp}")
        print(f"  - X_train: {self.X_train.shape}")
        print(f"  - X_test: {self.X_test.shape}")
        print(f"  - features: {len(self.feature_names)} признаков")
    def generate_report(self):
        if self.X is None:
            print("Нет данных для отчета")
            return
        print("\n" + "="*70)
        print("ИТОГОВЫЙ ОТЧЕТ")
        print("="*70)
        
        report = f"""
        ОТЧЕТ ПО АНАЛИЗУ ДАННЫХ
        =======================
        
        1. ОСНОВНАЯ ИНФОРМАЦИЯ
        ----------------------
        Файл: {getattr(self, 'current_file', 'неизвестно')}
        Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Всего записей: {len(self.X)}
        Всего признаков: {self.X.shape[1]}
        
        2. РАСПРЕДЕЛЕНИЕ КЛАССОВ
        ------------------------
        Нормальный трафик: {(self.y_binary == 0).sum()} ({(self.y_binary == 0).mean()*100:.1f}%)
        Аномальный трафик: {(self.y_binary == 1).sum()} ({(self.y_binary == 1).mean()*100:.1f}%)
        
        3. ХАРАКТЕРИСТИКИ ПРИЗНАКОВ
        ---------------------------
        Типы данных:
        - Числовые: {self.X.select_dtypes(include=[np.number]).shape[1]}
        - Категориальные: {self.X.select_dtypes(include=['object']).shape[1]}
        
        Пропущенные значения: {self.X.isnull().sum().sum()}
        
        4. ТОП-5 ПРИЗНАКОВ ПО КОРРЕЛЯЦИИ
        --------------------------------
        """
        correlations=[]
        for i, col in enumerate(self.feature_names):
            corr=np.corrcoef(self.X_scaled[:, i], self.y_binary)[0, 1]
            correlations.append((col, abs(corr), corr))
        correlations.sort(key=lambda x: x[1], reverse=True)
        for i, (col, abs_corr, corr) in enumerate(correlations[:5]):
            report+=f" {i+1}. {col}: {corr:.3f}\n"
        report+="""
        5. РЕКОМЕНДАЦИИ
        ---------------
        """        
        if (self.y_binary == 0).sum()<(self.y_binary == 1).sum() * 0.5:
            report+="Сильный дисбаланс классов - используйте взвешенные метки\n"
        if self.X.shape[1]>100:
            report+="Много признаков - рекомендуется отбор признаков\n"
        if self.X.isnull().sum().sum()>0:
            report+="Есть пропуски - проверьте качество данных\n"
        print(report)
        filename=f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Отчет сохранен: {filename}")
def interactive_mode():
    loader=DataLoaderAnalyzer()
    print("\n" + "="*70)
    print("УНИВЕРСАЛЬНЫЙ ЗАГРУЗЧИК ДАННЫХ")
    print("="*70)
    print("\nПоддерживаемые форматы: CSV, Excel, NumPy, TXT, JSON, Pickle")
    while True:
        print("\n"+"-"*50)
        print("МЕНЮ:")
        print("1. Загрузить файл")
        print("2. Показать информацию о данных")
        print("3. Визуализировать данные")
        print("4. Сгенерировать отчет")
        print("5. Сохранить обработанные данные")
        print("6. Выход")
        choice = input("\nВыберите действие (1-6): ").strip()
        if choice=='1':
            filepath=input("Введите путь к файлу: ").strip()
            if loader.load_data(filepath):
                loader.current_file = filepath
                if loader.prepare_data():
                    loader.analyze_data()
        elif choice=='2':
            loader.analyze_data()
        elif choice=='3':
            loader.plot_results()
        elif choice=='4':
            loader.generate_report()
        elif choice=='5':
            prefix = input("Введите префикс для сохранения (по умолчанию 'processed'): ").strip()
            if not prefix:
                prefix='processed'
            loader.save_processed_data(prefix)
        elif choice=='6':
            print("\nЗавершение выполнения")
            break
        else:
            print("Неверный выбор, попробуйте снова")
def quick_analyze(filepath):
    loader=DataLoaderAnalyzer()
    if loader.load_data(filepath):
        loader.current_file=filepath
        if loader.prepare_data():
            loader.analyze_data()
            loader.plot_results()
            loader.generate_report()
            return loader
    return None
if __name__=="__main__":
    import sys
    if len(sys.argv)>1:
        filepath=sys.argv[1]
        quick_analyze(filepath)
    else:
        interactive_mode()
