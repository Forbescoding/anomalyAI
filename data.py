import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings
warnings.filterwarnings('ignore')
class TrafficDataGenerator:
    def __init__(self):
        self.scaler=StandardScaler()
        self.label_encoders={}
    def load_from_analyzer(self, analyzer_file=None):
        print("\n" + "="*60)
        print("ЗАГРУЗКА ДАННЫХ ИЗ РАСПОЗНАВАТЕЛЯ")
        print("="*60)
        if analyzer_file and os.path.exists(analyzer_file):
            print(f"Загрузка из файла: {analyzer_file}")
            data = np.load(analyzer_file, allow_pickle=True).item()
            self.X_train=data.get('X_train')
            self.X_test=data.get('X_test')
            self.y_train=data.get('y_train')
            self.y_test=data.get('y_test')
            self.feature_names=data.get('feature_names')
            self.scaler=data.get('scaler')
            print(f"Данные загружены:")
            print(f"X_train: {self.X_train.shape}")
            print(f"X_test: {self.X_test.shape}")
            print(f"Признаков: {len(self.feature_names)}")
            return True
        processed_files = [f for f in os.listdir('.') if f.startswith('processed_') and f.endswith('.npy')]
        if processed_files:
            print("Найдены обработанные файлы:")
            for f in processed_files:
                print(f"  - {f}")
            import re
            pattern=r'processed_X_train_(\d+_\d+)\.npy'
            latest=None
            latest_time=None
            for f in processed_files:
                match=re.search(pattern, f)
                if match:
                    if latest_time is None or match.group(1)>latest_time:
                        latest_time=match.group(1)
                        latest=match.group(1)
            if latest:
                self.X_train=np.load(f'processed_X_train_{latest}.npy')
                self.X_test=np.load(f'processed_X_test_{latest}.npy')
                self.y_train=np.load(f'processed_y_train_{latest}.npy')
                self.y_test=np.load(f'processed_y_test_{latest}.npy')
                self.feature_names=joblib.load(f'processed_features_{latest}.pkl')
                self.scaler=joblib.load(f'processed_scaler_{latest}.pkl')
                print(f"\n Загружены данные от {latest}")
                print(f" X_train: {self.X_train.shape}")
                print(f" X_test: {self.X_test.shape}")
                return True
        return False
    def use_external_data(self, X_train, X_test, y_train, y_test, feature_names, scaler):
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test
        self.feature_names=feature_names
        self.scaler=scaler
        print("\n Внешние данные загружены в генератор")
    def generate_dataset(self, n_normal=10000, n_anomaly=2000, use_real=False):
        if use_real and hasattr(self, 'X_train'):
            print("\nИспользуются загруженные реальные данные")
            return self.X_train,self.X_test,self.y_train,self.y_test,self.feature_names
        print("\nГенерация синтетических данных...")
        normal_data=self._generate_normal_traffic(n_normal)
        anomaly_data=self._generate_anomalous_traffic(n_anomaly)        
        all_data=pd.concat([normal_data, anomaly_data], ignore_index=True)
        all_data=all_data.sample(frac=1, random_state=42).reset_index(drop=True)
        return all_data    
    def _generate_normal_traffic(self, n_samples):
        np.random.seed(42)
        data = {
            'duration': np.clip(np.random.exponential(300, n_samples), 0, 3600),
            'protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], n_samples, p=[0.7, 0.25, 0.05]),
            'src_port': np.random.randint(1024, 65535, n_samples),
            'dst_port': np.random.choice([80, 443, 22, 53, 21], n_samples),
            'src_bytes': np.clip(np.random.lognormal(5, 2, n_samples), 0, 10000).astype(int),
            'dst_bytes': np.clip(np.random.lognormal(6, 2, n_samples), 0, 10000).astype(int),
            'packet_count': np.random.poisson(10, n_samples),
            'urgent_flag': np.random.binomial(1, 0.001, n_samples),
            'ack_flag': np.random.binomial(1, 0.8, n_samples),
            'psh_flag': np.random.binomial(1, 0.3, n_samples),
            'rst_flag': np.random.binomial(1, 0.01, n_samples),
            'syn_flag': np.random.binomial(1, 0.4, n_samples),
            'fin_flag': np.random.binomial(1, 0.3, n_samples),
            'same_srv_rate': np.random.beta(8, 2, n_samples),
            'diff_srv_rate': np.random.beta(2, 8, n_samples),
            'serror_rate': np.random.beta(1, 15, n_samples),
            'srv_serror_rate': np.random.beta(1, 15, n_samples),
            'logged_in': np.random.binomial(1, 0.6, n_samples),
            'is_guest_login': np.random.binomial(1, 0.01, n_samples),
            'count': np.random.poisson(10, n_samples),
            'srv_count': np.random.poisson(10, n_samples),
            'dst_host_count': np.random.poisson(255, n_samples),
            'dst_host_srv_count': np.random.poisson(255, n_samples),
            'dst_host_same_srv_rate': np.random.beta(8, 2, n_samples),
            'dst_host_diff_srv_rate': np.random.beta(2, 8, n_samples),
        }
        df=pd.DataFrame(data)
        df['label']=0
        return df
    def _generate_anomalous_traffic(self, n_samples):
        np.random.seed(42)
        n_ddos=n_samples//3
        n_port_scan=n_samples//3
        n_bruteforce=n_samples-n_ddos-n_port_scan        
        anomalies=[]
        ddos_data={
            'duration': np.random.exponential(10, n_ddos),
            'protocol': 'TCP',
            'src_port': np.random.randint(1024, 65535, n_ddos),
            'dst_port': np.random.choice([80, 443], n_ddos),
            'src_bytes': np.random.randint(40, 100, n_ddos),
            'dst_bytes': np.zeros(n_ddos),
            'packet_count': np.random.poisson(1000, n_ddos),
            'urgent_flag': 0,
            'ack_flag': 1,
            'psh_flag': 0,
            'rst_flag': 0,
            'syn_flag': 1,
            'fin_flag': 0,
            'same_srv_rate': np.random.beta(1, 10, n_ddos),
            'diff_srv_rate': np.random.beta(10, 1, n_ddos),
            'serror_rate': np.random.beta(10, 1, n_ddos),
            'srv_serror_rate': np.random.beta(10, 1, n_ddos),
            'logged_in': 0,
            'is_guest_login': 1,
            'count': np.random.poisson(1000, n_ddos),
            'srv_count': np.random.poisson(1000, n_ddos),
            'dst_host_count': np.random.poisson(10, n_ddos),
            'dst_host_srv_count': np.random.poisson(10, n_ddos),
            'dst_host_same_srv_rate': np.random.beta(1, 10, n_ddos),
            'dst_host_diff_srv_rate': np.random.beta(10, 1, n_ddos),
        }
        ddos_df=pd.DataFrame(ddos_data)
        ddos_df['label']=1
        anomalies.append(ddos_df)
        scan_data={
            'duration': np.random.exponential(1, n_port_scan),
            'protocol': 'TCP',
            'src_port': np.random.randint(1024, 65535, n_port_scan),
            'dst_port': np.random.randint(1, 1024, n_port_scan),
            'src_bytes': np.random.randint(40, 60, n_port_scan),
            'dst_bytes': np.zeros(n_port_scan),
            'packet_count': np.random.poisson(50, n_port_scan),
            'urgent_flag': 0,
            'ack_flag': 0,
            'psh_flag': 0,
            'rst_flag': 1,
            'syn_flag': 1,
            'fin_flag': 0,
            'same_srv_rate': np.random.beta(1, 5, n_port_scan),
            'diff_srv_rate': np.random.beta(5, 1, n_port_scan),
            'serror_rate': np.random.beta(5, 1, n_port_scan),
            'srv_serror_rate': np.random.beta(5, 1, n_port_scan),
            'logged_in': 0,
            'is_guest_login': 1,
            'count': np.random.poisson(100, n_port_scan),
            'srv_count': np.random.poisson(100, n_port_scan),
            'dst_host_count': np.random.poisson(255, n_port_scan),
            'dst_host_srv_count': np.random.poisson(255, n_port_scan),
            'dst_host_same_srv_rate': np.random.beta(1, 5, n_port_scan),
            'dst_host_diff_srv_rate': np.random.beta(5, 1, n_port_scan),
        }
        scan_df=pd.DataFrame(scan_data)
        scan_df['label']=1
        anomalies.append(scan_df)
        brute_data = {
            'duration': np.random.exponential(600, n_bruteforce),
            'protocol': 'TCP',
            'src_port': np.random.randint(1024, 65535, n_bruteforce),
            'dst_port': 22,
            'src_bytes': np.random.randint(100, 500, n_bruteforce),
            'dst_bytes': np.random.randint(100, 500, n_bruteforce),
            'packet_count': np.random.poisson(500, n_bruteforce),
            'urgent_flag': 0,
            'ack_flag': 1,
            'psh_flag': 1,
            'rst_flag': 0,
            'syn_flag': 0,
            'fin_flag': 0,
            'same_srv_rate': np.random.beta(9, 1, n_bruteforce),
            'diff_srv_rate': np.random.beta(1, 9, n_bruteforce),
            'serror_rate': np.random.beta(1, 5, n_bruteforce),
            'srv_serror_rate': np.random.beta(1, 5, n_bruteforce),
            'logged_in': 0,
            'is_guest_login': 1,
            'count': np.random.poisson(100, n_bruteforce),
            'srv_count': np.random.poisson(100, n_bruteforce),
            'dst_host_count': np.random.poisson(1, n_bruteforce),
            'dst_host_srv_count': np.random.poisson(1, n_bruteforce),
            'dst_host_same_srv_rate': np.random.beta(9, 1, n_bruteforce),
            'dst_host_diff_srv_rate': np.random.beta(1, 9, n_bruteforce),
        }
        brute_df=pd.DataFrame(brute_data)
        brute_df['label']=1
        anomalies.append(brute_df)        
        return pd.concat(anomalies, ignore_index=True)



    
    def preprocess_data(self, data):
        df=data.copy()
        if 'protocol' in df.columns:
            le=LabelEncoder()
            df['protocol']=le.fit_transform(df['protocol'])
            self.label_encoders['protocol']=le
        if 'label' in df.columns:
            y=df['label'].values
            X=df.drop('label', axis=1)
        else:
            y=None
            X=df
        X_scaled=self.scaler.fit_transform(X)
        return X_scaled, y, X.columns.tolist()
    
def main():
    print("="*60)
    print("ПОДГОТОВКА ДАННЫХ ДЛЯ ОБНАРУЖЕНИЯ АНОМАЛИЙ")
    print("="*60)
    generator=TrafficDataGenerator()
    print("\nВыберите источник данных:")
    print("1. Сгенерировать синтетические данные")
    print("2. Загрузить данные из распознавателя")
    print("3. Указать путь к файлу с данными")
    choice = input("\nВаш выбор (1-3): ").strip()
    if choice=='2':
        if generator.load_from_analyzer():
            X_train, X_test, y_train, y_test=generator.X_train, generator.X_test, generator.y_train, generator.y_test
            feature_names=generator.feature_names
            print("\n Данные успешно загружены из распознавателя!")
        else:
            print("\n Не удалось загрузить данные из распознавателя.")
            print("Генерация синтетических данных...")
            data=generator.generate_dataset()
            X, y, feature_names=generator.preprocess_data(data)
            X_train, X_test, y_train, y_test=train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
    elif choice=='3':
        filepath=input("Введите путь к файлу с данными: ").strip()
        if generator.load_from_analyzer(filepath):
            X_train, X_test, y_train, y_test=generator.X_train, generator.X_test, generator.y_train, generator.y_test
            feature_names=generator.feature_names
        else:
            print(" Не удалось загрузить файл. Генерируем синтетические данные...")
            data=generator.generate_dataset()
            X, y, feature_names=generator.preprocess_data(data)
            X_train, X_test, y_train, y_test=train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
    else: 
        print("\nГенерация синтетических данных...")
        data=generator.generate_dataset(n_normal=10000, n_anomaly=2000)
        X, y, feature_names=generator.preprocess_data(data)
        X_train, X_test, y_train, y_test=train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    print("\nСохранение данных...")
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    joblib.dump(feature_names, 'feature_names.pkl')
    joblib.dump(generator.scaler, 'scaler.pkl')
    joblib.dump(generator.label_encoders, 'label_encoders.pkl')
    print(f"\nДанные сохранены:")
    print(f"  X_train.npy: {X_train.shape}")
    print(f"  X_test.npy: {X_test.shape}")
    print(f"  feature_names.pkl: {len(feature_names)} признаков")
if __name__=="__main__":
    main()
