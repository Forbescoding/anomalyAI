import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import seaborn as sns
import os
import glob
import re
class AnomalyDetector:
    def __init__(self, input_dim=None):
        self.input_dim = input_dim
        self.model = None
        self.threshold = None
        self.history = None
        self.feature_names = None        
    def load_data_from_analyzer(self, prefix=None):
        print("\n" + "=" * 60)
        print("ЗАГРУЗКА ДАННЫХ ИЗ РАСПОЗНАВАТЕЛЯ")
        print("=" * 60)
        if prefix:
            X_train_file = f'{prefix}_X_train.npy'
            X_test_file = f'{prefix}_X_test.npy'
            y_train_file = f'{prefix}_y_train.npy'
            y_test_file = f'{prefix}_y_test.npy'
            features_file = f'{prefix}_features.pkl'
        else:
            X_train_files = glob.glob('*_X_train_*.npy')
            if not X_train_files:
                print("Не найдены обработанные данные")
                return False
            latest = max(X_train_files, key=os.path.getctime)
            print(f"Найден файл: {latest}")
            import re
            match = re.search(r'(.*?)_X_train_(.*)\.npy', latest)
            if match:
                prefix = match.group(1)
                suffix = match.group(2)
                print(f"Префикс: {prefix}, суффикс: {suffix}")
                X_train_file = latest
                X_test_file = f'{prefix}_X_test_{suffix}.npy'
                y_train_file = f'{prefix}_y_train_{suffix}.npy'
                y_test_file = f'{prefix}_y_test_{suffix}.npy'
                features_file = f'{prefix}_features_{suffix}.pkl'
            else:
                print("Не удалось определить префикс из имени файла")
                return False
        try:
            required_files = [X_train_file, X_test_file, y_train_file, y_test_file, features_file]
            missing_files = []
            for file in required_files:
                if not os.path.exists(file):
                    missing_files.append(file)
            if missing_files:
                print("Отсутствуют файлы:")
                for file in missing_files:
                    print(f"  {file}")
                return False
            self.X_train = np.load(X_train_file)
            self.X_test = np.load(X_test_file)
            self.y_train = np.load(y_train_file)
            self.y_test = np.load(y_test_file)
            self.feature_names = joblib.load(features_file)
            self.input_dim = self.X_train.shape[1]
            print("\nПроверка соответствия данных:")
            print(f"  X_train: {self.X_train.shape}")
            print(f"  y_train: {self.y_train.shape}")
            if len(self.X_train) != len(self.y_train):
                print(f"  ОШИБКА: Несоответствие размеров!")
                print(f"    X_train: {len(self.X_train)} записей")
                print(f"    y_train: {len(self.y_train)} меток")
                print(f"  Обрезаем метки до размера данных ")
                min_len = min(len(self.X_train), len(self.y_train))
                self.X_train = self.X_train[:min_len]
                self.y_train = self.y_train[:min_len]
                print(f"  После обрезки: X_train {self.X_train.shape}, y_train {self.y_train.shape}")
            else:
                print(f"  Размеры совпадают")
            print(f"\nДанные загружены:")
            print(f"  X_train: {self.X_train.shape}")
            print(f"  X_test: {self.X_test.shape}")
            print(f"  Признаков: {self.input_dim}")
            train_labels = np.bincount(self.y_train.astype(int))
            test_labels = np.bincount(self.y_test.astype(int))
            train_display = train_labels if len(train_labels) >= 2 else [train_labels[0] if len(train_labels) > 0 else 0, 0]
            test_display = test_labels if len(test_labels) >= 2 else [test_labels[0] if len(test_labels) > 0 else 0, 0]
            print(f"  Норма/аномалия в train: {train_display}")
            print(f"  Норма/аномалия в test: {test_display}")
            return True
        except Exception as e:
            print(f"Ошибка загрузки: {e}")
            import traceback
            traceback.print_exc()
            return False
    def build_model(self):
        if self.input_dim is None:
            raise ValueError("Не указана размерность входных данных")
        inputs = keras.Input(shape=(self.input_dim,))
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        encoded = layers.Dense(16, activation='relu', name='bottleneck')(x)
        x = layers.Dense(32, activation='relu')(encoded)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        decoded = layers.Dense(self.input_dim, activation='linear')(x)
        autoencoder = keras.Model(inputs, decoded, name='autoencoder')
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return autoencoder
    def train(self, X_train=None, epochs=100, batch_size=32, validation_split=0.2):
        if X_train is None:
            if hasattr(self, 'X_train') and self.X_train is not None:
                X_train = self.X_train
            else:
                raise ValueError("Нет данных для обучения!")
        print("\nНачало обучения модели...")
        print(f"Данные: {X_train.shape}")
        X_train_normal = X_train
        if hasattr(self, 'y_train') and self.y_train is not None:
            if len(self.y_train) != len(X_train):
                print(f"ПРЕДУПРЕЖДЕНИЕ: Размеры X_train ({len(X_train)}) и y_train ({len(self.y_train)}) не совпадают!")
                print("Используем все данные для обучения")
            else:
                normal_indices = np.where(self.y_train == 0)[0]
                valid_indices = normal_indices[normal_indices < len(X_train)]
                if len(valid_indices) > 0:
                    X_train_normal = X_train[valid_indices]
                    print(f"Обучение только на нормальном трафике: {len(X_train_normal)} из {len(X_train)} записей")
                    print(f"Использованы индексы: {len(valid_indices)} из {len(normal_indices)} найденных")
                else:
                    print("ВНИМАНИЕ: Нет валидных индексов нормального трафика!")
                    print(f"Все найденные индексы: {normal_indices[:10] if len(normal_indices) > 10 else normal_indices}")
                    print(f"Размер X_train: {len(X_train)}")
        else:
            print("Обучение на всех данных (без учителя)")
        if len(X_train_normal) == 0:
            raise ValueError("Нет данных для обучения!")
        self.input_dim = X_train.shape[1]
        self.model = self.build_model()
        print(f"Модель построена. Параметров: {self.model.count_params():,}")
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        self.history = self.model.fit(
            X_train_normal, X_train_normal,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            shuffle=True,
            verbose=1,
            callbacks=[early_stopping, reduce_lr]
        )
        self._set_threshold(X_train_normal)
        print(f"\nОбучение завершено. Порог аномалий: {self.threshold:.4f}")  
    def _set_threshold(self, X_train, percentile=95):
        train_pred = self.model.predict(X_train, verbose=0)
        mse = np.mean(np.square(X_train - train_pred), axis=1)
        self.threshold = np.percentile(mse, percentile) 
    def predict(self, X):
        if self.model is None:
            raise ValueError("Модель не обучена!")
        reconstructed = self.model.predict(X, verbose=0)
        mse = np.mean(np.square(X - reconstructed), axis=1)
        predictions = (mse > self.threshold).astype(int)
        return predictions, mse    
    def evaluate(self, X_test=None, y_test=None):
        if X_test is None:
            if hasattr(self, 'X_test'):
                X_test = self.X_test
            else:
                raise ValueError("Нет тестовых данных!")
        if y_test is None:
            if hasattr(self, 'y_test'):
                y_test = self.y_test
            else:
                print("\nНет меток для оценки. Показываем распределение ошибок...")
                y_pred, mse_scores = self.predict(X_test)
                self._plot_error_distribution_only(mse_scores)
                return None
        y_pred, mse_scores = self.predict(X_test)
        print("\n" + "="*60)
        print("КЛАССИФИКАЦИОННЫЙ ОТЧЕТ")
        print("="*60)
        print(classification_report(y_test, y_pred, 
                                   target_names=['Норма', 'Аномалия']))
        auc = roc_auc_score(y_test, mse_scores)
        print(f"\nAUC-ROC: {auc:.4f}")
        self._plot_confusion_matrix(y_test, y_pred)
        self._plot_error_distribution(mse_scores, y_test)
        from sklearn.metrics import precision_score, recall_score, f1_score
        self.metrics = {
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc': auc,
            'threshold': self.threshold
        }        
        return {
            'predictions': y_pred,
            'scores': mse_scores,
            'auc': auc,
            'threshold': self.threshold,
            'metrics': self.metrics
        }    
    def _plot_error_distribution_only(self, mse_scores):
        plt.figure(figsize=(10, 6))
        plt.hist(mse_scores, bins=50, alpha=0.7, color='blue', density=True)
        plt.axvline(self.threshold, color='black', linestyle='--', 
                   linewidth=2, label=f'Порог: {self.threshold:.4f}')
        plt.xlabel('Ошибка реконструкции (MSE)')
        plt.ylabel('Плотность')
        plt.title('Распределение ошибок реконструкции')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('error_distribution.png', dpi=100)
        plt.show()    
    def _plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Норма', 'Аномалия'],
                    yticklabels=['Норма', 'Аномалия'])
        plt.title('Матрица ошибок')
        plt.ylabel('Истинные значения')
        plt.xlabel('Предсказания')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=100)
        plt.show()
    def _plot_error_distribution(self, errors, labels):
        plt.figure(figsize=(10, 6))
        normal_errors = errors[labels == 0]
        anomaly_errors = errors[labels == 1]        
        plt.hist(normal_errors, bins=50, alpha=0.7, 
                label='Нормальный трафик', color='green', density=True)
        plt.hist(anomaly_errors, bins=50, alpha=0.7, 
                label='Аномальный трафик', color='red', density=True)
        plt.axvline(self.threshold, color='black', linestyle='--', 
                   linewidth=2, label=f'Порог: {self.threshold:.4f}')
        plt.xlabel('Ошибка реконструкции (MSE)')
        plt.ylabel('Плотность')
        plt.title('Распределение ошибок реконструкции')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('error_distribution.png', dpi=100)
        plt.show()
    def plot_training_history(self):
        if self.history is None:
            print("Нет данных об обучении.")
            return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(self.history.history['loss'], label='Обучающая')
        ax1.plot(self.history.history['val_loss'], label='Валидационная')
        ax1.set_title('Функция потерь')
        ax1.set_xlabel('Эпоха')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend()
        ax1.grid(True)
        ax2.plot(self.history.history['mae'], label='Обучающая')
        ax2.plot(self.history.history['val_mae'], label='Валидационная')
        ax2.set_title('Средняя абсолютная ошибка')
        ax2.set_xlabel('Эпоха')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=100)
        plt.show()    
    def save_model(self, path='anomaly_detector'):
        if self.model:
            self.model.save(f'{path}.h5')
            joblib.dump(self.threshold, f'{path}_threshold.pkl')
            joblib.dump(self.feature_names, f'{path}_features.pkl')
            if hasattr(self, 'metrics'):
                joblib.dump(self.metrics, f'{path}_metrics.pkl')
            print(f"\n Модель сохранена в {path}.h5")
            print(f"  Порог: {self.threshold:.4f}")
    def load_model(self, path='anomaly_detector'):
        self.model = keras.models.load_model(f'{path}.h5')
        self.threshold = joblib.load(f'{path}_threshold.pkl')
        try:
            self.feature_names = joblib.load(f'{path}_features.pkl')
            self.input_dim = len(self.feature_names)
        except:
            self.input_dim = self.model.input_shape[1]
        print(f"\n Модель загружена из {path}.h5")
        print(f"  Порог: {self.threshold:.4f}")
        print(f"  Признаков: {self.input_dim}")
def main():
    print("=" * 60)
    print("ОБУЧЕНИЕ МОДЕЛИ ОБНАРУЖЕНИЯ АНОМАЛИЙ")
    print("=" * 60)
    detector = AnomalyDetector()    
    print("\nПоиск данных от распознавателя...")
    if detector.load_data_from_analyzer():
        print("\n Данные от распознавателя загружены!")
        use_analyzer = input("\nИспользовать эти данные для обучения? (y/n): ").strip().lower()
        if use_analyzer == 'y':
            X_train = detector.X_train
            y_train = detector.y_train
            X_test = detector.X_test
            y_test = detector.y_test
        else:
            try:
                X_train = np.load('X_train.npy')
                X_test = np.load('X_test.npy')
                y_train = np.load('y_train.npy')
                y_test = np.load('y_test.npy')
                detector.feature_names = joblib.load('feature_names.pkl')
                print("\n Стандартные данные загружены")
            except:
                print("Стандартные данные не найдены")
                return
    else:
        try:
            X_train = np.load('X_train.npy')
            X_test = np.load('X_test.npy')
            y_train = np.load('y_train.npy')
            y_test = np.load('y_test.npy')
            detector.feature_names = joblib.load('feature_names.pkl')
            print("\n Стандартные данные загружены")
        except FileNotFoundError:
            print("\n Данные не найдены!")
            print("Сначала запустите data_preparation.py")
            return
    print(f"\nДанные загружены:")
    print(f"  Обучающая выборка: {X_train.shape}")
    print(f"  Тестовая выборка: {X_test.shape}")
    print(f"  Норма/аномалия в train: {np.bincount(y_train.astype(int))}")
    print(f"  Норма/аномалия в test: {np.bincount(y_test.astype(int))}")
    print("\nОбучение на нормальном трафике...")
    X_train_normal = X_train[y_train == 0]
    epochs = input("\nКоличество эпох (по умолчанию 50): ").strip()
    epochs = int(epochs) if epochs else 50
    batch_size = input("Размер батча (по умолчанию 64): ").strip()
    batch_size = int(batch_size) if batch_size else 64
    detector.train(X_train_normal, epochs=epochs, batch_size=batch_size)
    detector.plot_training_history()
    results = detector.evaluate(X_test, y_test)
    model_name = input("\nИмя для сохранения модели (по умолчанию 'anomaly_detector'): ").strip()
    if not model_name:
        model_name = 'anomaly_detector'
    detector.save_model(model_name)
    print("\n" + "="*60)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("="*60)
    print(f"Модель сохранена: {model_name}.h5")
    if results:
        print(f"AUC-ROC: {results['auc']:.4f}")
        print(f"F1-score: {results['metrics']['f1']:.4f}")
if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    main()
