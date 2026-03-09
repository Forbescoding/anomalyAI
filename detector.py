import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
import time
from datetime import datetime
import os
import sys
import warnings
warnings.filterwarnings('ignore')
class RealtimeAnomalyDetector:    
    def __init__(self, model_path=None, use_analyzer_data=False, force_demo=False):
        print("=" * 60)
        print("ИНИЦИАЛИЗАЦИЯ ДЕТЕКТОРА АНОМАЛИЙ")
        print("=" * 60)        
        self.model = None
        self.scaler = None
        self.threshold = None
        self.feature_names = None
        self.model_path = model_path
        self.demo_mode = force_demo
        self.stats = {
            'total_packets': 0,
            'normal_packets': 0,
            'anomaly_packets': 0,
            'anomaly_rate': 0.0,
            'last_anomaly_time': None
        }
        self.buffer = []
        self.buffer_size = 10
        if force_demo:
            print("\n" + "="*60)
            print(" ЗАПУСК В ДЕМО-РЕЖИМЕ")
            print("="*60)
            self.demo_mode = True
            self.threshold = 0.5
            print(f"Порог: {self.threshold:.4f}")
            return
        print("\nПопытка загрузки модели...")
        if not self._load_model_safe():
            print("\n" + "="*60)
            print(" НЕ УДАЛОСЬ ЗАГРУЗИТЬ МОДЕЛЬ")
            print("="*60)
            print("Возможные причины:")
            print("1. Модель не обучена (запустите сначала model_training.py)")
            print("2. Проблема совместимости версий TensorFlow")
            print("3. Файлы модели повреждены или отсутствуют")
            print("\n  РАБОЧИЙ РЕЖИМ НЕ АКТИВИРОВАН - МОДЕЛЬ НЕ НАЙДЕНА")
            self.demo_mode = False
            self.model = None
            self.threshold = 0.1
            print(f"Порог (по умолчанию): {self.threshold:.4f}")
    def _load_model_safe(self):
        possible_paths = [
            ('anomaly_detector.h5', 'anomaly_threshold.pkl', 'scaler.pkl', 'feature_names.pkl'),
            ('model.h5', 'threshold.pkl', 'scaler.pkl', 'features.pkl'),
            ('best_model.h5', 'best_threshold.pkl', 'scaler.pkl', None),
        ]
        analyzer_models = [f for f in os.listdir('.') if f.endswith('_model.h5') or 
                          (f.endswith('.h5') and 'anomaly' in f.lower())]
        for model_file in analyzer_models:
            base = model_file.replace('.h5', '')
            possible_paths.append((
                model_file,
                f'{base}_threshold.pkl',
                f'{base}_scaler.pkl',
                f'{base}_features.pkl'
            ))
        for model_path, threshold_path, scaler_path, features_path in possible_paths:
            try:
                if os.path.exists(model_path):
                    print(f"\nПопытка загрузки модели: {model_path}")
                    try:
                        self.model = keras.models.load_model(model_path)
                        print(f"   Модель загружена стандартным способом")
                    except Exception as e1:
                        print(f"   Стандартная загрузка не удалась: {type(e1).__name__}")
                        try:
                            self.model = keras.models.load_model(
                                model_path, 
                                compile=False
                            )
                            print(f"   Модель загружена без компиляции")
                        except Exception as e2:
                            print(f"   Загрузка без компиляции не удалась: {type(e2).__name__}")
                            try:
                                custom_objects = {
                                    'mse': tf.keras.losses.MeanSquaredError(),
                                    'mae': tf.keras.losses.MeanAbsoluteError()
                                }
                                self.model = keras.models.load_model(
                                    model_path,
                                    custom_objects=custom_objects,
                                    compile=False
                                )
                                print(f"   Модель загружена с пользовательскими объектами")
                            except Exception as e3:
                                print(f"   Все способы загрузки не удались")
                                continue
                    if os.path.exists(threshold_path):
                        try:
                            self.threshold = joblib.load(threshold_path)
                            print(f"   Порог загружен: {self.threshold:.4f}")
                        except:
                            self.threshold = 0.1
                            print(f"   Используется порог по умолчанию: {self.threshold:.4f}")
                    else:
                        self.threshold = 0.1
                        print(f"   Файл порога не найден, используется: {self.threshold:.4f}")
                    if scaler_path and os.path.exists(scaler_path):
                        try:
                            self.scaler = joblib.load(scaler_path)
                            print(f"   Scaler загружен")
                        except:
                            self.scaler = None
                            print(f"   Scaler не загружен")
                    if features_path and os.path.exists(features_path):
                        try:
                            self.feature_names = joblib.load(features_path)
                            print(f"   Загружено {len(self.feature_names)} признаков")
                        except:
                            self.feature_names = None
                    print(f"\n Модель успешно загружена из {model_path}")
                    return True                    
            except Exception as e:
                continue        
        return False    
    def preprocess_packet(self, packet_data):
        try:
            if isinstance(packet_data, np.ndarray):
                if packet_data.ndim == 1:
                    packet_data = packet_data.reshape(1, -1)
                if self.scaler:
                    return self.scaler.transform(packet_data)[0]
                return packet_data[0]            
            if isinstance(packet_data, dict):
                df = pd.DataFrame([packet_data])
                if self.feature_names:
                    for feature in self.feature_names:
                        if feature not in df.columns:
                            df[feature] = 0
                    df = df[self.feature_names]
                if self.scaler:
                    scaled = self.scaler.transform(df)
                    return scaled[0]
                else:
                    return df.values[0]
            if isinstance(packet_data, list):
                return np.array(packet_data)
            return packet_data
        except Exception as e:
            print(f" Ошибка предобработки: {e}")
            return np.random.randn(78)
    def detect_anomaly(self, packet_data):        
        if self.demo_mode:
            mse = np.random.random()
            is_anomaly = mse>self.threshold
            result = {
                'is_anomaly': bool(is_anomaly),
                'mse_score': float(mse),
                'threshold': float(self.threshold),
                'anomaly_probability': float(min(mse / self.threshold, 1.0)),
                'timestamp': datetime.now().isoformat(),
                'mode': 'DEMO'
            }
        elif self.model is None:
            mse = np.random.random() * 0.5
            is_anomaly = False              
            result = {
                'is_anomaly': bool(is_anomaly),
                'mse_score': float(mse),
                'threshold': float(self.threshold),
                'anomaly_probability': float(min(mse / self.threshold, 0.3)),
                'timestamp': datetime.now().isoformat(),
                'mode': 'WORKING (NO MODEL)',
                'warning': 'Модель не загружена'
            }
        else:
            try:
                processed = self.preprocess_packet(packet_data)
                if processed.ndim == 1:
                    processed = processed.reshape(1, -1)
                self.buffer.append(processed[0])
                if len(self.buffer) > self.buffer_size:
                    self.buffer.pop(0)
                if len(self.buffer) == self.buffer_size:
                    buffer_array = np.array(self.buffer)
                    current_features = np.mean(buffer_array, axis=0).reshape(1, -1)
                else:
                    current_features = processed
                reconstructed = self.model.predict(current_features, verbose=0)
                mse = np.mean(np.square(current_features - reconstructed))
                is_anomaly = mse>self.threshold
                result = {
                    'is_anomaly': bool(is_anomaly),
                    'mse_score': float(mse),
                    'threshold': float(self.threshold),
                    'anomaly_probability': float(min(mse / self.threshold, 1.0)),
                    'timestamp': datetime.now().isoformat(),
                    'mode': 'WORKING'
                }
            except Exception as e:
                print(f" Ошибка обнаружения: {e}")
                result = {
                    'is_anomaly': False,
                    'mse_score': 0.0,
                    'threshold': float(self.threshold),
                    'anomaly_probability': 0.0,
                    'timestamp': datetime.now().isoformat(),
                    'mode': 'WORKING (ERROR)',
                    'error': str(e)
                }
        self._update_stats(result['is_anomaly'], result['mse_score'])
        return result
    def _update_stats(self, is_anomaly, mse):
        self.stats['total_packets'] += 1        
        if is_anomaly:
            self.stats['anomaly_packets'] += 1
            self.stats['last_anomaly_time'] = datetime.now()
        else:
            self.stats['normal_packets'] += 1        
        self.stats['anomaly_rate'] = (
            self.stats['anomaly_packets'] / max(self.stats['total_packets'], 1)
        )
    def generate_test_packet(self, anomaly_type='normal'):
        np.random.seed(int(time.time() * 1000) % 1000)
        if self.model and not self.demo_mode:
            input_shape = self.model.input_shape[1]
            if anomaly_type == 'normal':
                return np.random.randn(input_shape) * 0.5
            else:
                return np.random.randn(input_shape) * 2.0
        if anomaly_type == 'normal':
            packet = {
                'duration': float(np.random.exponential(300)),
                'protocol': np.random.choice(['TCP', 'UDP', 'ICMP']),
                'src_port': int(np.random.randint(1024, 65535)),
                'dst_port': int(np.random.choice([80, 443, 22, 53])),
                'src_bytes': int(np.clip(np.random.lognormal(5, 2), 0, 10000)),
                'dst_bytes': int(np.clip(np.random.lognormal(6, 2), 0, 10000)),
                'packet_count': int(np.random.poisson(10))
            }
        elif anomaly_type == 'ddos':
            packet = {
                'duration': float(np.random.exponential(10)),
                'protocol': 'TCP',
                'src_port': int(np.random.randint(1024, 65535)),
                'dst_port': 80,
                'src_bytes': int(np.random.randint(40, 100)),
                'dst_bytes': 0,
                'packet_count': int(np.random.poisson(1000))
            }
        elif anomaly_type == 'scan':
            packet = {
                'duration': float(np.random.exponential(1)),
                'protocol': 'TCP',
                'src_port': int(np.random.randint(1024, 65535)),
                'dst_port': int(np.random.randint(1, 1024)),
                'src_bytes': int(np.random.randint(40, 60)),
                'dst_bytes': 0,
                'packet_count': int(np.random.poisson(50))
            }
        else:  
            packet = {
                'duration': float(np.random.exponential(600)),
                'protocol': 'TCP',
                'src_port': int(np.random.randint(1024, 65535)),
                'dst_port': 22,
                'src_bytes': int(np.random.randint(100, 500)),
                'dst_bytes': int(np.random.randint(100, 500)),
                'packet_count': int(np.random.poisson(500))
            }
        return packet
    def print_stats(self):
        print("\n" + "="*60)
        print("СТАТИСТИКА ДЕТЕКТОРА")
        print("="*60)
        print(f"Всего пакетов: {self.stats['total_packets']}")
        print(f"Нормальных: {self.stats['normal_packets']}")
        print(f"Аномальных: {self.stats['anomaly_packets']}")
        print(f"Процент аномалий: {self.stats['anomaly_rate']*100:.2f}%")
        if self.demo_mode:
            print(f"Режим: ДЕМО(случайные данные)")
        elif self.model is None:
            print(f"Режим: РАБОЧИЙ(МОДЕЛЬ НЕ ЗАГРУЖЕНА)")
        else:
            print(f"Режим: РАБОЧИЙ(модель загружена)")
        if self.stats['last_anomaly_time']:
            print(f"Последняя аномалия: {self.stats['last_anomaly_time'].strftime('%H:%M:%S')}")
        print("="*60)
def simulate_realtime_traffic(detector, duration_seconds=30):
    print(f"\nСимуляция трафика ({duration_seconds} секунд)")
    print("Нажмите Ctrl+C для остановки\n")
    start_time = time.time()
    packets_per_second = 2
    try:
        packet_count = 0
        while time.time() - start_time < duration_seconds:
            for _ in range(packets_per_second):
                packet_count += 1
                traffic_type = np.random.choice(
                    ['normal', 'ddos', 'scan', 'bruteforce'],
                    p=[0.6, 0.15, 0.15, 0.1]
                )
                packet = detector.generate_test_packet(traffic_type)
                result = detector.detect_anomaly(packet)
                mode_indicator = "D" if result.get('mode') == 'DEMO' else "W"
                
                if result['is_anomaly']:
                    print(f"️ [{packet_count:3d}] АНОМАЛИЯ | {traffic_type:10} | MSE: {result['mse_score']:.4f} | Вероятность: {result['anomaly_probability']*100:.1f}% | [{mode_indicator}]")
                else:
                    print(f" [{packet_count:3d}] НОРМА    | {traffic_type:10} | MSE: {result['mse_score']:.4f} | Вероятность: {result['anomaly_probability']*100:.1f}% | [{mode_indicator}]")
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\nСимуляция остановлена пользователем")    
    detector.print_stats()
def main():
    print("=" * 60)
    print("ДЕТЕКТОР АНОМАЛИЙ В РЕАЛЬНОМ ВРЕМЕНИ")
    print("=" * 60)
    print("Для навигации используйте цифры от 1 до 5")
    print()
    print("Выберите режим работы:")
    print("1. Рабочий режим (с загрузкой модели)")
    print("2. Демо-режим (без модели, случайные данные)")
    print("3. Протестировать на данных распознавателя")
    print("4. Информация о системе")
    print("5. Выход")
    while True:
        try:
            choice = input("\nВаш выбор (1-5): ").strip()
            if choice == '1':
                print("\nЗапуск в рабочем режиме...")
                detector = RealtimeAnomalyDetector(force_demo=False)
                break
            elif choice == '2':
                print("\nЗапуск в демо-режиме...")
                detector = RealtimeAnomalyDetector(force_demo=True)
                break
            elif choice == '3':
                detector = RealtimeAnomalyDetector(force_demo=False)
                test_files = [f for f in os.listdir('.') if 'test' in f.lower() and f.endswith('.npy')]
                if test_files:
                    print(f"\nНайдены файлы для тестирования:")
                    for i, f in enumerate(test_files, 1):
                        print(f"  {i}. {f}")                    
                    file_choice = input("\nВыберите файл (или Enter для отмены): ").strip()
                    if file_choice:
                        try:
                            idx = int(file_choice) - 1
                            if 0 <= idx < len(test_files):
                                test_data = np.load(test_files[idx])
                                print(f"\nЗагружено {len(test_data)} тестовых примеров")                                
                                print("\nРезультаты тестирования:")
                                for i in range(min(20, len(test_data))):
                                    result = detector.detect_anomaly(test_data[i])
                                    status = "⚠" if result['is_anomaly'] else "✓"
                                    mode = result.get('mode', 'UNKNOWN')
                                    print(f"{status} Пример {i+1:2d}: MSE={result['mse_score']:.4f}, Вероятность={result['anomaly_probability']*100:.1f}% | [{mode}]")                                
                                input("\nНажмите Enter для продолжения...")
                        except Exception as e:
                            print(f" Ошибка при загрузке файла: {e}")
                            input("Нажмите Enter для продолжения...")
                else:
                    print(" Не найдены файлы для тестирования")
                    input("Нажмите Enter для продолжения...")
                continue                
            elif choice == '4':
                print("\n" + "="*60)
                print("ИНФОРМАЦИЯ О СИСТЕМЕ")
                print("="*60)
                print("Детектор аномалий сетевого трафика")
                print(f"Версия Python: {sys.version}")
                print(f"TensorFlow: {tf.__version__}")
                print(f"NumPy: {np.__version__}")
                print(f"Pandas: {pd.__version__}")
                print("\nРежимы работы:")
                print("- Рабочий: использует обученную модель")
                print("- Демо: генерирует случайные результаты")
                print("- Тест: проверка на загруженных данных")
                print("\nДоступные модели:")
                models_found = False
                for f in os.listdir('.'):
                    if f.endswith('.h5'):
                        print(f"  - {f}")
                        models_found = True
                if not models_found:
                    print("  Модели не найдены")                
                input("\nНажмите Enter для продолжения...")
                continue
            elif choice == '5':
                print("\n Выполнение закончено")
                return
            else:
                print(" Неверный выбор. Введите число от 1 до 5")                
        except KeyboardInterrupt:
            print("\n\n Программа прервана пользователем")
            return
        except Exception as e:
            print(f" Ошибка: {e}")
            continue
    while True:
        print("\n" + "-"*50)
        print("МЕНЮ ДЕТЕКТОРА:")
        print("1. Запустить симуляцию трафика")
        print("2. Протестировать один пакет")
        print("3. Показать статистику")
        print("4. Переключить режим (рабочий/демо)")
        print("5. Вернуться в главное меню")
        print("6. Выход")
        try:
            sub_choice = input("\nВыберите действие: ").strip()
            if sub_choice == '1':
                duration = input("Длительность симуляции (сек, по умолчанию 20): ").strip()
                duration = int(duration) if duration else 20
                simulate_realtime_traffic(detector, duration)
                input("\nНажмите Enter для продолжения...")
            elif sub_choice == '2':
                print("\nТипы пакетов:")
                print("1. Нормальный")
                print("2. DDoS атака")
                print("3. Сканирование портов")
                print("4. Brute-force атака")
                print("5. Случайный")
                p_type = input("Выберите тип (1-5): ").strip()
                types = {
                    '1': 'normal', 
                    '2': 'ddos', 
                    '3': 'scan', 
                    '4': 'bruteforce',
                    '5': np.random.choice(['normal', 'ddos', 'scan', 'bruteforce'])
                }
                if p_type in types:
                    packet = detector.generate_test_packet(types[p_type])
                    result = detector.detect_anomaly(packet)                    
                    print(f"\n{'='*60}")
                    print(f"РЕЗУЛЬТАТ ТЕСТИРОВАНИЯ")
                    print(f"{'='*60}")
                    print(f"Тип пакета: {types[p_type]}")
                    print(f"Статус: {' АНОМАЛИЯ' if result['is_anomaly'] else '✓ НОРМА'}")
                    print(f"MSE ошибка: {result['mse_score']:.4f}")
                    print(f"Порог: {result['threshold']:.4f}")
                    print(f"Вероятность аномалии: {result['anomaly_probability']*100:.1f}%")
                    print(f"Режим: {result.get('mode', 'UNKNOWN')}")
                    print(f"Время: {result['timestamp']}")
                    if 'warning' in result:
                        print(f"Предупреждение: {result['warning']}")
                    if 'error' in result:
                        print(f"Ошибка: {result['error']}")
                else:
                    print(" Неверный тип пакета")
                input("\nНажмите Enter для продолжения...")
            elif sub_choice == '3':
                detector.print_stats()
                input("\nНажмите Enter для продолжения...")
            elif sub_choice == '4':
                current_mode = detector.demo_mode
                detector.demo_mode = not current_mode                
                if detector.demo_mode:
                    print(f"\n Режим переключен на: ДЕМО")
                else:
                    if detector.model is None:
                        print(f"\n Режим переключен на: РАБОЧИЙ(НО МОДЕЛЬ НЕ ЗАГРУЖЕНА)")
                        print("Детектор будет работать с ограниченной функциональностью")
                    else:
                        print(f"\n Режим переключен на: РАБОЧИЙ(МОДЕЛЬ ЗАГРУЖЕНА)")
                input("Нажмите Enter для продолжения...")
            elif sub_choice == '5':
                print("\nВозврат в главное меню...")
                main()
                return
            elif sub_choice == '6':
                print("\n Выполнение закончено")
                break            
            else:
                print(" Неверный выбор. Введите 1-6")
                input("Нажмите Enter для продолжения...")                
        except KeyboardInterrupt:
            print("\n\n Программа прервана пользователем")
            break
        except Exception as e:
            print(f" Ошибка: {e}")
            import traceback
            traceback.print_exc()
            input("Нажмите Enter для продолжения...")
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Программа прервана пользователем")
    except Exception as e:
        print(f"\n Непредвиденная ошибка: {e}")
        import traceback
        traceback.print_exc()    
    print("\n" + "="*60)
    print(" РАБОТА ДЕТЕКТОРА ЗАВЕРШЕНА")
    print("="*60)
    input("\nНажмите Enter для закрытия окна...")
