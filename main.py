import os
import subprocess
import sys
import time
import glob
def debug_print(msg):
    print(f" DEBUG: {msg}")
    sys.stdout.flush()
def run_script(script_name, description):
    debug_print(f"Попытка запуска {script_name}")
    print(f"\n{'='*70}")
    print(f"ЗАПУСК: {description}")
    print(f"{'='*70}")
    if not os.path.exists(script_name):
        print(f"Файл {script_name} не найден!")
        debug_print(f"Файл {script_name} отсутствует")
        input("Нажмите Enter для продолжения")
        return False
    try:
        debug_print(f"Запуск процесса: {sys.executable} {script_name}")
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=300
        )
        debug_print(f"Процесс завершен с кодом {result.returncode}")
        if result.stdout:
            print("\n--- ВЫВОД ПРОГРАММЫ ---")
            print(result.stdout)
        if result.stderr:
            print("\n--- ОШИБКИ/ПРЕДУПРЕЖДЕНИЯ ---")
            print(result.stderr)        
        if result.returncode == 0:
            print(f"\n {description} завершена успешно!")
            debug_print("Успешное завершение")
            return True
        else:
            print(f"\n {description} завершилась с ошибкой (код: {result.returncode})")
            debug_print(f"Ошибка выполнения, код: {result.returncode}")
            return False
    except subprocess.TimeoutExpired:
        print(f"\n {description} превысила время ожидания")
        debug_print("Таймаут")
        return False
    except Exception as e:
        print(f"\n Ошибка при запуске {script_name}: {e}")
        debug_print(f"Исключение: {type(e).__name__} - {e}")
        return False
def check_files():
    required_files = {
        'data.py': 'Подготовка данных',
        'model.py': 'Обучение модели',
        'detector.py': 'Детектор'
    }
    print("\n" + "="*70)
    print("ПРОВЕРКА ФАЙЛОВ")
    print("="*70)
    all_found = True
    current_dir = os.getcwd()
    print(f"Текущая директория: {current_dir}")
    print("\nСодержимое директории:")
    py_files = [f for f in os.listdir('.') if f.endswith('.py')]
    if py_files:
        for f in py_files:
            print(f"  - {f}")
    else:
        print(" Python файлы не найдены!")
    print()    
    for file_name, description in required_files.items():
        if os.path.exists(file_name):
            print(f" {file_name} - найден")
        else:
            print(f" {file_name} - не найден! (нужен для: {description})")
            all_found = False
    return all_found
def show_analyzer_data():
    analyzer_files = []
    for f in os.listdir('.'):
        if f.startswith('processed_') and (f.endswith('.npy') or f.endswith('.pkl')):
            analyzer_files.append(f)
    if analyzer_files:
        print("\n Найдены данные от распознавателя:")
        for f in sorted(analyzer_files)[:10]:
            size = os.path.getsize(f) / 1024
            print(f"  - {f} ({size:.1f} KB)")
        if len(analyzer_files) > 10:
            print(f"  ... и еще {len(analyzer_files) - 10} файлов")
        return True
    return False
def load_analyzer_data():
    print("\n Работа с данными от распознавателя")
    X_train_files = glob.glob('processed_*_X_train_*.npy')
    if not X_train_files:
        print(" Данные от распознавателя не найдены!")
        print(" Сначала запустите data.py и выберите загрузку из распознавателя")
        input("Нажмите Enter для продолжения...")
        return False
    print("\nНайденные наборы данных:")
    for i, f in enumerate(X_train_files, 1):
        parts = f.split('_')
        timestamp = '_'.join(parts[2:4]) if len(parts) >= 4 else "unknown"
        size = os.path.getsize(f) / 1024 / 1024
        print(f"  {i}. Данные от {timestamp.replace('.npy', '')} ({size:.1f} MB)")
    data_choice = input("\nВыберите набор данных (или Enter для отмены): ").strip()
    if not data_choice:
        return False
    try:
        idx = int(data_choice) - 1
        if 0 <= idx < len(X_train_files):
            selected_file = X_train_files[idx]
            base_parts = selected_file.split('_')
            if len(base_parts) >= 4:
                timestamp = '_'.join(base_parts[2:4]).replace('.npy', '')
            else:
                print(" Не удалось определить временную метку")
                return False
            print(f"\nЗагрузка данных с меткой: {timestamp}")
            import shutil            
            files_to_copy = [
                (f'processed_X_train_{timestamp}.npy', 'X_train.npy'),
                (f'processed_X_test_{timestamp}.npy', 'X_test.npy'),
                (f'processed_y_train_{timestamp}.npy', 'y_train.npy'),
                (f'processed_y_test_{timestamp}.npy', 'y_test.npy'),
                (f'processed_features_{timestamp}.pkl', 'feature_names.pkl'),
                (f'processed_scaler_{timestamp}.pkl', 'scaler.pkl')
            ]            
            copied = 0
            for src, dst in files_to_copy:
                if os.path.exists(src):
                    shutil.copy2(src, dst)
                    print(f" {dst} загружен")
                    copied += 1
                else:
                    print(f" {src} не найден")
            if copied >= 4:
                print(f"\n Данные от распознавателя загружены ({copied} файлов)")
                train_now = input("\nОбучить модель на этих данных? (y/n): ").strip().lower()
                if train_now == 'y':
                    run_script('model.py', 'Обучение модели на данных распознавателя')
                return True
            else:
                print(f"\n Загружено только {copied} из 6 файлов. Данные неполные.")
                return False
    except ValueError:
        print(" Ошибка: введите число")
    except Exception as e:
        print(f" Ошибка при загрузке данных: {e}")
    return False
def main():    
    debug_print("Начало выполнения main()") 
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=" * 70)
    print("ЗАПУСК СИСТЕМЫ ОБНАРУЖЕНИЯ АНОМАЛИЙ")
    print("=" * 70)
    if not check_files():
        print("\n Не все необходимые файлы найдены!")
        print("   Убедитесь, что все три программы находятся в текущей папке:")
        print("   - data.py")
        print("   - model.py")
        print("   - detector.py")
        input("\nНажмите Enter для выхода...")
        return
    has_analyzer_data = show_analyzer_data()
    while True:
        print("\n" + "="*70)
        print(" МЕНЮ ВЫБОРА РЕЖИМА")
        print("="*70)
        print("1. Полный цикл (подготовка + обучение + детектор)")
        print("2. Только подготовка данных")
        print("3. Только обучение модели")
        print("4. Только детектор")
        print("5. Загрузить данные из распознавателя и обучить")
        print("6. Выход")
        if has_analyzer_data:
            print("\n Выберите пункт 5 для работы с данными от распознавателя")
        try:
            choice = input("\n ВАш выбор (1-6): ").strip()
            debug_print(f"Пользователь ввел: '{choice}'")
        except (EOFError, KeyboardInterrupt):
            debug_print("Прерывание ввода")
            print("\n\n Программа прервана пользователем")
            break
        if choice == '1':
            print("\n ЗАПУСК ПОЛНОГО ЦИКЛА")
            if not run_script('data.py', 'Подготовка данных'):
                print("\n Остановка: ошибка при подготовке данных")
                input("Нажмите Enter для продолжения")
                continue
            if not os.path.exists('X_train.npy'):
                print("\n Файлы данных не созданы!")
                input("Нажмите Enter для продолжения")
                continue
            if not run_script('model.py', 'Обучение модели'):
                print("\n Остановка: ошибка при обучении модели")
                input("Нажмите Enter для продолжения")
                continue
            print("\n ЗАПУСК ДЕТЕКТОРА")
            print("Детектор будет запущен.")
            print("После завершения работы детектора вернитесь сюда.")
            input("\nНажмите Enter для запуска детектора")
            try:
                subprocess.run([sys.executable, 'detector.py'])
            except Exception as e:
                print(f"Ошибка при запуске детектора: {e}")
            
            print("\n Полный цикл завершен!")
            input("Нажмите Enter для продолжения")
        elif choice == '2':
            run_script('data.py', 'Подготовка данных')
            input("\nНажмите Enter для продолжения")
        elif choice == '3':
            if not os.path.exists('X_train.npy'):
                print("\n Данные для обучения не найдены!")
                print("Сначала выполните подготовку данных (пункт 2)")
                input("Нажмите Enter для продолжения")
                continue
            run_script('model.py', 'Обучение модели')
            input("\nНажмите Enter для продолжения")
        elif choice == '4':
            print("\n ЗАПУСК ДЕТЕКТОРА")
            print("Детектор будет запущен.")
            print("Для выхода из детектора используйте его меню.")
            input("\nНажмите Enter для запуска детектора")
            try:
                subprocess.run([sys.executable, 'detector.py'])
            except Exception as e:
                print(f" Ошибка при запуске детектора: {e}")
            print("\n Детектор завершил работу")
            input("Нажмите Enter для продолжения")
        elif choice == '5':
            if load_analyzer_data():
                print("\n Данные успешно загружены!")
            else:
                print("\n Не удалось загрузить данные")
            input("Нажмите Enter для продолжения")
        elif choice == '6':
            print("\n Выполнение завершено")
            debug_print("Выход по выбору пользователя")
            break
        else:
            print(f" Неверный выбор: '{choice}'. Введите число от 1 до 6")
            input("Нажмите Enter для продолжения")
    debug_print("Завершение main()")
if __name__ == "__main__":
    debug_print("Скрипт запущен")
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nПрограмма прервана пользователем")
        debug_print("KeyboardInterrupt в главном блоке")
    except Exception as e:
        print(f"\n Критическая ошибка: {e}")
        debug_print(f"Исключение в главном блоке: {type(e).__name__} - {e}")
        import traceback
        traceback.print_exc()    
    debug_print("Скрипт завершается")
    print("\n" + "="*70)
    print("РАБОТА ПРОГРАММЫ ЗАВЕРШЕНА")
    print("="*70)
    input("\nНажмите Enter для закрытия окна...")
