# rootkit_detector_advanced.py

import pandas as pd
import numpy as np
import re
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from datetime import datetime
import json
import sys

class RootkitDetector:
    """
    Самообучающаяся система обнаружения руткитов
    """
    
    def __init__(self, model_dir='models', logs_dir='detection_logs', data_dir='training_data'):
        # Определяем базовую директорию (для exe)
        if getattr(sys, 'frozen', False):
            self.base_dir = os.path.dirname(sys.executable)
        else:
            self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.model_dir = os.path.join(self.base_dir, model_dir)
        self.logs_dir = os.path.join(self.base_dir, logs_dir)
        self.data_dir = os.path.join(self.base_dir, data_dir)
        
        self.model_filename = os.path.join(self.model_dir, 'isolation_forest_model.pkl')
        self.scaler_filename = os.path.join(self.model_dir, 'scaler.pkl')
        self.training_history_file = os.path.join(self.model_dir, 'training_history.json')
        
        # Создаем директории
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.model = None
        self.scaler = None
        self.training_history = self._load_training_history()
    
    def _load_training_history(self):
        """Загрузка истории обучения"""
        if os.path.exists(self.training_history_file):
            with open(self.training_history_file, 'r') as f:
                return json.load(f)
        return {
            'version': 1,
            'training_sessions': [],
            'total_samples': 0,
            'last_updated': None
        }
    
    def _save_training_history(self):
        """Сохранение истории обучения"""
        with open(self.training_history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def analyze_results(self, predictions, scan_mode='unknown'):
        """
        Анализ результатов сканирования для РЕАЛЬНЫХ сценариев
        
        Parameters:
        -----------
        predictions : array-like
            Предсказания модели (1 = норма, -1 = аномалия)
        scan_mode : str
            'clean' - сканируем заведомо чистую систему (для тестирования)
            'infected' - сканируем заведомо зараженную систему (для тестирования)
            'unknown' - реальное сканирование (не знаем результат)
        
        Returns:
        --------
        dict : Словарь с результатами анализа
        """
        y_pred = np.array([1 if x == -1 else 0 for x in predictions])
        
        total = len(predictions)
        anomalies_count = np.sum(y_pred == 1)
        normal_count = total - anomalies_count
        anomaly_percentage = (anomalies_count / total) * 100
        
        # Базовая статистика
        results = {
            'total_samples': total,
            'normal_count': int(normal_count),
            'anomalies_count': int(anomalies_count),
            'anomaly_percentage': anomaly_percentage,
            'scan_mode': scan_mode
        }

        if scan_mode == 'unknown':
            if anomaly_percentage < 6:
                threat_level = 'НИЗКИЙ'
                threat_risk = 'LOW'
                assessment = 'Система чистая или минимальные ложные срабатывания'
            elif anomaly_percentage < 10:
                threat_level = 'СРЕДНИЙ'
                threat_risk = 'MEDIUM'
                assessment = 'Обнаружена подозрительная активность, требуется проверка'
            elif anomaly_percentage < 15:
                threat_level = 'ВЫСОКИЙ'
                threat_risk = 'HIGH'
                assessment = 'Высокая вероятность заражения руткитом!'
            else:
                threat_level = 'КРИТИЧЕСКИЙ'
                threat_risk = 'CRITICAL'
                assessment = 'ВНИМАНИЕ! Обнаружена активная вредоносная деятельность!'
            
            results.update({
                'threat_level': threat_level,
                'threat_risk': threat_risk,
                'assessment': assessment,
                'recommendation': self._get_recommendation(anomaly_percentage)
            })
        
        # ТЕСТИРОВАНИЕ МОДЕЛИ (с известным результатом)
        elif scan_mode in ['clean', 'infected']:
            metrics = self._calculate_test_metrics(predictions, scan_mode)
            results.update(metrics)
        
        return results
    
    def _get_recommendation(self, anomaly_percentage):
        """Рекомендации на основе процента аномалий"""
        if anomaly_percentage < 3:
            return " Система работает нормально. Продолжайте мониторинг."
        elif anomaly_percentage < 8:
            return " Рекомендуется проверить подозрительные процессы и логи."
        elif anomaly_percentage < 15:
            return " СРОЧНО: Проведите полное сканирование антивирусом и проверьте автозагрузку."
        else:
            return " КРИТИЧНО: Изолируйте систему, создайте резервную копию данных, переустановите ОС."
    
    def _calculate_test_metrics(self, predictions, scan_mode):
        """
        Расчет метрик для ТЕСТИРОВАНИЯ модели
        """
        y_pred = np.array([1 if x == -1 else 0 for x in predictions])
        
        if scan_mode == 'clean':
            y_true = np.zeros(len(predictions), dtype=int)
            
            tn = np.sum((y_pred == 0) & (y_true == 0))  
            fp = np.sum((y_pred == 1) & (y_true == 0))  #
            fn = 0  
            tp = 0  

            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0          
            accuracy = tn / len(y_pred)                            
            
            quality_assessment = self._assess_clean_system_quality(specificity, fpr)
            
            return {
                'test_type': 'Тестирование на ЧИСТОЙ системе',
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'true_positives': int(tp),
                'accuracy': accuracy,
                'specificity': specificity,
                'false_positive_rate': fpr,
                'quality_assessment': quality_assessment,
                'interpretation': f"Модель правильно определила {tn:,} ({accuracy*100:.1f}%) нормальных записей. "
                                f"Ложных срабатываний: {fp:,} ({fpr*100:.1f}%)"
            }
        
        elif scan_mode == 'infected':

            anomaly_percentage = (np.sum(y_pred == 1) / len(y_pred)) * 100

            tp = np.sum(y_pred == 1) 
            fn = 0
            tn = np.sum(y_pred == 0) 
            fp = 0 
            
            detection_rate = anomaly_percentage
            quality_assessment = self._assess_infected_system_quality(detection_rate)
            
            return {
                'test_type': 'Тестирование на ЗАРАЖЕННОЙ системе',
                'detected_anomalies': int(tp),
                'normal_behavior': int(tn),
                'detection_rate': detection_rate,
                'quality_assessment': quality_assessment,
                'interpretation': f"Обнаружено {tp:,} подозрительных записей ({detection_rate:.1f}% от общего числа). "
                                f"Это соответствует активности руткита при нормальной работе системы."
            }
    
    def _assess_clean_system_quality(self, specificity, fpr):
        """Оценка качества на чистой системе"""
        if specificity >= 0.97 and fpr <= 0.03:
            return {
                'grade': 'ОТЛИЧНО ',
                'score': 'A',
                'comment': 'Модель показывает превосходные результаты! Минимум ложных срабатываний.'
            }
        elif specificity >= 0.93 and fpr <= 0.07:
            return {
                'grade': 'ХОРОШО ✓',
                'score': 'B',
                'comment': 'Качество модели достаточное для практического применения.'
            }
        elif specificity >= 0.85 and fpr <= 0.15:
            return {
                'grade': 'УДОВЛЕТВОРИТЕЛЬНО ',
                'score': 'C',
                'comment': 'Требуется дополнительная настройка - слишком много ложных тревог.'
            }
        else:
            return {
                'grade': 'ТРЕБУЕТ УЛУЧШЕНИЯ ',
                'score': 'D',
                'comment': 'Необходимо переобучение с увеличенным contamination.'
            }
    
    def _assess_infected_system_quality(self, detection_rate):
        """Оценка качества на зараженной системе"""
        if 10 <= detection_rate <= 30:
            return {
                'grade': 'ОТЛИЧНО ',
                'score': 'A',
                'comment': 'Оптимальный уровень обнаружения! Соответствует реальной активности руткита.'
            }
        elif 5 <= detection_rate < 10 or 30 < detection_rate <= 40:
            return {
                'grade': 'ХОРОШО ✓',
                'score': 'B',
                'comment': 'Приемлемый уровень обнаружения, возможна небольшая корректировка.'
            }
        elif detection_rate < 5:
            return {
                'grade': 'НИЗКАЯ ЧУВСТВИТЕЛЬНОСТЬ ',
                'score': 'C',
                'comment': 'Модель пропускает слишком много угроз. Уменьшите contamination.'
            }
        else:
            return {
                'grade': 'ВЫСОКАЯ ЧУВСТВИТЕЛЬНОСТЬ ',
                'score': 'C',
                'comment': 'Слишком много обнаружений - возможны ложные срабатывания.'
            }
    
    def print_analysis(self, results):
        """
        Красивый вывод результатов анализа
        """
        print(f"\n{'='*70}")
        print(f"{' РЕЗУЛЬТАТЫ АНАЛИЗА':^70}")
        print(f"{'='*70}")
        
        print(f"\n СТАТИСТИКА СКАНИРОВАНИЯ:")
        print(f"{'─'*70}")
        print(f"Всего проанализировано записей  : {results['total_samples']:,}")
        print(f"Нормальных записей              : {results['normal_count']:,} ({(results['normal_count']/results['total_samples']*100):.1f}%)")
        print(f"Обнаружено аномалий             : {results['anomalies_count']:,} ({results['anomaly_percentage']:.2f}%)")
        
        if results['scan_mode'] == 'unknown':
            # РЕАЛЬНОЕ СКАНИРОВАНИЕ
            print(f"\n{'─'*70}")
            print(f"{' ОЦЕНКА УГРОЗЫ':^70}")
            print(f"{'─'*70}")
            print(f"\n{results['threat_risk']} Уровень угрозы: {results['threat_level']}")
            print(f"\n Заключение:")
            print(f"   {results['assessment']}")
            print(f"\n Рекомендация:")
            print(f"   {results['recommendation']}")
            
            # Интерпретация процента аномалий
            print(f"\n{'─'*70}")
            print(f"СПРАВКА: Что означают проценты аномалий?")
            print(f"{'─'*70}")
            print(f"  0-6%    Норма (чистая система + минимум ошибок модели)")
            print(f"  6-9%    Подозрительно (требует внимания)")
            print(f"  9-15%   Высокая угроза (вероятное заражение)")
            print(f"  >15%    Критично (активная вредоносная деятельность)")
            
        elif results['scan_mode'] in ['clean', 'infected']:
            print(f"\n{'─'*70}")
            print(f"{ + results['test_type']:^70}")
            print(f"{'─'*70}")
            
            if results['scan_mode'] == 'clean':
                print(f"\n МЕТРИКИ КАЧЕСТВА:")
                print(f"   Accuracy (Точность)        : {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
                print(f"   Specificity (Специфичность): {results['specificity']:.4f} ({results['specificity']*100:.2f}%)")
                print(f"   FPR (Ложные тревоги)       : {results['false_positive_rate']:.4f} ({results['false_positive_rate']*100:.2f}%)")
                
                print(f"\n МАТРИЦА РЕЗУЛЬТАТОВ:")
                print(f"   True Negatives  (правильно норма)  : {results['true_negatives']:,}")
                print(f"   False Positives (ложная тревога)   : {results['false_positives']:,}")
                
                print(f"\n ИНТЕРПРЕТАЦИЯ:")
                print(f"   {results['interpretation']}")
            
            elif results['scan_mode'] == 'infected':
                print(f"\n РЕЗУЛЬТАТЫ ОБНАРУЖЕНИЯ:")
                print(f"   Обнаружено аномалий        : {results['detected_anomalies']:,}")
                print(f"   Процент обнаружения        : {results['detection_rate']:.2f}%")
                print(f"   Нормальное поведение       : {results['normal_behavior']:,}")
                
                print(f"\n ИНТЕРПРЕТАЦИЯ:")
                print(f"   {results['interpretation']}")
            
            qa = results['quality_assessment']
            print(f"\n{'─'*70}")
            print(f"ОЦЕНКА ДЛЯ ДИПЛОМА: {qa['grade']} (Оценка: {qa['score']})")
            print(f"└─ {qa['comment']}")
            print(f"{'─'*70}")
        
        print(f"{'='*70}\n")
    
    def plot_results(self, df, results, save_path=None):
        """
        Визуализация результатов
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Анализ обнаружения руткитов', fontsize=16, fontweight='bold')
        
        ax1 = axes[0, 0]
        labels = ['Normal', 'Anomaly']
        sizes = [results['normal_count'], results['anomalies_count']]
        colors = ['#3498db', '#e74c3c']
        explode = (0, 0.1)
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
                                             colors=colors, explode=explode, startangle=90,
                                             textprops={'fontsize': 12, 'fontweight': 'bold'})
        for autotext in autotexts:
            autotext.set_color('white')
        
        ax1.set_title(f'Распределение\n(Всего: {results["total_samples"]:,} записей)', 
                     fontsize=14, fontweight='bold')
        
        # 2. Распределение anomaly scores
        ax2 = axes[0, 1]
        ax2.hist(df['anomaly_score'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        threshold = df[df['is_anomaly']]['anomaly_score'].max()
        ax2.axvline(threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Порог аномалий ({threshold:.3f})')
        ax2.set_xlabel('Anomaly Score', fontsize=11)
        ax2.set_ylabel('Частота', fontsize=11)
        ax2.set_title('Распределение оценок аномальности', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Топ системных вызовов с аномалиями
        ax3 = axes[1, 0]
        if results['anomalies_count'] > 0:
            top_anomalous = df[df['is_anomaly']]['syscall'].value_counts().head(10)
            colors_bar = ['#e74c3c' if i < 3 else '#f39c12' for i in range(len(top_anomalous))]
            top_anomalous.plot(kind='barh', ax=ax3, color=colors_bar, edgecolor='black')
            ax3.set_xlabel('Количество аномалий', fontsize=11)
            ax3.set_title('ТОП-10 системных вызовов с аномалиями', fontsize=14, fontweight='bold')
            ax3.grid(axis='x', alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Аномалий не обнаружено', 
                    ha='center', va='center', fontsize=14, transform=ax3.transAxes)
            ax3.set_xticks([])
            ax3.set_yticks([])
        
        # 4. Временная линия аномалий
        ax4 = axes[1, 1]
        window = max(1000, len(df) // 100)
        anomaly_timeline = df['is_anomaly'].rolling(window=window).sum()
        ax4.plot(df.index, anomaly_timeline, color='red', linewidth=2)
        ax4.fill_between(df.index, anomaly_timeline, alpha=0.3, color='red')
        ax4.set_xlabel('Индекс записи', fontsize=11)
        ax4.set_ylabel('Количество аномалий (скользящее окно)', fontsize=11)
        ax4.set_title(f'Временная динамика обнаружения (окно={window})', 
                     fontsize=14, fontweight='bold')
        ax4.grid(alpha=0.3)
        
        # Добавляем информацию об уровне угрозы
        if results['scan_mode'] == 'unknown':
            fig.text(0.5, 0.02, 
                    f"{results['threat_risk']} Уровень угрозы: {results['threat_level']} | "
                    f"Аномалий: {results['anomaly_percentage']:.1f}%",
                    ha='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ График сохранен: {save_path}")
        
        plt.show()
    
    def extract_features(self, file_path):
        """Извлечение признаков из файла логов"""
        print(f"\n{'='*60}")
        print(f"ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ: {os.path.basename(file_path)}")
        print(f"{'='*60}")
        
        durations = []
        syscalls = []
        timestamps = []
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                match_duration = re.search(r'\(\s*([\d\.]+)\s*ms\)', line)
                if match_duration:
                    val = float(match_duration.group(1))
                    
                    if val > 5000.0:
                        continue
                        
                    match_syscall = re.search(r'([a-zA-Z_]\w*)\s*\(', line)
                    
                    durations.append(val)
                    syscalls.append(match_syscall.group(1) if match_syscall else 'unknown')
                    timestamps.append(i)
                
                if i % 100000 == 0 and i > 0:
                    print(f"  Обработано строк: {i:,}")
        
        print(f" Всего извлечено вызовов: {len(durations):,}")
        
        df = pd.DataFrame({
            'duration': durations,
            'syscall': syscalls,
            'timestamp': timestamps
        })
        
        print("\n[+] Создание дополнительных признаков...")
        
        window_size = 100
        df['duration_rolling_mean'] = df['duration'].rolling(window=window_size, min_periods=1).mean()
        df['duration_rolling_std'] = df['duration'].rolling(window=window_size, min_periods=1).std().fillna(0)
        df['duration_rolling_max'] = df['duration'].rolling(window=window_size, min_periods=1).max()
        
        syscall_counts = df['syscall'].value_counts()
        df['syscall_frequency'] = df['syscall'].map(syscall_counts)
        df['time_delta'] = df['timestamp'].diff().fillna(1)
        
        median_duration = df['duration'].median()
        df['deviation_from_median'] = abs(df['duration'] - median_duration)
        df['duration_zscore'] = (df['duration'] - df['duration'].mean()) / (df['duration'].std() + 1e-10)
        
        print(f"✓ Создано признаков: {len(df.columns)}")
        return df
    
    def train(self, normal_files, contamination=0.05, incremental=False):
        """
        Обучение или дообучение модели
        """
        print(f"\n{'='*60}")
        print("РЕЖИМ ОБУЧЕНИЯ МОДЕЛИ" if not incremental else "РЕЖИМ ДООБУЧЕНИЯ МОДЕЛИ")
        print(f"{'='*60}")
        
        all_data = []
        
        for file_path in normal_files:
            if not os.path.exists(file_path):
                print(f"❌ ОШИБКА: Файл не найден: {file_path}")
                continue
            
            print(f"\n[*] Загрузка: {os.path.basename(file_path)}")
            df = self.extract_features(file_path)
            all_data.append(df)
            
            backup_file = os.path.join(self.data_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            df.to_csv(backup_file, index=False)
            print(f"  └─ Резервная копия: {os.path.basename(backup_file)}")
        
        if not all_data:
            print(" ОШИБКА: Нет данных для обучения!")
            return
        
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"\n[+] Объединено записей: {len(combined_df):,}")
        
        if incremental and os.path.exists(self.model_filename):
            print("\n[+] Загрузка предыдущих обучающих данных...")
            old_training_files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.startswith('training_')]
            if old_training_files:
                old_data = [pd.read_csv(f) for f in old_training_files]
                combined_df = pd.concat([combined_df] + old_data, ignore_index=True)
                print(f"  └─ Общий объем данных: {len(combined_df):,}")
        
        feature_columns = [
            'duration', 'duration_rolling_mean', 'duration_rolling_std',
            'duration_rolling_max', 'syscall_frequency', 'time_delta',
            'deviation_from_median', 'duration_zscore'
        ]
        
        X = combined_df[feature_columns].fillna(0)
        
        print("\n[+] Нормализация признаков...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"\n[+] Обучение Isolation Forest (contamination={contamination})...")
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1,
            verbose=1
        )
        
        self.model.fit(X_scaled)
        
        self.training_history['version'] += 1
        self.training_history['total_samples'] = len(combined_df)
        self.training_history['last_updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.training_history['training_sessions'].append({
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'samples': len(combined_df),
            'contamination': contamination,
            'files': [os.path.basename(f) for f in normal_files],
            'incremental': incremental
        })
        
        print("\n[+] Сохранение модели...")
        joblib.dump(self.model, self.model_filename)
        joblib.dump(self.scaler, self.scaler_filename)
        self._save_training_history()
        
        print(f"\n{'='*60}")
        print("✓ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print(f"{'='*60}")
        print(f"Версия модели: {self.training_history['version']}")
        print(f"Обучающих примеров: {len(combined_df):,}")
        print(f"Contamination: {contamination}")
        print(f"{'='*60}")
    
    def detect(self, file_path, visualize=True, save_results=True, scan_mode='unknown'):
        """
        Обнаружение аномалий в файле
        
        Parameters:
        -----------
        file_path : str
            Путь к файлу для анализа
        visualize : bool
            Показать визуализацию
        save_results : bool
            Сохранить результаты
        scan_mode : str
            'unknown' - обычное сканирование
            'clean' - тестирование на чистой системе
            'infected' - тестирование на зараженной системе
        """
        print(f"\n{'='*60}")
        print(f"РЕЖИМ ОБНАРУЖЕНИЯ")
        print(f"{'='*60}")
        
        if not os.path.exists(self.model_filename):
            print(" ОШИБКА: Модель не обучена!")
            print("Сначала выполните обучение (пункт 1 или 2 в меню)")
            return
        
        print("\n[+] Загрузка модели...")
        self.model = joblib.load(self.model_filename)
        self.scaler = joblib.load(self.scaler_filename)
        print("✓ Модель загружена")
        
        df = self.extract_features(file_path)
        
        feature_columns = [
            'duration', 'duration_rolling_mean', 'duration_rolling_std',
            'duration_rolling_max', 'syscall_frequency', 'time_delta',
            'deviation_from_median', 'duration_zscore'
        ]
        
        X = df[feature_columns].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        print("\n[+] Анализ данных...")
        predictions = self.model.predict(X_scaled)
        anomaly_scores = self.model.score_samples(X_scaled)
        
        df['prediction'] = predictions
        df['anomaly_score'] = anomaly_scores
        df['is_anomaly'] = predictions == -1
        
        # Анализ результатов
        results = self.analyze_results(predictions, scan_mode)
        self.print_analysis(results)
        

        if save_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            results_file = os.path.join(self.logs_dir, f'detection_{timestamp}.csv')
            df.to_csv(results_file, index=False)
            print(f"✓ Результаты сохранены: {results_file}")
            
            report_file = os.path.join(self.logs_dir, f'report_{timestamp}.json')
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"✓ Отчет сохранен: {report_file}")

        if visualize:
            print("\n[+] Создание визуализации...")
            plot_file = os.path.join(self.logs_dir, f'analysis_{timestamp}.png') if save_results else None
            self.plot_results(df, results, plot_file)

def print_menu():
    """Меню программы"""
    print("\n╔═══════════════════════════════════════════════════════╗")
    print("║                    ГЛАВНОЕ МЕНЮ                       ║")
    print("╠═══════════════════════════════════════════════════════╣")
    print("║  [1]  Первичное обучение модели                       ║")
    print("║  [2]  Дообучение существующей модели                  ║")
    print("║  [3]  Сканирование файла                              ║")
    print("║  [4]  Информация о модели                             ║")
    print("╚═══════════════════════════════════════════════════════╝")

def get_file_path(prompt):
    """Запрос пути к файлу"""
    print(prompt)
    print("Введите полный путь к файлу (или перетащите файл в терминал):")
    file_path = input("> ").strip().strip('"').strip("'")
    
    if not file_path:
        return None
    
    if not os.path.exists(file_path):
        print(f"\n ОШИБКА: Файл не найден: {file_path}")
        return None
    
    return file_path

def model_info(detector):
    """Информация о модели"""
    print("\n" + "="*60)
    print("ИНФОРМАЦИЯ О МОДЕЛИ")
    print("="*60)
    
    if os.path.exists(detector.model_filename):
        print("✓ Модель обучена и готова к использованию")
        print(f"\n Расположение модели:")
        print(f"  └─ {detector.model_filename}")
        print(f"  └─ {detector.scaler_filename}")
        
        model_size = os.path.getsize(detector.model_filename) / 1024
        scaler_size = os.path.getsize(detector.scaler_filename) / 1024
        print(f"\n Размер файлов:")
        print(f"  └─ Модель: {model_size:.2f} KB")
        print(f"  └─ Scaler: {scaler_size:.2f} KB")
        
        print(f"\n История обучения:")
        print(f"  └─ Версия модели: {detector.training_history['version']}")
        print(f"  └─ Всего обучающих примеров: {detector.training_history['total_samples']:,}")
        print(f"  └─ Последнее обновление: {detector.training_history['last_updated']}")
        print(f"  └─ Количество сессий обучения: {len(detector.training_history['training_sessions'])}")
        
        if detector.training_history['training_sessions']:
            print(f"\n Последние 3 сессии обучения:")
            for i, session in enumerate(detector.training_history['training_sessions'][-3:], 1):
                print(f"  {i}. {session['date']} | {session['samples']:,} примеров | {', '.join(session['files'])}")
    else:
        print(" Модель не обучена")
        print("\nДля начала работы необходимо:")
        print("  1. Выбрать пункт меню '1 - Первичное обучение'")
        print("  2. Предоставить файл с нормальными системными вызовами")
    
    print("="*60)

def training_mode(detector, incremental=False):
    """Режим обучения или дообучения"""
    mode_name = "ДООБУЧЕНИЕ" if incremental else "ПЕРВИЧНОЕ ОБУЧЕНИЕ"
    
    print("\n" + "="*60)
    print(f"РЕЖИМ: {mode_name} МОДЕЛИ")
    print("="*60)
    
    if incremental and not os.path.exists(detector.model_filename):
        print("\n ОШИБКА: Невозможно дообучить - базовая модель не найдена!")
        print("Сначала выполните первичное обучение (пункт 1)")
        input("\nНажмите Enter для возврата...")
        return
    
    print(f"\nДля обучения необходим файл с логами {'дополнительной ' if incremental else ''}НОРМАЛЬНОЙ работы системы.")
    
    file_path = get_file_path("\n Укажите файл с нормальными данными:")
    
    if file_path is None:
        print(f"\n {mode_name} отменено")
        return
    
    print("\n" + "-"*60)
    print("НАСТРОЙКА ПАРАМЕТРОВ")
    print("-"*60)
    print("Contamination - процент аномалий в обучающей выборке")
    print("  0.05 (5%)  - Рекомендуемая детекция")
    
    while True:
        contamination_input = input("\nВведите contamination [0.05]: ").strip()
        if not contamination_input:
            contamination = 0.05
            break
        try:
            contamination = float(contamination_input)
            if 0.01 < contamination < 0.5:
                break
            else:
                print(" Значение должно быть между 0.01 и 0.5")
        except ValueError:
            print(" Введите корректное число")
    
    print("\n" + "-"*60)
    print(f"Режим: {mode_name}")
    print(f"Файл: {os.path.basename(file_path)}")
    print(f"Contamination: {contamination}")
    print("-"*60)
    confirm = input(f"\nНачать {'дообучение' if incremental else 'обучение'}? (y/n): ").strip().lower()
    
    if confirm == 'y':
        detector.train(
            normal_files=[file_path],
            contamination=contamination,
            incremental=incremental
        )
        input("\nНажмите Enter для продолжения...")
    else:
        print(f"\n {mode_name} отменено")

def scanning_mode(detector):
    """Режим сканирования"""
    print("\n" + "="*60)
    print("РЕЖИМ СКАНИРОВАНИЯ")
    print("="*60)
    
    file_path = get_file_path("\n Укажите файл для сканирования:")
    
    if file_path is None:
        print("\n Сканирование отменено")
        return
    
    print("\n" + "-"*60)
    print("НАСТРОЙКИ СКАНИРОВАНИЯ")
    print("-"*60)
    
    visualize = input("Показать графики? (y/n) [y]: ").strip().lower()
    visualize = visualize != 'n'
    
    save_results = input("Сохранить результаты? (y/n) [y]: ").strip().lower()
    save_results = save_results != 'n'
    
    print("\n" + "-"*60)
    print(f"Файл: {os.path.basename(file_path)}")
    print(f"Визуализация: {'Да' if visualize else 'Нет'}")
    print(f"Сохранение: {'Да' if save_results else 'Нет'}")
    print("-"*60)

    detector.detect(
        file_path=file_path,
        visualize=visualize,
        save_results=save_results
    )

def testing_mode(detector):
    """Режим тестирования модели"""
    print("\n" + "="*60)
    print("РЕЖИМ ТЕСТИРОВАНИЯ МОДЕЛИ")
    print("="*60)
    print("\nЭтот режим используется для оценки качества модели")
    print("когда вы ЗНАЕТЕ, что система чистая или зараженная.")
    
    file_path = get_file_path("\n Укажите файл для тестирования:")
    
    if file_path is None:
        print("\n Тестирование отменено")
        return
    
    print("\n" + "-"*60)
    print("ТИП ТЕСТОВЫХ ДАННЫХ")
    print("-"*60)
    print(" [1] Чистая система (для проверки ложных срабатываний)")
    print(" [2] Зараженная система (для проверки обнаружения)")
    
    choice = input("\nВыберите тип [1-2]: ").strip()
    
    if choice == '1':
        scan_mode = 'clean'
        print("\n✓ Будет проверена точность на чистой системе")
    elif choice == '2':
        scan_mode = 'infected'
        print("\n✓ Будет проверена способность обнаружения руткита")
    else:
        print("\n Неверный выбор!")
        return
    
    print("\n" + "-"*60)
    print("НАСТРОЙКИ")
    print("-"*60)
    
    visualize = input("Показать графики? (y/n) [y]: ").strip().lower()
    visualize = visualize != 'n'
    
    save_results = input("Сохранить результаты? (y/n) [y]: ").strip().lower()
    save_results = save_results != 'n'
    
    confirm = input("\nНачать тестирование? (y/n): ").strip().lower()
    
    if confirm == 'y':
        detector.detect(
            file_path=file_path,
            visualize=visualize,
            save_results=save_results,
            scan_mode=scan_mode
        )
        input("\nНажмите Enter для продолжения...")
    else:
        print("\n Тестирование отменено")

def main():
    """Главная функция"""
    detector = RootkitDetector()
    
    model_exists = os.path.exists(detector.model_filename)
    
    while True:
        
        if model_exists:
            print("\n✓ Обнаружена обученная модель")
            print(f"  Версия: {detector.training_history['version']} | Примеров: {detector.training_history['total_samples']:,}")
        else:
            print("\n Модель не найдена - необходимо первичное обучение")
        
        print_menu()
        
        choice = input("Выберите действие [0-5]: ").strip()
        
        if choice == '1':
            training_mode(detector, incremental=False)
            model_exists = os.path.exists(detector.model_filename)
        
        elif choice == '2':
            training_mode(detector, incremental=True)
            model_exists = os.path.exists(detector.model_filename)
        
        elif choice == '3':
            scanning_mode(detector)
        
        elif choice == '5':
            testing_mode(detector)
        
        elif choice == '4':
            model_info(detector)
            input("\nНажмите Enter для продолжения...")
        
        elif choice == '0':
            print("\n" + "="*60)
            print("Спасибо за использование Rootkit Detector!")
            print("="*60)
            print("\n До свидания!\n")
            break
        
        else:
            print("\n Неверный выбор!")
            input("Нажмите Enter для продолжения...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Программа прервана")
        print(" До свидания!\n")
    except Exception as e:
        print(f"\n ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        input("\nНажмите Enter для выхода...")